from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import Request, Depends
from starlette.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import sqlite3

try:
    import pandas as pd  # type: ignore
except Exception as exc:
    raise RuntimeError("pandas must be installed; check requirements.txt") from exc

# Prefer existing client/state if present
try:
    from poly_data import global_state  # type: ignore
    from poly_data.polymarket_client import PolymarketClient  # type: ignore
except Exception as exc:
    raise RuntimeError("poly_data package not found in workspace") from exc

def _csv_env_set(name: str) -> set[str]:
    return set(x.strip() for x in (os.getenv(name, "") or "").split(",") if x.strip())

ALLOWED_GH_IDS = _csv_env_set("ALLOWED_GH_IDS")

# Use existing trade/PNL helpers from wallet_pnl
try:
    from wallet_pnl import (  # type: ignore
        fetch_activity_trades,
        fetch_reward_activities,
        compute_earnings_by_market,
        build_pnl_rows_from_activity,
        get_best_price,
    )
except Exception as exc:
    raise RuntimeError("wallet_pnl.fetch_activity_trades not found") from exc
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _ensure_client() -> PolymarketClient:
    if getattr(global_state, "client", None) is None:
        logger.info("Initializing PolymarketClient for dashboard")
        global_state.client = PolymarketClient(initialize_api=True)
    return global_state.client  # type: ignore[return-value]


def _df_to_records(df: pd.DataFrame, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    if fields is not None:
        existing = [c for c in fields if c in df.columns]
        df = df[existing]
    # Replace NaNs for JSON
    return pd.DataFrame(df).where(pd.notna(df), None).to_dict(orient="records")


app = FastAPI(title="Poly Maker Dashboard", version="0.1.0", root_path="/poly-maker")

# after `app = FastAPI(..., root_path="/poly-maker")`
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "CHANGE_ME"),
    session_cookie="poly_maker_sess",
    https_only=True,
    same_site="lax",
    max_age=60*60*12,  # 12h
)
oauth = OAuth()
oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    authorize_url="https://github.com/login/oauth/authorize",
    access_token_url="https://github.com/login/oauth/access_token",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email", "token_placement": "header"},
)

def require_user(req: Request):
    user = req.session.get("user")
    if not user:
        next_url = req.url.path
        raise HTTPException(status_code=307, headers={"Location": f"{req.app.root_path}/login?next={next_url}"})
    if ALLOWED_GH_IDS and str(user.get("id", "")) not in ALLOWED_GH_IDS:
        req.session.clear()
        raise HTTPException(status_code=403, detail="Not authorized")
    return user

@app.get("/login", include_in_schema=False)
async def login(request: Request):
    redirect_uri = os.getenv("OAUTH_REDIRECT_URI") or str(request.url_for("auth_callback"))
    return await oauth.github.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback", include_in_schema=False, name="auth_callback")
async def auth_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get("user", token=token)
    resp.raise_for_status()
    u = resp.json()

    gh_id = str(u.get("id", ""))

    # >>> ADD THIS BLOCK <<<
    if ALLOWED_GH_IDS and gh_id not in ALLOWED_GH_IDS:
        request.session.clear()
        raise HTTPException(status_code=403, detail="Not authorized")
    # <<< END ADD >>>

    request.session["user"] = {
        "id": gh_id,
        "login": u.get("login"),
        "name": u.get("name") or u.get("login"),
        "avatar_url": u.get("avatar_url"),
    }
    next_url = request.query_params.get("next") or f"{request.app.root_path}/"
    return RedirectResponse(next_url)


@app.get("/logout", include_in_schema=False)
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(f"{request.app.root_path}/")


# Static/UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(STATIC_DIR):
    logger.warning("Static directory %s does not exist", STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Removed Google Sheets-based metadata mapping. Resolution now relies on Gamma API exclusively.


def _fetch_markets_metadata(
    market_ids: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Fetch market name and token outcomes for given market IDs via public APIs.

    Returns:
        market_to_name, token_to_outcome
    """
    headers = {
        "User-Agent": "poly-maker-dashboard/1.0",
        "Accept": "application/json",
    }
    market_to_name: Dict[str, str] = {}
    token_to_outcome: Dict[str, str] = {}

    def harvest(items: List[Dict[str, Any]]) -> None:
        for m in items:
            try:
                mid = str(m.get("id") or m.get("market") or m.get("conditionId") or "")
                if mid:
                    name = (m.get("question") or m.get("title") or m.get("name") or m.get("slug") or m.get("questionTitle") or "")
                    market_to_name[mid] = str(name)
                for key in ("outcomes", "tokens", "outcomeTokens"):
                    if key in m and isinstance(m[key], list):
                        for o in m[key]:
                            if isinstance(o, dict):
                                tid = o.get("token_id") or o.get("tokenId") or o.get("id")
                                out = o.get("outcome") or o.get("answer") or o.get("title")
                                if tid:
                                    token_to_outcome[str(tid)] = str(out or "")
            except Exception:
                continue

    urls_group = [
        "https://data-api.polymarket.com/markets?ids=",
        "https://clob.polymarket.com/markets?ids=",
    ]
    # Try batch queries first
    if market_ids:
        ids_param = ",".join(sorted(set([str(x) for x in market_ids if x])))
        for base in urls_group:
            try:
                resp = requests.get(base + ids_param, headers=headers, timeout=10)
                if not resp.ok:
                    continue
                data = resp.json()
                items: List[Dict[str, Any]]
                if isinstance(data, dict) and "markets" in data:
                    items = data.get("markets", [])
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]
                harvest(items)
                # If we got at least one name, stop
                if market_to_name:
                    return market_to_name, token_to_outcome
            except Exception:
                continue

    # Fall back to per-market requests
    for mid in market_ids:
        for base in [
            "https://data-api.polymarket.com/markets/",
            "https://clob.polymarket.com/markets/",
        ]:
            try:
                resp = requests.get(base + str(mid), headers=headers, timeout=6)
                if not resp.ok:
                    continue
                data = resp.json()
                items: List[Dict[str, Any]]
                if isinstance(data, dict) and "markets" in data:
                    items = data.get("markets", [])
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]
                harvest(items)
                if str(mid) in market_to_name:
                    break
            except Exception:
                continue

    return market_to_name, token_to_outcome


def _fetch_assets_metadata(
    token_ids: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Fetch metadata for given asset (token) IDs using Gamma API.

    Returns:
        token_to_market_name, token_to_outcome, token_to_market_id
    """
    headers = {
        "User-Agent": "poly-maker-dashboard/1.0",
        "Accept": "application/json",
    }
    token_to_market_name: Dict[str, str] = {}
    token_to_outcome: Dict[str, str] = {}
    token_to_market_id: Dict[str, str] = {}

    def harvest_from_gamma(items: List[Dict[str, Any]]) -> None:
        for item in items:
            try:
                # Extract token ID from various possible fields
                tid = item.get("tokenId") or item.get("id") or item.get("token_id") or item.get("asset_id")
                if tid is None:
                    continue
                tid_s = str(tid)
                
                # Extract market/condition ID
                mid = item.get("conditionId") or item.get("marketId") or item.get("market_id")
                
                # Extract market name/question
                name = (item.get("question") or item.get("title") or item.get("marketTitle") or item.get("market_title") or item.get("name"))
                
                # Extract outcome/answer
                out = (item.get("outcome") or item.get("answer") or item.get("outcomeTitle") or item.get("outcome_title") or item.get("title"))
                
                if mid:
                    token_to_market_id[tid_s] = str(mid)
                if name:
                    token_to_market_name[tid_s] = str(name)
                if out:
                    token_to_outcome[tid_s] = str(out)
                    
                logger.debug(f"Processed token {tid_s}: market='{name}', outcome='{out}', conditionId='{mid}'")
            except Exception as e:
                logger.debug(f"Error processing item: {e}")
                continue

    if not token_ids:
        logger.warning("No token IDs provided to _fetch_assets_metadata")
        return token_to_market_name, token_to_outcome, token_to_market_id

    # Use Gamma API instead of CLOB
    gamma_base_url = "https://gamma-api.polymarket.com"
    
    logger.info(f"Fetching assets metadata for {len(token_ids)} tokens from Gamma API")
    
    # Process tokens in batches to avoid very long URLs
    batch_size = 10
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i + batch_size]
        ids_param = ",".join(batch)
        
        try:
            # Try to fetch from Gamma API using the tokens endpoint
            url = f"{gamma_base_url}/tokens?ids={ids_param}"
            logger.info(f"Trying Gamma API batch {i//batch_size + 1}: {url}")
            
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.ok:
                data = resp.json()
                items = data if isinstance(data, list) else data.get("tokens", []) if isinstance(data, dict) else []
                
                logger.info(f"Received {len(items)} items from Gamma API for batch {i//batch_size + 1}")
                harvest_from_gamma(items)
                
                if token_to_market_name:
                    batch_names = len([k for k in token_to_market_name.keys() if k in batch])
                    logger.info(f"Successfully fetched metadata from Gamma API batch {i//batch_size + 1}: {batch_names} market names")
            else:
                logger.warning(f"Gamma API request failed for batch {i//batch_size + 1}: Status {resp.status_code}")
                
        except Exception as e:
            logger.warning(f"Error fetching from Gamma API batch {i//batch_size + 1}: {e}")
            continue

    # If we still don't have market names, try to fetch them using condition IDs
    if token_to_market_id and not token_to_market_name:
        logger.info("No market names from tokens endpoint, trying condition endpoint...")
        condition_ids = list(set(token_to_market_id.values()))
        
        for i in range(0, len(condition_ids), batch_size):
            batch = condition_ids[i:i + batch_size]
            ids_param = ",".join(batch)
            
            try:
                url = f"{gamma_base_url}/conditions?ids={ids_param}"
                logger.info(f"Trying Gamma conditions API batch {i//batch_size + 1}: {url}")
                
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.ok:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("conditions", []) if isinstance(data, dict) else []
                    
                    for item in items:
                        try:
                            condition_id = str(item.get("id") or item.get("conditionId"))
                            if condition_id:
                                name = item.get("question") or item.get("title") or item.get("name")
                                if name:
                                    # Find all tokens that map to this condition
                                    for tid, cid in token_to_market_id.items():
                                        if str(cid) == condition_id:
                                            token_to_market_name[tid] = str(name)
                                            logger.debug(f"Mapped token {tid} to market name '{name}' via condition {condition_id}")
                        except Exception as e:
                            logger.debug(f"Error processing condition item: {e}")
                            continue
                            
            except Exception as e:
                logger.warning(f"Error fetching from Gamma conditions API batch {i//batch_size + 1}: {e}")
                continue

    logger.info(f"Final Gamma API results: {len(token_to_market_name)} market names, {len(token_to_outcome)} outcomes, {len(token_to_market_id)} market IDs")
    return token_to_market_name, token_to_outcome, token_to_market_id


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/pnl", include_in_schema=False)
def pnl_page(user=Depends(require_user)) -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "pnl.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="pnl.html not found")
    return FileResponse(index_path)


def _get_db_path() -> str:
    """Return path to odds SQLite DB with env override."""
    try:
        env_path = (os.getenv("ODDS_DB_PATH") or "").strip()
        if env_path:
            return env_path
    except Exception:
        pass
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root_dir, "data", "odds.db")


def _query_db(sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        logger.warning("SQLite DB not found at %s", db_path)
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.exception("DB query failed: %s", str(exc))
        return []


@app.get("/odds", include_in_schema=False)
def odds_page(req: Request, user=Depends(require_user)) -> RedirectResponse:
    return RedirectResponse(url=f"{req.app.root_path}/static/odds.html")

@app.get("/odds/", include_in_schema=False)
def odds_page_slash(req: Request, user=Depends(require_user)) -> RedirectResponse:
    return RedirectResponse(url=f"{req.app.root_path}/static/odds.html")

@app.get("/odds.html", include_in_schema=False)
def odds_page_html(req: Request, user=Depends(require_user)) -> RedirectResponse:
    return RedirectResponse(url=f"{req.app.root_path}/static/odds.html")

@app.get("/api/odds")
def get_odds(limit: Optional[int] = None, user=Depends(require_user)) -> Dict[str, Any]:
    """Return markets with p_model and hourly p_actuals for both tokens (and yes/no when available)."""
    markets = _query_db(
        "SELECT question, title, category, token1, token2, p_model, confidence FROM markets"
    )
    if not markets:
        return {"data": []}

    out: List[Dict[str, Any]] = []
    for m in markets:
        token1 = str(m.get("token1", "") or "")
        token2 = str(m.get("token2", "") or "")
        if not token1 and not token2:
            snapshots: List[Dict[str, Any]] = []
        else:
            if limit is not None and limit > 0:
                sql = (
                    "SELECT ts_utc, p_token1, p_token2, p_yes, p_no FROM market_prices_hourly "
                    "WHERE token1 = ? AND token2 = ? ORDER BY ts_utc DESC LIMIT ?"
                )
                rows = _query_db(sql, (token1, token2, int(limit)))
            else:
                sql = (
                    "SELECT ts_utc, p_token1, p_token2, p_yes, p_no FROM market_prices_hourly "
                    "WHERE token1 = ? AND token2 = ? ORDER BY ts_utc DESC"
                )
                rows = _query_db(sql, (token1, token2))
            snapshots = rows or []

        out.append({
            "market": (m.get("question") or m.get("title") or ""),
            "category": m.get("category"),
            "p_model": m.get("p_model"),
            "confidence": m.get("confidence"),
            "token1": token1,
            "token2": token2,
            "snapshots": snapshots,
        })

    return {"data": out}


@app.get("/api/orders")
def get_open_orders(user=Depends(require_user)) -> Dict[str, Any]:
    client = _ensure_client()
    try:
        orders_df = client.get_all_orders()
        logger.info(f"Fetched {len(orders_df) if orders_df is not None else 0} orders")
        
        if orders_df is None or orders_df.empty:
            return {"data": []}
        # Filter to open/live if status exists
        if "status" in orders_df.columns:
            orders_df = orders_df[orders_df["status"].isin(["LIVE", "OPEN"])].copy()
            logger.info(f"Filtered to {len(orders_df)} open/live orders")
        # Ensure target columns exist for safe masking/assignment
        if "market_name" not in orders_df.columns:
            orders_df["market_name"] = ""
        if "outcome" not in orders_df.columns:
            orders_df["outcome"] = ""
        
        # Log the columns we have to debug
        logger.info(f"Order columns: {list(orders_df.columns)}")
        
        # 0) Use existing title column if available (most reliable)
        if "title" in orders_df.columns and orders_df["title"].notna().any():
            logger.info("Found title column, using it for market names")
            orders_df["market_name"] = orders_df["title"].fillna("")
            logger.info(f"Set {orders_df['title'].notna().sum()} market names from title column")
        
        # 1) Assets API by token IDs - this is the secondary method
        # Look for asset column with different possible names (match positions order)
        asset_col = None
        for col in ["asset", "token_id", "tokenId", "asset_id"]:
            if col in orders_df.columns:
                asset_col = col
                break
        
        if asset_col:
            try:
                token_ids = sorted(set(orders_df[asset_col].dropna().astype(str)))
                logger.info(f"Found {len(token_ids)} unique token IDs in column '{asset_col}': {token_ids[:5]}...")
            except Exception:
                token_ids = []
            if token_ids:
                tm_f, to_f, _ = _fetch_assets_metadata(token_ids)
                logger.info(f"Fetched metadata for {len(tm_f)} tokens, {len(to_f)} outcomes")
                if tm_f:
                    orders_df["market_name"] = orders_df[asset_col].astype(str).map(tm_f).fillna(orders_df["market_name"])
                    logger.info(f"Mapped {orders_df['market_name'].notna().sum()} market names from assets API")
                if to_f:
                    orders_df["outcome"] = orders_df[asset_col].astype(str).map(to_f).fillna(orders_df["outcome"])
                    logger.info(f"Mapped {orders_df['outcome'].notna().sum()} outcomes from assets API")
        else:
            logger.warning("No asset/token column found in orders data. Available columns: %s", list(orders_df.columns))
        
        # 2) Try to use conditionId column if available (this is more reliable)
        if "conditionId" in orders_df.columns and orders_df["market_name"].eq("").any():
            missing_conditions = sorted(set(orders_df.loc[orders_df["market_name"].eq("") & orders_df["conditionId"].notna(), "conditionId"].astype(str)))
            if missing_conditions:
                logger.info(f"Found conditionId column, fetching market metadata for {len(missing_conditions)} conditions")
                m2n_f, t2o_f = _fetch_markets_metadata(missing_conditions)
                if m2n_f:
                    orders_df.loc[orders_df["market_name"].eq("") & orders_df["conditionId"].notna(), "market_name"] = (
                        orders_df.loc[orders_df["market_name"].eq("") & orders_df["conditionId"].notna(), "conditionId"].astype(str).map(m2n_f).fillna("")
                    )
                    logger.info(f"Updated {len(m2n_f)} market names using conditionId")
                if t2o_f and asset_col:
                    mask = orders_df["outcome"].eq("") & orders_df[asset_col].notna()
                    orders_df.loc[mask, "outcome"] = orders_df.loc[mask, asset_col].astype(str).map(t2o_f).fillna("")
                    logger.info(f"Updated {len(t2o_f)} outcomes using conditionId")
        
        # 3) Markets API by market IDs for any still-missing names (fallback)
        if "market_name" in orders_df.columns and orders_df["market_name"].eq("").any() and "market" in orders_df.columns:
            missing_mids = sorted(set(orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market"].astype(str)))
            if missing_mids:
                logger.info(f"Fetching market metadata for {len(missing_mids)} missing markets")
                m2n_f, t2o_f = _fetch_markets_metadata(missing_mids)
                if m2n_f:
                    orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market_name"] = (
                        orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market"].astype(str).map(m2n_f).fillna("")
                    )
                    logger.info(f"Updated {len(m2n_f)} market names from markets API")
                if t2o_f and asset_col:
                    mask = orders_df["outcome"].eq("") & orders_df[asset_col].notna()
                    orders_df.loc[mask, "outcome"] = orders_df.loc[mask, asset_col].astype(str).map(t2o_f).fillna("")
                    logger.info(f"Updated {len(t2o_f)} outcomes from markets API")
        
        # Positions path has no extra ad-hoc token fallback; keep logic aligned
        # No Google Sheets fallback; rely on Gamma/data APIs only
        
        total_orders = len(orders_df)
        orders_with_names = orders_df["market_name"].notna().sum()
        orders_with_outcomes = orders_df["outcome"].notna().sum()
        logger.info(f"Final enrichment results: {orders_with_names}/{total_orders} orders have market names, {orders_with_outcomes}/{total_orders} have outcomes")
        
        # Log which tokens still don't have market names for debugging
        if asset_col and orders_with_names < total_orders:
            missing_tokens = orders_df.loc[orders_df["market_name"].eq(""), asset_col].astype(str).unique()
            logger.warning(f"Tokens still missing market names: {[t[:20] + '...' for t in missing_tokens]}")
            logger.info("Some tokens still missing market names after API lookups")
        
        # Normalize fields commonly used
        wanted = [
            "id",
            "market_name",
            "outcome",
            "side",
            "price",
            "original_size",
            "size_matched",
            "status",
            "created_at",
        ]
        result = {"data": _df_to_records(orders_df, wanted)}
        logger.info(f"Returning {len(result['data'])} enriched orders")
        return result
    except Exception as exc:
        logger.exception("Error fetching open orders: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch orders")


@app.get("/api/orders/debug")
def get_orders_debug(user=Depends(require_user)) -> Dict[str, Any]:
    """Debug endpoint to see raw order data and API-based metadata enrichment (Gamma/markets)."""
    client = _ensure_client()
    try:
        orders_df = client.get_all_orders()
        debug_info = {
            "total_orders": len(orders_df) if orders_df is not None else 0,
            "columns": list(orders_df.columns) if orders_df is not None else [],
            "sample_data": []
        }
        
        if orders_df is not None and not orders_df.empty:
            # Filter to open/live if status exists
            if "status" in orders_df.columns:
                orders_df = orders_df[orders_df["status"].isin(["LIVE", "OPEN"])].copy()
                debug_info["filtered_orders"] = len(orders_df)
            
            # Show first few rows for debugging
            for i, row in orders_df.head(3).iterrows():
                debug_info["sample_data"].append({
                    "index": i,
                    "id": str(row.get("id", "")),
                    "asset_id": str(row.get("asset_id", "")),
                    "market": str(row.get("market", "")),
                    "side": str(row.get("side", "")),
                    "price": row.get("price", ""),
                    "size": row.get("original_size", "")
                })
            
            # Test metadata fetching
            asset_col = None
            for col in ["asset_id", "asset", "token_id", "tokenId"]:
                if col in orders_df.columns:
                    asset_col = col
                    break
            
            if asset_col:
                token_ids = sorted(set(orders_df[asset_col].dropna().astype(str)))[:5]  # First 5 tokens
                debug_info["test_token_ids"] = token_ids
                debug_info["asset_column_used"] = asset_col
                
                # Test assets API
                tm_f, to_f, _ = _fetch_assets_metadata(token_ids)
                debug_info["assets_api_results"] = {
                    "market_names": dict(list(tm_f.items())[:3]),  # First 3 results
                    "outcomes": dict(list(to_f.items())[:3])
                }
                
                # No Google Sheets mappings (removed); include empty placeholders
                debug_info["sheet_mappings"] = {
                    "token_to_market": {},
                    "token_to_outcome": {},
                    "market_to_name": {}
                }
            else:
                debug_info["asset_column_used"] = "None found"
                debug_info["available_columns"] = list(orders_df.columns)
        
        return debug_info
    except Exception as exc:
        logger.exception("Error in orders debug endpoint: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch orders debug info")


@app.get("/api/positions/debug")
def get_positions_debug(user=Depends(require_user)) -> Dict[str, Any]:
    """Debug endpoint to see raw position data and API-based metadata enrichment (Gamma/markets)."""
    client = _ensure_client()
    try:
        pos_df = client.get_all_positions()
        debug_info = {
            "total_positions": len(pos_df) if pos_df is not None else 0,
            "columns": list(pos_df.columns) if pos_df is not None else [],
            "sample_data": []
        }
        
        if pos_df is not None and not pos_df.empty:
            # Show first few rows for debugging
            for i, row in pos_df.head(3).iterrows():
                debug_info["sample_data"].append({
                    "index": i,
                    "asset": str(row.get("asset", "")),
                    "market": str(row.get("market", "")),
                    "size": row.get("size", ""),
                    "avgPrice": row.get("avgPrice", ""),
                    "curPrice": row.get("curPrice", ""),
                    "percentPnl": row.get("percentPnl", "")
                })
            
            # Test metadata fetching
            asset_col = None
            for col in ["asset", "token_id", "tokenId", "asset_id"]:
                if col in pos_df.columns:
                    asset_col = col
                    break
            
            if asset_col:
                token_ids = sorted(set(pos_df[asset_col].dropna().astype(str)))[:5]  # First 5 tokens
                debug_info["test_token_ids"] = token_ids
                debug_info["asset_column_used"] = asset_col
                
                # Test assets API
                tm_f, to_f, _ = _fetch_assets_metadata(token_ids)
                debug_info["assets_api_results"] = {
                    "market_names": dict(list(tm_f.items())[:3]),  # First 3 results
                    "outcomes": dict(list(to_f.items())[:3])
                }
                
                # No Google Sheets mappings (removed); include empty placeholders
                debug_info["sheet_mappings"] = {
                    "token_to_market": {},
                    "token_to_outcome": {},
                    "market_to_name": {}
                }
            else:
                debug_info["asset_column_used"] = "None found"
                debug_info["available_columns"] = list(pos_df.columns)
        
        return debug_info
    except Exception as exc:
        logger.exception("Error in debug endpoint: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch debug info")


@app.get("/api/positions")
def get_positions(user=Depends(require_user)) -> Dict[str, Any]:
    client = _ensure_client()
    try:
        pos_df = client.get_all_positions()
        logger.info(f"Fetched {len(pos_df) if pos_df is not None else 0} positions")
        
        # Enrich with market name and outcome (API-first)
        if pos_df is not None and not pos_df.empty:
            # Ensure target columns exist for safe masking/assignment
            if "market_name" not in pos_df.columns:
                pos_df["market_name"] = ""
            if "outcome" not in pos_df.columns:
                pos_df["outcome"] = ""
            
            # 0) Use existing title column if available (most reliable)
            if "title" in pos_df.columns and pos_df["title"].notna().any():
                logger.info("Found title column, using it for market names")
                pos_df["market_name"] = pos_df["title"].fillna("")
                logger.info(f"Set {pos_df['title'].notna().sum()} market names from title column")
            
            # Log the columns we have to debug
            logger.info(f"Position columns: {list(pos_df.columns)}")
            
            # 1) Assets API by token IDs - this is the secondary method
            # Look for asset column with different possible names
            asset_col = None
            for col in ["asset", "token_id", "tokenId", "asset_id"]:
                if col in pos_df.columns:
                    asset_col = col
                    break
            
            if asset_col:
                try:
                    token_ids = sorted(set(pos_df[asset_col].dropna().astype(str)))
                    logger.info(f"Found {len(token_ids)} unique token IDs in column '{asset_col}': {token_ids[:5]}...")
                    
                    if token_ids:
                        tm_f, to_f, _ = _fetch_assets_metadata(token_ids)
                        logger.info(f"Fetched metadata for {len(tm_f)} tokens, {len(to_f)} outcomes")
                        
                        if tm_f:
                            # Map market names using token IDs; preserve existing names if mapping missing
                            pos_df["market_name"] = (
                                pos_df[asset_col].astype(str).map(tm_f).fillna(pos_df["market_name"])
                            )
                            logger.info(f"Mapped {pos_df['market_name'].notna().sum()} market names from assets API (preserving existing)")
                        
                        if to_f:
                            # Map outcomes using token IDs; preserve existing outcomes if mapping missing
                            pos_df["outcome"] = (
                                pos_df[asset_col].astype(str).map(to_f).fillna(pos_df["outcome"])
                            )
                            logger.info(f"Mapped {pos_df['outcome'].notna().sum()} outcomes from assets API (preserving existing)")
                except Exception as e:
                    logger.warning(f"Error fetching assets metadata: {e}")
            else:
                logger.warning("No asset/token column found in positions data. Available columns: %s", list(pos_df.columns))

            # 2) Try to use conditionId column if available (this is more reliable)
            if "conditionId" in pos_df.columns and pos_df["market_name"].eq("").any():
                missing_conditions = sorted(set(pos_df.loc[pos_df["market_name"].eq("") & pos_df["conditionId"].notna(), "conditionId"].astype(str)))
                if missing_conditions:
                    logger.info(f"Found conditionId column, fetching market metadata for {len(missing_conditions)} conditions")
                    m2n_f, t2o_f = _fetch_markets_metadata(missing_conditions)
                    if m2n_f:
                        pos_df.loc[pos_df["market_name"].eq("") & pos_df["conditionId"].notna(), "market_name"] = (
                            pos_df.loc[pos_df["market_name"].eq("") & pos_df["conditionId"].notna(), "conditionId"].astype(str).map(m2n_f).fillna("")
                        )
                        logger.info(f"Updated {len(m2n_f)} market names using conditionId")
                    if t2o_f and asset_col:
                        mask = pos_df["outcome"].eq("") & pos_df[asset_col].notna()
                        pos_df.loc[mask, "outcome"] = pos_df.loc[mask, asset_col].astype(str).map(t2o_f).fillna("")
                        logger.info(f"Updated {len(t2o_f)} outcomes using conditionId")

            # 3) Markets API by market IDs for any still-missing names (fallback)
            if "market_name" in pos_df.columns and pos_df["market_name"].eq("").any() and "market" in pos_df.columns:
                missing_mids = sorted(set(pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market"].astype(str)))
                if missing_mids:
                    logger.info(f"Fetching market metadata for {len(missing_mids)} missing markets")
                    m2n_f, t2o_f = _fetch_markets_metadata(missing_mids)
                    if m2n_f:
                        pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market_name"] = (
                            pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market"].astype(str).map(m2n_f).fillna("")
                        )
                        logger.info(f"Updated {len(m2n_f)} market names from markets API")
                    if t2o_f and asset_col:
                        mask = pos_df["outcome"].eq("") & pos_df[asset_col].notna()
                        pos_df.loc[mask, "outcome"] = pos_df.loc[mask, asset_col].astype(str).map(t2o_f).fillna("")
                        logger.info(f"Updated {len(t2o_f)} outcomes from markets API")
            
            # No Google Sheets fallback; rely on Gamma/data APIs only
            
            if "market" in pos_df.columns:
                # Fill in missing market names using market IDs (if previously fetched via markets API)
                mask = pos_df["market_name"].eq("") & pos_df["market"].notna()
                if mask.any():
                    logger.debug("Some market names still missing after API enrichment")
        
        wanted = ["market_name", "outcome", "size", "avgPrice", "curPrice", "percentPnl"]
        result = {"data": _df_to_records(pos_df, wanted)}
        logger.info(f"Returning {len(result['data'])} enriched positions")
        return result
    except Exception as exc:
        logger.exception("Error fetching positions: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch positions")


@app.get("/api/trades")
def get_recent_trades(limit: int = 10, page: int = 1, user=Depends(require_user)) -> Dict[str, Any]:
    wallet = (os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "").strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="BROWSER_WALLET/BROWSER_ADDRESS not configured")
    try:
        activity_df = fetch_activity_trades(wallet, per_page_limit=max(10, limit))
        if activity_df is None or activity_df.empty:
            return {"data": [], "page": 1, "limit": limit, "total": 0}
        # sort newest first
        ts_col = None
        for c in ("timestamp", "ts", "created_at", "createdAt"):
            if c in activity_df.columns:
                ts_col = c
                break
        if ts_col:
            activity_df = activity_df.sort_values(ts_col, ascending=False)
            # add readable iso time
            try:
                activity_df["datetime"] = pd.to_datetime(activity_df[ts_col], unit="s", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                activity_df["datetime"] = ""
        # Prefer 'title'/'outcome' from activity if present
        if "title" in activity_df.columns:
            activity_df["market_name"] = activity_df["title"].astype(str)
        else:
            token_col = None
            for c in ("asset", "token_id", "tokenId", "asset_id"):
                if c in activity_df.columns:
                    token_col = c
                    break
            if token_col is not None:
                # Resolve market names via Gamma API using tokens present on this page
                token_ids = list(sorted(set(activity_df[token_col].dropna().astype(str))))[:20]
                tm_f, _, _ = _fetch_assets_metadata(token_ids)
                activity_df["market_name"] = activity_df[token_col].astype(str).map(tm_f).fillna("")
        if "outcome" not in activity_df.columns or activity_df["outcome"].isna().all():
            token_col = None
            for c in ("asset", "token_id", "tokenId", "asset_id"):
                if c in activity_df.columns:
                    token_col = c
                    break
            if token_col is not None:
                token_ids = list(sorted(set(activity_df[token_col].dropna().astype(str))))[:20]
                _, t2o_page, _ = _fetch_assets_metadata(token_ids)
                activity_df["outcome"] = activity_df[token_col].astype(str).map(t2o_page).fillna("")
        # As last resort, try fetch by market if available
        if activity_df.get("market_name") is not None and activity_df["market_name"].eq("").any():
            mid_col = None
            for c in ("market", "conditionId", "id"):
                if c in activity_df.columns:
                    mid_col = c
                    break
            if mid_col is not None:
                missing_mids = sorted(set(activity_df.loc[activity_df["market_name"].eq("") & activity_df[mid_col].notna(), mid_col].astype(str)))
                if missing_mids:
                    m2n_f, t2o_f = _fetch_markets_metadata(missing_mids)
                    if m2n_f:
                        mask = activity_df["market_name"].eq("") & activity_df[mid_col].notna()
                        activity_df.loc[mask, "market_name"] = activity_df.loc[mask, mid_col].astype(str).map(m2n_f).fillna("")
        # Select core fields
        wanted = [
            "market_name",
            "outcome",
            "side",
            "size",
            "price",
            "datetime",
        ]
        # Pagination
        try:
            p = max(1, int(page))
        except Exception:
            p = 1
        try:
            limit_num = max(1, int(limit))
        except Exception:
            limit_num = 10
        start = (p - 1) * limit_num
        end = start + limit_num
        total = int(len(activity_df))
        records = _df_to_records(activity_df, wanted)[start:end]
        return {"data": records, "page": p, "limit": limit_num, "total": total}
    except Exception as exc:
        logger.exception("Error fetching trades: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch trades")


def _compute_current_pnl(wallet: str) -> Dict[str, Any]:
    """Compute realized, unrealized, and total PnL grouped by market/outcome.

    Returns a dict with keys: realized, unrealized, earnings, total, rows.
    Each row contains: market, yes_no, position_size, realized, unrealized, earnings, pnl.
    """
    try:
        activity_df = fetch_activity_trades(wallet, per_page_limit=500)
    except Exception as exc:
        logger.exception("Failed to fetch activity trades: %s", str(exc))
        activity_df = None

    try:
        rewards_df = fetch_reward_activities(wallet, per_page_limit=500)
    except Exception as exc:
        logger.info("Failed to fetch reward activities: %s", str(exc))
        rewards_df = None

    if activity_df is None or getattr(activity_df, "empty", True):
        return {
            "realized": 0.0,
            "unrealized": 0.0,
            "earnings": 0.0,
            "total": 0.0,
            "rows": [],
        }

    try:
        pnl_rows = build_pnl_rows_from_activity(activity_df)
    except Exception as exc:
        logger.exception("Failed to map activity to PnL rows: %s", str(exc))
        pnl_rows = None

    if pnl_rows is None or getattr(pnl_rows, "empty", True):
        return {
            "realized": 0.0,
            "unrealized": 0.0,
            "earnings": 0.0,
            "total": 0.0,
            "rows": [],
        }

    # Ensure needed columns and types
    try:
        pnl_rows["shares"] = pd.to_numeric(pnl_rows.get("shares", 0), errors="coerce").fillna(0.0)
        pnl_rows["price"] = pd.to_numeric(pnl_rows.get("price", 0), errors="coerce").fillna(0.0)
    except Exception:
        pass

    # Group keys
    group_keys: List[str] = ["market"]
    if "yes/no" in pnl_rows.columns:
        group_keys.append("yes/no")

    # Position size and realized pnl (cash flow sum)
    summary = (
        pnl_rows.groupby(group_keys, dropna=False)[["shares", "price"]]
        .sum()
        .reset_index()
        .rename(columns={"shares": "position_size", "price": "realized"})
    )

    # Map group -> representative token_id via latest row
    try:
        pnl_rows["date_ts"] = pd.to_datetime(pnl_rows.get("date"), errors="coerce").astype("int64") // 10**9
        latest_by_key = pnl_rows.sort_values("date_ts").groupby(group_keys).tail(1)
    except Exception:
        latest_by_key = pnl_rows.groupby(group_keys).tail(1)

    token_map: Dict[Tuple[str, str], str] = {}
    use_two_keys = len(group_keys) == 2
    for _, r in latest_by_key.iterrows():
        m = str(r.get("market", ""))
        y = str(r.get("yes/no", "")) if use_two_keys else ""
        tok = str(r.get("token_id", ""))
        if m and tok:
            token_map[(m, y)] = tok

    # Compute unrealized using best price for closing side
    summary["unrealized"] = 0.0
    for idx, row in summary.iterrows():
        m = str(row.get("market", ""))
        y = str(row.get("yes/no", "")) if use_two_keys else ""
        size = float(row.get("position_size", 0.0) or 0.0)
        tok = token_map.get((m, y))
        if not tok or size == 0.0:
            continue
        side = "sell" if size > 0 else "buy"
        try:
            best = get_best_price(tok, side)
        except Exception:
            best = 0.0
        summary.at[idx, "unrealized"] = abs(best * size)

    # Earnings (rewards) by market title
    try:
        earnings_df = compute_earnings_by_market(rewards_df) if rewards_df is not None else pd.DataFrame()
    except Exception:
        earnings_df = pd.DataFrame()
    if not earnings_df.empty:
        summary = summary.merge(earnings_df.rename(columns={"Earnings": "earnings"}), on="market", how="left")
    if "earnings" not in summary.columns:
        summary["earnings"] = 0.0

    # Total per row and totals
    summary["pnl"] = sum([
        pd.to_numeric(summary.get("realized", 0.0), errors="coerce").fillna(0.0),
        pd.to_numeric(summary.get("unrealized", 0.0), errors="coerce").fillna(0.0),
        pd.to_numeric(summary.get("earnings", 0.0), errors="coerce").fillna(0.0),
    ])

    realized_total = float(pd.to_numeric(summary.get("realized", 0.0), errors="coerce").fillna(0.0).sum())
    unrealized_total = float(pd.to_numeric(summary.get("unrealized", 0.0), errors="coerce").fillna(0.0).sum())
    earnings_total = float(pd.to_numeric(summary.get("earnings", 0.0), errors="coerce").fillna(0.0).sum())
    total = float(pd.to_numeric(summary.get("pnl", 0.0), errors="coerce").fillna(0.0).sum())

    # Prepare rows for JSON
    fields = ["market", "position_size", "realized", "unrealized", "earnings", "pnl"]
    if use_two_keys:
        fields.insert(1, "yes/no")
    rows = _df_to_records(summary, fields)

    return {
        "realized": realized_total,
        "unrealized": unrealized_total,
        "earnings": earnings_total,
        "total": total,
        "rows": rows,
    }


@app.get("/api/pnl")
def get_pnl(user=Depends(require_user)) -> Dict[str, Any]:
    wallet = (os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "").strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="BROWSER_WALLET/BROWSER_ADDRESS not configured")
    try:
        result = _compute_current_pnl(wallet)
        return result
    except Exception as exc:
        logger.exception("Error computing PnL: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute PnL")


@app.get("/api/pnl/trades")
def get_pnl_trades(limit: int = 100, page: int = 1, user=Depends(require_user)) -> Dict[str, Any]:
    """Return per-trade cash flows derived from activity for PnL accounting."""
    wallet = (os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "").strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="BROWSER_WALLET/BROWSER_ADDRESS not configured")
    try:
        activity_df = fetch_activity_trades(wallet, per_page_limit=max(100, int(limit)))
        if activity_df is None or activity_df.empty:
            return {"data": [], "page": 1, "limit": limit, "total": 0}

        pnl_rows = build_pnl_rows_from_activity(activity_df)
        if pnl_rows is None or pnl_rows.empty:
            return {"data": [], "page": 1, "limit": limit, "total": 0}

        # Prepare fields and formatting
        try:
            pnl_rows["date"] = pd.to_datetime(pnl_rows.get("date"), errors="coerce")
            pnl_rows["datetime"] = pnl_rows["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pnl_rows["datetime"] = ""

        pnl_rows = pnl_rows.rename(columns={"yes/no": "outcome"}).fillna({"outcome": ""})

        wanted_cols = [
            "market",
            "outcome",
            "side",
            "shares",
            "open_price",
            "price",  # cash flow (+sell, -buy)
            "datetime",
        ]

        # Sort newest first
        try:
            pnl_rows = pnl_rows.sort_values("date", ascending=False)
        except Exception:
            pass

        # Pagination
        try:
            page_num = max(1, int(page))
        except Exception:
            page_num = 1
        try:
            limit_num = max(1, int(limit))
        except Exception:
            limit_num = 100
        start = (page_num - 1) * limit_num
        end = start + limit_num
        total = int(len(pnl_rows))
        records = _df_to_records(pnl_rows, wanted_cols)[start:end]
        # Rename price -> cash_flow for clarity in API response
        for r in records:
            if "price" in r:
                r["cash_flow"] = r.pop("price")
        return {"data": records, "page": page_num, "limit": limit_num, "total": total}
    except Exception as exc:
        logger.exception("Error fetching per-trade PnL: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch per-trade PnL")
