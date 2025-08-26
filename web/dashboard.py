import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests

try:
    import pandas as pd  # type: ignore
except Exception as exc:
    raise RuntimeError("pandas must be installed; check requirements.txt") from exc

# Prefer existing client/state if present
try:
    from poly_data import global_state  # type: ignore
    from poly_data.polymarket_client import PolymarketClient  # type: ignore
    from poly_data.utils import get_sheet_df  # type: ignore
except Exception as exc:
    raise RuntimeError("poly_data package not found in workspace") from exc

# Use existing trade fetcher from wallet_pnl
try:
    from wallet_pnl import fetch_activity_trades  # type: ignore
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


def _ensure_sheet_loaded() -> None:
    try:
        df = getattr(global_state, "df", None)
        if df is None or getattr(df, "empty", True):
            logger.info("Loading market config from Google Sheet (read-only)")
            sheet_df, sheet_params = get_sheet_df(read_only=True)
            global_state.df = sheet_df
            global_state.params = sheet_params
    except Exception as exc:
        logger.info("Sheet metadata unavailable: %s", str(exc))


def _df_to_records(df: pd.DataFrame, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    if fields is not None:
        existing = [c for c in fields if c in df.columns]
        df = df[existing]
    # Replace NaNs for JSON
    return pd.DataFrame(df).where(pd.notna(df), None).to_dict(orient="records")


app = FastAPI(title="Poly Maker Dashboard", version="0.1.0")


# Static/UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(STATIC_DIR):
    logger.warning("Static directory %s does not exist", STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _build_metadata_maps() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build maps for enriching rows with market name and outcome using existing state.

    Returns:
        token_to_market_name, token_to_outcome, market_to_name
    """
    token_to_market: Dict[str, str] = {}
    token_to_outcome: Dict[str, str] = {}
    market_to_name: Dict[str, str] = {}

    try:
        _ensure_sheet_loaded()
        df = getattr(global_state, "df", None)
        if df is not None and not df.empty:  # type: ignore[attr-defined]
            # Expected columns: question, condition_id, token1, token2, answer1, answer2
            for _, r in df.iterrows():
                q = str(r.get("question", ""))
                m = str(r.get("condition_id", ""))
                t1 = str(r.get("token1", ""))
                t2 = str(r.get("token2", ""))
                a1 = str(r.get("answer1", ""))
                a2 = str(r.get("answer2", ""))
                if m:
                    market_to_name[m] = q
                if t1:
                    token_to_market[t1] = q
                    token_to_outcome[t1] = a1
                if t2:
                    token_to_market[t2] = q
                    token_to_outcome[t2] = a2
    except Exception:
        # Keep empty maps if df unavailable
        pass

    return token_to_market, token_to_outcome, market_to_name


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
                    name = (
                        m.get("question")
                        or m.get("title")
                        or m.get("name")
                        or m.get("slug")
                        or m.get("questionTitle")
                        or ""
                    )
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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/orders")
def get_open_orders() -> Dict[str, Any]:
    client = _ensure_client()
    try:
        orders_df = client.get_all_orders()
        if orders_df is None or orders_df.empty:
            return {"data": []}
        # Filter to open/live if status exists
        if "status" in orders_df.columns:
            orders_df = orders_df[orders_df["status"].isin(["LIVE", "OPEN"])].copy()
        # Enrich with market name and outcome using mapping
        t2m, t2o, m2n = _build_metadata_maps()
        if "asset_id" in orders_df.columns:
            orders_df["market_name"] = orders_df["asset_id"].astype(str).map(t2m).fillna("")
            orders_df["outcome"] = orders_df["asset_id"].astype(str).map(t2o).fillna("")
        if "market" in orders_df.columns and ("market_name" not in orders_df.columns or orders_df["market_name"].eq("").all()):
            orders_df["market_name"] = orders_df["market"].astype(str).map(m2n).fillna(orders_df.get("market_name", ""))
        # If still missing names, try remote fetch by market ids
        if "market_name" in orders_df.columns and orders_df["market_name"].eq("").any() and "market" in orders_df.columns:
            missing_mids = sorted(set(orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market"].astype(str)))
            if missing_mids:
                m2n_f, t2o_f = _fetch_markets_metadata(missing_mids)
                if m2n_f:
                    orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market_name"] = (
                        orders_df.loc[orders_df["market_name"].eq("") & orders_df["market"].notna(), "market"].astype(str).map(m2n_f).fillna("")
                    )
                if t2o_f and "asset_id" in orders_df.columns:
                    mask = orders_df["outcome"].eq("") & orders_df["asset_id"].notna()
                    orders_df.loc[mask, "outcome"] = orders_df.loc[mask, "asset_id"].astype(str).map(t2o_f).fillna("")
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
        return {"data": _df_to_records(orders_df, wanted)}
    except Exception as exc:
        logger.exception("Error fetching open orders: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch orders")


@app.get("/api/positions")
def get_positions() -> Dict[str, Any]:
    client = _ensure_client()
    try:
        pos_df = client.get_all_positions()
        # Enrich with market name and outcome
        t2m, t2o, m2n = _build_metadata_maps()
        if pos_df is not None and not pos_df.empty:
            if "asset" in pos_df.columns:
                pos_df["market_name"] = pos_df["asset"].astype(str).map(t2m).fillna("")
                pos_df["outcome"] = pos_df["asset"].astype(str).map(t2o).fillna("")
            if "market" in pos_df.columns and ("market_name" not in pos_df.columns or pos_df["market_name"].eq("").all()):
                pos_df["market_name"] = pos_df["market"].astype(str).map(m2n).fillna(pos_df.get("market_name", ""))
            # Fetch missing names remotely if needed
            if "market_name" in pos_df.columns and pos_df["market_name"].eq("").any() and "market" in pos_df.columns:
                missing_mids = sorted(set(pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market"].astype(str)))
                if missing_mids:
                    m2n_f, t2o_f = _fetch_markets_metadata(missing_mids)
                    if m2n_f:
                        pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market_name"] = (
                            pos_df.loc[pos_df["market_name"].eq("") & pos_df["market"].notna(), "market"].astype(str).map(m2n_f).fillna("")
                        )
                    if t2o_f and "asset" in pos_df.columns:
                        mask = pos_df["outcome"].eq("") & pos_df["asset"].notna()
                        pos_df.loc[mask, "outcome"] = pos_df.loc[mask, "asset"].astype(str).map(t2o_f).fillna("")
        wanted = ["market_name", "outcome", "size", "avgPrice", "curPrice", "percentPnl"]
        return {"data": _df_to_records(pos_df, wanted)}
    except Exception as exc:
        logger.exception("Error fetching positions: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch positions")


@app.get("/api/trades")
def get_recent_trades(limit: int = 10, page: int = 1) -> Dict[str, Any]:
    wallet = (os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "").strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="BROWSER_WALLET/BROWSER_ADDRESS not configured")
    try:
        activity_df = fetch_activity_trades(wallet, per_page_limit=max(10, limit))
        if activity_df is None or activity_df.empty:
            return {"data": []}
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
        # Enrich with market name and outcome if available
        t2m, t2o, _ = _build_metadata_maps()
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
                activity_df["market_name"] = activity_df[token_col].astype(str).map(t2m).fillna("")
        if "outcome" not in activity_df.columns or activity_df["outcome"].isna().all():
            token_col = None
            for c in ("asset", "token_id", "tokenId", "asset_id"):
                if c in activity_df.columns:
                    token_col = c
                    break
            if token_col is not None:
                activity_df["outcome"] = activity_df[token_col].astype(str).map(t2o).fillna("")
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
            l = max(1, int(limit))
        except Exception:
            l = 10
        start = (p - 1) * l
        end = start + l
        total = int(len(activity_df))
        records = _df_to_records(activity_df, wanted)[start:end]
        return {"data": records, "page": p, "limit": l, "total": total}
    except Exception as exc:
        logger.exception("Error fetching trades: %s", str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch trades")


