import asyncio
import argparse
import logging
from typing import Dict, List
import time
import os
from datetime import datetime

import pandas as pd
import requests

from mm.config import load_config
from mm.market_data import MarketData
from mm.orders import OrdersClient
from mm.state import StateStore
from mm.strategy import AvellanedaLite, build_layered_quotes, apply_inventory_risk, should_requote
from store_selected_markets import read_sheet
from poly_utils.google_utils import get_spreadsheet


logger = logging.getLogger("mm")


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def normalize_selected_markets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns from Selected Markets without filtering.
    """
    df = df.copy()
    col_map = {
        "Liquidity": "liquidity",
        "Volume_24h": "volume24h",
        "Volume_7d": "volume7d",
        "Volume_30d": "volume30d",
        "market_id": "market_id",
        "yes_token_id": "yes_token_id",
        "no_token_id": "no_token_id",
        "token1": "token1",
        "token2": "token2",
        "condition_id": "condition_id",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = df[src]
    for c in ["liquidity", "volume24h", "volume7d", "volume30d"]:
        series = df[c] if c in df.columns else pd.Series([0.0] * len(df), index=df.index)
        df[c] = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # Ensure token/ids are strings
    for c in ["token1", "token2", "yes_token_id", "no_token_id", "condition_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def enrich_gamma(df: pd.DataFrame, gamma_base: str) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        # Build targeted list of clob token ids strictly from token1/token2
        token_ids: List[str] = []
        for col in ("token1", "token2"):
            if col in df.columns:
                token_ids.extend([str(x) for x in df[col].dropna().astype(str).tolist() if str(x)])
        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            # No ids available; return unchanged (no fallback)
            return df

        # Query Gamma in chunks using clob_token_ids filter
        def _chunks(lst: List[str], n: int) -> List[List[str]]:
            return [lst[i:i + n] for i in range(0, len(lst), n)]

        items: List[dict] = []
        for chunk in _chunks(token_ids, 200):
            params = "&".join([f"clob_token_ids={requests.utils.quote(t)}" for t in chunk])
            url = f"{gamma_base}/markets?{params}"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            batch = resp.json() or []
            if batch:
                items.extend(batch)
        gdf = pd.DataFrame(items)
        if gdf.empty:
            return df

        # Join back to original by token id
        left = df.copy()
        # Join strictly on token1
        left_key = "token1"
        if left_key not in left.columns:
            return df
        left[left_key] = left[left_key].astype(str)

        # Gamma clobTokenIds may be list or comma-separated; derive first token as yes side proxy
        def _first_token(x) -> str:
            if isinstance(x, list) and x:
                return str(x[0])
            if isinstance(x, str) and x:
                return x.split(",")[0].strip()
            return ""

        gdf[left_key] = gdf.get("clobTokenIds", "").map(_first_token)
        merged = left.merge(gdf, on=left_key, how="left", suffixes=("", "_g"))
        return merged
    except Exception:
        return df


async def main_async(test_mode: bool = False) -> None:
    # Logging setup
    log_level = logging.DEBUG if test_mode else logging.INFO
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    if test_mode:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"logs/mm_test_{ts}.log")
        log_handlers.append(fh)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=log_handlers)
    # Suppress noisy third-party logs; keep our package logs
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    cfg = load_config()
    state = StateStore()

    # Read Selected Markets
    ss = get_spreadsheet(read_only=False)
    selected = read_sheet(ss, cfg.selected_sheet_name)
    if selected.empty:
        logger.info("Selected Markets empty; exiting")
        return

    # Normalize (no filtering in the MM daemon)
    normalized = normalize_selected_markets(selected)
    if normalized.empty:
        logger.info("Selected Markets had no rows; exiting")
        return

    # Enrich via Gamma
    enriched = enrich_gamma(normalized, cfg.gamma_base_url)

    # Build token->market mapping (use condition_id or Gamma conditionId)
    token_to_market: Dict[str, str] = {}
    cond_col = "condition_id" if "condition_id" in enriched.columns else ("conditionId" if "conditionId" in enriched.columns else None)
    if cond_col is not None:
        for _, row in enriched.iterrows():
            market_hex = str(row.get(cond_col) or "").strip()
            if not market_hex:
                continue
            for tok_col in ("token1", "token2"):
                if tok_col in enriched.columns:
                    tok = str(row.get(tok_col) or "").strip()
                    if tok:
                        token_to_market[tok] = market_hex

    # Build token id list strictly from token1/token2
    token_ids: List[str] = []
    for col in ("token1", "token2"):
        if col in enriched.columns:
            token_ids.extend([str(x) for x in enriched[col].dropna().astype(str).tolist() if str(x)])
    token_ids = list(dict.fromkeys(token_ids))

    # Market data
    md = MarketData(cfg.clob_ws_url, cfg.clob_base_url)
    if token_ids:
        md.backfill_prices(token_ids)
        ws_task = asyncio.create_task(md.run_ws(token_ids))
    else:
        logger.warning("No token_ids derived from Selected Markets/Gamma; skipping WS subscription")
        ws_task = None

    # Orders client (dry-run in test mode)
    if test_mode:
        class _DryRunOrders:
            def __init__(self, logger_: logging.Logger) -> None:
                self._logger = logger_

            def place_order(self, token_id: str, side: str, price: float, size: float) -> dict:
                self._logger.info("DRY RUN order: token=%s side=%s price=%.4f size=%.2f", token_id, side, price, size)
                return {"dry_run": True, "token_id": token_id, "side": side, "price": price, "size": size}

        orders = _DryRunOrders(logger)
    else:
        if not (cfg.pk and cfg.browser_address):
            logger.error("PK and BROWSER_ADDRESS required for order placement")
            return
        orders = OrdersClient(cfg.clob_base_url, cfg.pk, cfg.browser_address, state)

    # Strategy per YES token id
    strategies: Dict[str, AvellanedaLite] = {}
    for token in token_ids:
        strategies[token] = AvellanedaLite(
            alpha_fair=cfg.alpha_fair,
            k_vol=cfg.k_vol,
            k_fee_ticks=cfg.k_fee_ticks,
            inv_gamma=cfg.inv_gamma,
        )

    # Main loop: compute quotes and place layered orders (skeleton)
    last_quote_ts: Dict[str, float] = {}
    last_mid_seen: Dict[str, float] = {}
    try:
        while True:
            for token in token_ids:
                market_id = token_to_market.get(token)
                if not market_id:
                    logger.debug("No market mapping for token %s; skipping", token)
                    continue
                book = md.books.get(market_id)
                if not book:
                    continue
                # Inventory norm: placeholder 0 for now; integrate from state later
                q_norm = 0.0
                q = strategies[token].compute_quote(book, q_norm)
                if q is None:
                    continue
                mid_now = book.mid()
                if not should_requote(
                    last_mid=last_mid_seen.get(token),
                    current_mid=mid_now,
                    last_timestamp=last_quote_ts.get(token),
                    order_max_age_sec=cfg.order_max_age_sec,
                    requote_mid_ticks=cfg.requote_mid_ticks,
                ):
                    continue
                lq = build_layered_quotes(
                    base_quote=q,
                    layers=cfg.order_layers,
                    base_size=cfg.base_size_usd,
                    max_size=cfg.max_size_usd,
                )
                lq = apply_inventory_risk(
                    quotes=lq,
                    inventory_norm=q_norm,
                    soft_cap=cfg.soft_cap_delta_pct,
                    hard_cap=cfg.hard_cap_delta_pct,
                )
                logger.info(
                    "Quoting token=%s market=%s mid=%.4f bid=%.4f ask=%.4f layers=%d",
                    token,
                    market_id,
                    (mid_now if mid_now is not None else -1.0),
                    q.bid,
                    q.ask,
                    cfg.order_layers,
                )
                try:
                    for price, size in zip(lq.bid_prices, lq.sizes):
                        if size > 0:
                            orders.place_order(token, "BUY", price, size)
                    for price, size in zip(lq.ask_prices, lq.sizes):
                        if size > 0:
                            orders.place_order(token, "SELL", price, size)
                    last_quote_ts[token] = time.time()
                    if mid_now is not None:
                        last_mid_seen[token] = mid_now
                except Exception:
                    logger.exception("Order placement failed for token %s", token)
            await asyncio.sleep(1.0)
    finally:
        if ws_task is not None:
            ws_task.cancel()
            try:
                await ws_task
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Market-Making Daemon")
    parser.add_argument("--test", action="store_true", help="Dry run: no orders sent, debug logs to file")
    args = parser.parse_args()
    asyncio.run(main_async(test_mode=bool(args.test)))


if __name__ == "__main__":
    main()
