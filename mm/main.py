import asyncio
import argparse
import logging
from typing import Dict, List
import time
import os
from datetime import datetime
import signal

import pandas as pd
import requests

from mm.config import load_config
from mm.market_data import MarketData
from mm.orders import OrdersClient
from mm.state import StateStore
from mm.strategy import AvellanedaLite, build_layered_quotes, should_requote
from mm.risk import RiskManager
from mm.selection import SelectionManager
from store_selected_markets import read_sheet
from poly_utils.google_utils import get_spreadsheet


logger = logging.getLogger("mm")


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _fetch_positions_by_token(address: str) -> Dict[str, float]:
    """Fetch current positions via Data-API and return token_id -> shares (float)."""
    out: Dict[str, float] = {}
    try:
        url = f"https://data-api.polymarket.com/positions?user={address}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = resp.json() or []
        for row in arr:
            token = str(row.get("token_id") or row.get("asset_id") or row.get("id") or "").strip()
            if not token:
                continue
            shares = row.get("shares")
            if shares is None:
                shares = row.get("balance") or row.get("qty") or 0.0
            try:
                val = float(shares)
            except Exception:
                val = 0.0
            out[token] = val
    except Exception:
        logger.exception("Failed to fetch positions for %s", address)
    return out


def _fetch_nav_usd(address: str) -> float:
    try:
        url = f"https://data-api.polymarket.com/value?user={address}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, dict):
            return float(body.get("value") or 0.0)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, dict) and "value" in first:
                return float(first.get("value") or 0.0)
            try:
                return float(first)
            except Exception:
                return 0.0
        return 0.0
    except Exception:
        logger.exception("Failed to fetch NAV for %s", address)
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
    risk = RiskManager(
        soft_cap_delta_pct=cfg.soft_cap_delta_pct,
        hard_cap_delta_pct=cfg.hard_cap_delta_pct,
        daily_loss_limit_pct=cfg.daily_loss_limit_pct,
    )

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
    ws_task: asyncio.Task | None = None

    def _restart_ws(tokens: List[str]) -> None:
        nonlocal ws_task
        if ws_task is not None:
            try:
                ws_task.cancel()
            except Exception:
                pass
            ws_task = None
        if tokens:
            md.backfill_prices(tokens)
            ws_task = asyncio.create_task(md.run_ws(tokens))
        else:
            logger.warning("No token_ids derived; skipping WS subscription")

    _restart_ws(token_ids)

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
    halt_event = asyncio.Event()

    # Selection supervisor (15 min re-pull)
    sel = SelectionManager(cfg.gamma_base_url, state_store=state)
    
    # Log initial market selection
    initial_tokens, _ = sel.pull()
    logger.info("🎯 Initial market selection: %d markets loaded from sheet", len(initial_tokens))
    if initial_tokens:
        logger.info("Initial markets: %s", initial_tokens[:5])  # Show first 5 for brevity
        if len(initial_tokens) > 5:
            logger.info("... and %d more markets", len(initial_tokens) - 5)

    async def selection_loop() -> None:
        nonlocal token_ids, token_to_market, strategies
        logger.info("Selection loop started - checking for market changes every 15 minutes")
        while not halt_event.is_set():
            try:
                to_add, to_remove = sel.tick()
                if to_add or to_remove:
                    logger.info("🔄 MARKET SELECTION CHANGE DETECTED - Updating trading configuration")
                    logger.info("Previous active markets: %d", len(token_ids))
                    
                    # Update tokens and subscriptions
                    token_ids = sel.active_tokens
                    logger.info("New active markets: %d", len(token_ids))
                    
                    # Re-enrich to capture latest condition ids
                    _, enr = sel.pull()
                    old_market_count = len(token_to_market)
                    token_to_market.clear()
                    cond_col2 = "condition_id" if "condition_id" in enr.columns else ("conditionId" if "conditionId" in enr.columns else None)
                    if cond_col2 is not None:
                        for _, row in enr.iterrows():
                            mhex = str(row.get(cond_col2) or "").strip()
                            if not mhex:
                                continue
                            for tc in ("token1", "token2"):
                                if tc in enr.columns:
                                    tval = str(row.get(tc) or "").strip()
                                    if tval:
                                        token_to_market[tval] = mhex
                    
                    logger.info("Market mappings: %d -> %d", old_market_count, len(token_to_market))
                    
                    # Restart websocket subscriptions
                    logger.info("Restarting websocket subscriptions for %d tokens", len(token_ids))
                    _restart_ws(token_ids)
                    
                    # Ensure strategies exist for all tokens
                    new_strategies = 0
                    for t in token_ids:
                        if t not in strategies:
                            strategies[t] = AvellanedaLite(
                                alpha_fair=cfg.alpha_fair,
                                k_vol=cfg.k_vol,
                                k_fee_ticks=cfg.k_fee_ticks,
                                inv_gamma=cfg.inv_gamma,
                            )
                            new_strategies += 1
                    
                    if new_strategies > 0:
                        logger.info("Created %d new trading strategies", new_strategies)
                    
                    # Log summary of changes
                    logger.info("✅ Market selection update complete:")
                    logger.info("   - Active markets: %d", len(token_ids))
                    logger.info("   - Market mappings: %d", len(token_to_market))
                    logger.info("   - Trading strategies: %d", len(strategies))
                else:
                    logger.debug("No market selection changes detected")
                
                # Log periodic status every hour (4th check)
                if (time.time() - sel.ts) > 3600:  # 1 hour since last change
                    logger.info("📊 Selection status: %d active markets, %d strategies, %d market mappings", 
                               len(token_ids), len(strategies), len(token_to_market))
                
                await asyncio.sleep(900)  # 15 minutes
            except Exception:
                logger.exception("Selection loop error")
                await asyncio.sleep(30)

    # Config hot-reload (SIGHUP)
    def _on_sighup() -> None:
        try:
            _ = load_config()
            logger.info("Config reloaded on SIGHUP")
        except Exception:
            logger.exception("Config reload failed")

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, _on_sighup)
    except Exception:
        pass
    sel_task = asyncio.create_task(selection_loop())
    try:
        while True:
            # Pull live positions and NAV for inventory-aware quoting
            wallet = os.getenv("BROWSER_ADDRESS", "")
            positions_by_token = _fetch_positions_by_token(wallet) if wallet else {}
            nav_usd = _fetch_nav_usd(wallet) if wallet else 0.0

            for token in token_ids:
                market_id = token_to_market.get(token)
                if not market_id:
                    logger.debug("No market mapping for token %s; skipping", token)
                    continue
                book = md.books.get(market_id)
                if not book:
                    continue
                # Inventory-aware: estimate inventory_usd from shares * mid
                shares = positions_by_token.get(token, 0.0)
                mid = book.mid() or 0.0
                inventory_usd = shares * mid
                q = strategies[token].compute_quote(book, 0.0)
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
                # Apply risk manager multipliers to spreads/gamma at the strategy level if using advanced
                adj = risk.apply(
                    state=type("S", (), {
                        "fair": mid_now or mid,
                        "sigma": 0.0,
                        "inventory_usd": inventory_usd,
                        "bankroll_usd": nav_usd if nav_usd > 0 else 1.0,
                        "time_to_resolution_sec": 24 * 3600.0,
                        "tick": 0.01,
                    })(),
                    token_id=token,
                    nav_usd=nav_usd,
                )
                # Reflect size multiplier from risk
                if adj.size_multiplier != 1.0:
                    lq = type(lq)(
                        bid_prices=lq.bid_prices,
                        ask_prices=lq.ask_prices,
                        sizes=[s * adj.size_multiplier for s in lq.sizes],
                        timestamp=lq.timestamp,
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
        halt_event.set()
        try:
            sel_task.cancel()
        except Exception:
            pass
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
