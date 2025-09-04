import asyncio
import argparse
import logging
from typing import Dict, List
import time
import os
import glob
from datetime import datetime
import signal

import pandas as pd
import requests

from mm.config import load_config
from mm.market_data import MarketData
from mm.orders import OrdersClient, OrdersEngine, DesiredQuote
from mm.orders import NonRetryableOrderError
from mm.state import StateStore
from mm.strategy import AvellanedaLite, build_layered_quotes
from mm.risk import RiskManager
from mm.selection import SelectionManager
from mm.merge_manager import MergeManager, MergeConfig
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


def _cleanup_old_logs(prefix: str, keep_count: int = 5) -> None:
    """Clean up old log files, keeping only the most recent ones."""
    try:
        log_pattern = f"logs/{prefix}*.log"
        log_files = glob.glob(log_pattern)
        
        if len(log_files) > keep_count:
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old files beyond the keep count
            for old_file in log_files[keep_count:]:
                try:
                    os.remove(old_file)
                    logger.debug("Removed old log file: %s", old_file)
                except Exception as e:
                    logger.warning("Failed to remove old log file %s: %s", old_file, e)
    except Exception as e:
        logger.warning("Failed to cleanup old logs: %s", e)


async def main_async(test_mode: bool = False) -> None:
    # Logging setup
    log_level = logging.DEBUG if test_mode else logging.DEBUG
    logging.getLogger("mm.market_data").setLevel(logging.DEBUG)
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    # Always create logs directory and add file handler
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp for log filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/mm_main_{ts}.log"
    
    # Clear old main log files (keep only the most recent 5)
    _cleanup_old_logs("mm_main_")
    
    # Create file handler and add to handlers
    fh = logging.FileHandler(log_filename)
    log_handlers.append(fh)
    
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=log_handlers)
    
    # Log that file logging has started
    logger.info("Logging to file: %s", log_filename)
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
    # Map any token to its token1 (canonical) for fair price computation
    token_to_token1: Dict[str, str] = {}
    # Track token pairs (token1, token2) for symmetric processing
    token_pairs: List[tuple[str, str]] = []
    cond_col = "condition_id" if "condition_id" in enriched.columns else ("conditionId" if "conditionId" in enriched.columns else None)
    if cond_col is not None:
        for _, row in enriched.iterrows():
            market_hex = str(row.get(cond_col) or "").strip()
            if not market_hex:
                continue
            t1 = str(row.get("token1") or "").strip() if "token1" in enriched.columns else ""
            t2 = str(row.get("token2") or "").strip() if "token2" in enriched.columns else ""
            if t1:
                token_to_market[t1] = market_hex
                token_to_token1[t1] = t1
            if t2:
                token_to_market[t2] = market_hex
                if t1:
                    token_to_token1[t2] = t1
            if t1 and t2:
                token_pairs.append((t1, t2))

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

            def cancel_market_orders(self, market: str | None = None, asset_id: str | None = None) -> None:
                self._logger.info("DRY RUN cancel orders: market=%s asset_id=%s", market, asset_id)

            def get_orders(self) -> list[dict]:
                return []

        orders = _DryRunOrders(logger)
    else:
        if not (cfg.pk and cfg.browser_address):
            logger.error("PK and BROWSER_ADDRESS required for order placement")
            return
        orders = OrdersClient(cfg.clob_base_url, cfg.pk, cfg.browser_address, state)
    # Orders engine for diffing and lifecycle
    engine = OrdersEngine(
        client=orders,  # type: ignore[arg-type]
        tick=0.01,
        partial_fill_pct=50.0,
        order_max_age_sec=cfg.order_max_age_sec,
        requote_mid_ticks=cfg.requote_mid_ticks,
        requote_queue_levels=cfg.requote_queue_levels,
    )

    # Strategy per YES token id
    strategies: Dict[str, AvellanedaLite] = {}
    for token in token_ids:
        strategies[token] = AvellanedaLite(
            alpha_fair=cfg.alpha_fair,
            k_vol=cfg.k_vol,
            k_fee_ticks=cfg.k_fee_ticks,
            inv_gamma=cfg.inv_gamma,
        )

    # Main loop: compute quotes and sync layered orders via engine
    last_quote_ts: Dict[str, float] = {}
    last_mid_seen: Dict[str, float] = {}
    last_fair_seen: Dict[str, float] = {}
    halt_event = asyncio.Event()
    cooldown_until: Dict[str, float] = {}
    _last_heartbeat: float = 0.0

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
        nonlocal token_ids, token_to_market, token_to_token1, token_pairs, strategies
        logger.info("Selection loop started - checking for market changes every 15 minutes")
        while not halt_event.is_set():
            try:
                to_add, to_remove = sel.tick()
                if to_add or to_remove:
                    logger.info("🔄 MARKET SELECTION CHANGE DETECTED - Updating trading configuration")
                    logger.info("Previous active markets: %d", len(token_ids))
                    # Cancel orders for removed tokens before reconfiguring
                    if to_remove:
                        for tok in to_remove:
                            try:
                                orders.cancel_market_orders(asset_id=tok)  # type: ignore[attr-defined]
                                logger.info("Cancelled orders for removed token %s", tok)
                            except Exception:
                                logger.warning("Failed to cancel orders for removed token %s", tok)

                    # Update tokens and subscriptions
                    token_ids = sel.active_tokens
                    logger.info("New active markets: %d", len(token_ids))
                    
                    # Re-enrich to capture latest condition ids
                    _, enr = sel.pull()
                    old_market_count = len(token_to_market)
                    token_to_market.clear()
                    token_to_token1.clear()
                    # Rebuild token pairs from fresh selection
                    new_pairs: List[tuple[str, str]] = []
                    cond_col2 = "condition_id" if "condition_id" in enr.columns else ("conditionId" if "conditionId" in enr.columns else None)
                    if cond_col2 is not None:
                        for _, row in enr.iterrows():
                            mhex = str(row.get(cond_col2) or "").strip()
                            if not mhex:
                                continue
                            t1v = str(row.get("token1") or "").strip() if "token1" in enr.columns else ""
                            t2v = str(row.get("token2") or "").strip() if "token2" in enr.columns else ""
                            if t1v:
                                token_to_market[t1v] = mhex
                                token_to_token1[t1v] = t1v
                            if t2v:
                                token_to_market[t2v] = mhex
                                if t1v:
                                    token_to_token1[t2v] = t1v
                            if t1v and t2v:
                                new_pairs.append((t1v, t2v))
                    token_pairs = new_pairs
                    
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
                    # Prune strategies for removed tokens
                    removed_count = 0
                    for t in list(strategies.keys()):
                        if t not in token_ids:
                            strategies.pop(t, None)
                            removed_count += 1
                    
                    if new_strategies > 0:
                        logger.info("Created %d new trading strategies", new_strategies)
                    if removed_count > 0:
                        logger.info("Pruned %d strategies for removed tokens", removed_count)
                    
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
    # Start merge manager loop
    merge_cfg = MergeConfig(
        merge_scan_interval_sec=cfg.merge_scan_interval_sec,
        min_merge_usdc=cfg.min_merge_usdc,
        merge_chunk_usdc=cfg.merge_chunk_usdc,
        merge_max_retries=cfg.merge_max_retries,
        merge_retry_backoff_ms=cfg.merge_retry_backoff_ms,
        dry_run=bool(cfg.merge_dry_run),
    )
    merge_mgr = MergeManager(merge_cfg)
    merge_task = asyncio.create_task(merge_mgr.run_loop(os.getenv("BROWSER_ADDRESS", "")))
    try:
        while True:
            # Pull live positions and NAV for inventory-aware quoting
            wallet = os.getenv("BROWSER_ADDRESS", "")
            positions_by_token = _fetch_positions_by_token(wallet) if wallet else {}
            nav_usd = _fetch_nav_usd(wallet) if wallet else 0.0

            # Heartbeat every 5s to confirm loop is alive
            try:
                now_hb = time.time()
                if now_hb - _last_heartbeat >= 5.0:
                    logger.debug("MM tick: tokens=%d books=%d", len(token_ids), len(md.books))
                    _last_heartbeat = now_hb
            except Exception:
                pass

            # Process by token pairs to ensure token1/token2 consistency
            # Quote strictly by token pairs to avoid side mixing
            for t1 in list({p[0] for p in token_pairs}):
                # Find paired token2 for this t1 (first match)
                t2 = next((p[1] for p in token_pairs if p[0] == t1), None)
                cid = token_to_market.get(t1) or token_to_market.get(t2 or "") or "-"
                # Token1 book and fair (books are keyed by DECIMAL token ids)
                book1 = md.books.get(t1)
                if not book1:
                    logger.debug("No orderbook for token1=%s (market=%s)", t1, cid)
                    continue
                bb1 = book1.best_bid()
                ba1 = book1.best_ask()
                if bb1 is None and ba1 is None:
                    now_bf = time.time()
                    # throttle backfill to once per 10s per token1
                    globals_dict = globals()
                    if "_last_backfill_at" not in globals_dict:
                        globals_dict["_last_backfill_at"] = {}
                    _lba = globals_dict["_last_backfill_at"]
                    last_ts = _lba.get(t1, 0.0)
                    if now_bf - last_ts > 10.0:
                        try:
                            md.backfill_top_of_book([t1])
                        except Exception:
                            logger.exception("token1 REST backfill failed for %s", t1)
                        _lba[t1] = now_bf
                    # skip this pair for now; next loop should have levels
                    continue
                if bb1 is None and ba1 is None:
                    logger.debug("No best levels for token1=%s", t1)
                    continue
                if bb1 is not None and ba1 is not None:
                    mid1 = (float(bb1) + float(ba1)) / 2.0
                    if bb1 > ba1:
                        low, high = float(ba1), float(bb1)
                        mid1 = min(max(mid1, low), high)
                elif bb1 is not None:
                    mid1 = float(bb1)
                else:
                    mid1 = float(ba1)
                mid1 = float(min(0.99, max(0.01, mid1)))
                mid2 = float(min(0.99, max(0.01, 1.0 - mid1)))
                last_fair_seen[t1] = mid1
                if t2:
                    last_fair_seen[t2] = mid2

                desired_quotes: List[DesiredQuote] = []
                mid_by_token: Dict[str, float] = {}

                # Quote helper for a single token (collect desired quotes)
                def _quote_token(tok: str, fair_mid_val: float) -> None:
                    # Cooldown
                    now_ts = time.time()
                    until = cooldown_until.get(tok)
                    if until is not None and now_ts < until:
                        logger.debug("Cooldown active for token %s for %.1fs", tok, until - now_ts)
                        return
                    bookx = md.books.get(tok)
                    if not bookx:
                        logger.debug("No orderbook for token=%s (market=%s)", tok, cid)
                        return
                    shares = positions_by_token.get(tok, 0.0)
                    inventory_usd = shares * fair_mid_val
                    bankroll = nav_usd if nav_usd > 0 else 1.0
                    inventory_norm = (inventory_usd / bankroll) if bankroll > 0 else 0.0
                    strat = strategies.get(tok) or AvellanedaLite(
                        alpha_fair=cfg.alpha_fair,
                        k_vol=cfg.k_vol,
                        k_fee_ticks=cfg.k_fee_ticks,
                        inv_gamma=cfg.inv_gamma,
                    )
                    strategies[tok] = strat
                    # Drive strategy sigma/microprice off token1 orderbook consistently
                    quote = strat.compute_quote(book1, inventory_norm, fair_hint=fair_mid_val)
                    if quote is None:
                        return
                    # Apply risk multipliers to pricing (spread and reservation price) before layering
                    adj = risk.apply(
                        state=type("S", (), {
                            "fair": fair_mid_val,
                            "sigma": 0.0,
                            "inventory_usd": inventory_usd,
                            "bankroll_usd": nav_usd if nav_usd > 0 else 1.0,
                            "time_to_resolution_sec": 24 * 3600.0,
                            "tick": 0.01,
                        })(),
                        token_id=tok,
                        nav_usd=nav_usd,
                    )
                    try:
                        center0 = (float(quote.bid) + float(quote.ask)) / 2.0
                        delta_r0 = center0 - float(fair_mid_val)
                        h0 = max(0.0, (float(quote.ask) - float(quote.bid)) / 2.0)
                        delta_r1 = delta_r0 * float(adj.gamma_multiplier)
                        h1 = h0 * float(adj.h_multiplier)
                        new_bid = max(0.01, min(0.99, float(fair_mid_val) + delta_r1 - h1))
                        new_ask = max(0.01, min(0.99, float(fair_mid_val) + delta_r1 + h1))
                        quote = type(quote)(bid=float(new_bid), ask=float(new_ask))
                    except Exception:
                        pass
                    lq = build_layered_quotes(
                        base_quote=quote,
                        layers=cfg.order_layers,
                        base_size=cfg.base_size_usd,
                        max_size=cfg.max_size_usd,
                    )
                    if adj.size_multiplier != 1.0:
                        lq = type(lq)(
                            bid_prices=lq.bid_prices,
                            ask_prices=lq.ask_prices,
                            sizes=[s * adj.size_multiplier for s in lq.sizes],
                            timestamp=lq.timestamp,
                        )
                    api_bid = bookx.best_bid()
                    api_ask = bookx.best_ask()
                    logger.info(
                        "Quoting token=%s market=%s mid=%.4f bid=%.4f ask=%.4f layers=%d",
                        tok,
                        cid,
                        fair_mid_val,
                        (api_bid if api_bid is not None else -1.0),
                        (api_ask if api_ask is not None else -1.0),
                        cfg.order_layers,
                    )
                    # Build desired quotes for diffing engine
                    for idx, (bp, sz) in enumerate(zip(lq.bid_prices, lq.sizes)):
                        if sz > 0:
                            desired_quotes.append(DesiredQuote(token_id=tok, side="BUY", price=float(bp), size=float(sz), level=idx))
                    for idx, (ap, sz) in enumerate(zip(lq.ask_prices, lq.sizes)):
                        if sz > 0:
                            desired_quotes.append(DesiredQuote(token_id=tok, side="SELL", price=float(ap), size=float(sz), level=idx))
                    mid_by_token[tok] = float(fair_mid_val)

                _quote_token(t1, mid1)
                if t2:
                    _quote_token(t2, mid2)
                # Sync desired quotes once per loop for both tokens
                try:
                    if not desired_quotes:
                        logger.debug("No desired quotes built this cycle (cooldowns? size=0?). Skipping engine sync.")
                        actions = {"placed": [], "cancelled": [], "replaced": [], "errors": []}
                    else:
                        actions = engine.sync(desired_quotes, mid_by_token)
                    # Aggregate placed USD by side for diagnostics
                    placed = actions.get("placed", []) + actions.get("replaced", [])
                    buy_usd = 0.0
                    sell_usd = 0.0
                    buy_cnt = 0
                    sell_cnt = 0
                    for a in placed:
                        try:
                            side = str(a.get("side", "")).upper()
                            price = float(a.get("price", 0.0))
                            size = float(a.get("size", 0.0))
                            usd = price * size
                            if side == "BUY":
                                buy_usd += usd
                                buy_cnt += 1
                            elif side == "SELL":
                                sell_usd += usd
                                sell_cnt += 1
                        except Exception:
                            continue
                    if buy_cnt or sell_cnt:
                        logger.info("Placed quotes this cycle: BUY count=%d usd=%.2f, SELL count=%d usd=%.2f", buy_cnt, buy_usd, sell_cnt, sell_usd)
                    errs = actions.get("errors", [])
                    if errs:
                        # Log first few errors and set cooldown only for affected tokens
                        for e in errs[:5]:
                            logger.warning("Order error token=%s side=%s price=%.4f size=%.2f type=%s err=%s", e.get("token"), e.get("side"), float(e.get("price", 0.0)), float(e.get("size", 0.0)), e.get("type"), e.get("error"))
                        now_bk = time.time()
                        for e in errs:
                            tok_e = str(e.get("token") or "")
                            if tok_e and str(e.get("type")) == "nonretryable":
                                cooldown_until[tok_e] = now_bk + float(cfg.nonretryable_cooldown_sec)
                    for tok, m in mid_by_token.items():
                        last_quote_ts[tok] = time.time()
                        last_mid_seen[tok] = m
                except NonRetryableOrderError as nre:
                    logger.warning("Non-retryable order error: %s", nre)
                    # Apply cooldown/backoff for all tokens we attempted to quote this cycle
                    now_bk = time.time()
                    for tok in {dq.token_id for dq in desired_quotes}:
                        cooldown_until[tok] = now_bk + float(cfg.nonretryable_cooldown_sec)
                        logger.info("Cooldown applied for token %s for %.1fs due to non-retryable error", tok, float(cfg.nonretryable_cooldown_sec))
                except Exception:
                    logger.exception("Orders engine sync failed")
            # Also process explicit token pairs to ensure token2 mid=1-mid(token1)
            for t1, t2 in token_pairs:
                cid = token_to_market.get(t1) or token_to_market.get(t2) or "-"
                # Only use token1 orderbook to maintain canonical fair
                book1 = md.books.get(t1)
                if not book1:
                    continue
                bb1 = book1.best_bid()
                ba1 = book1.best_ask()
                if bb1 is None and ba1 is None:
                    continue
                if bb1 is not None and ba1 is not None:
                    mid1 = (float(bb1) + float(ba1)) / 2.0
                    if bb1 > ba1:
                        low, high = float(ba1), float(bb1)
                        mid1 = min(max(mid1, low), high)
                elif bb1 is not None:
                    mid1 = float(bb1)
                else:
                    mid1 = float(ba1)
                mid1 = float(min(0.99, max(0.01, mid1)))
                # Token2 mid
                mid2 = float(min(0.99, max(0.01, 1.0 - mid1)))
                last_fair_seen[t1] = mid1
                last_fair_seen[t2] = mid2
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
        try:
            merge_task.cancel()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Market-Making Daemon")
    parser.add_argument("--test", action="store_true", help="Dry run: no orders sent, debug logs to file")
    args = parser.parse_args()
    asyncio.run(main_async(test_mode=bool(args.test)))


if __name__ == "__main__":
    main()
