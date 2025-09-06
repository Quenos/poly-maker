import asyncio
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, List
import time
import os
 
import signal

import requests

from mm.config import load_config
from mm.market_data import MarketData
from mm.orders import OrdersClient, OrdersEngine, DesiredQuote, Side, SyncActions
from mm.orders import NonRetryableOrderError
from mm.state import StateStore
from mm.strategy import AvellanedaLite
from mm.risk import RiskManager
from mm.selection import SelectionManager
from mm.merge_manager import MergeManager, MergeConfig
from web3 import Web3
from poly_data.abis import erc20_abi
from poly_data.polymarket_client import PolymarketClient

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

def _fetch_positions_by_token_via_client(client: PolymarketClient) -> Dict[str, float]:
    """Fetch current positions via PolymarketClient.get_all_positions and return token_id -> shares (float)."""
    out: Dict[str, float] = {}
    try:
        df = client.get_all_positions()
        if df is None:
            return out
        # Determine token id column
        tok_col = None
        for c in ("asset", "token_id", "tokenId", "asset_id", "id"):
            if c in df.columns:
                tok_col = c
                break
        if tok_col is None or "size" not in df.columns:
            return out
        for _, row in df.iterrows():
            try:
                tok = str(row.get(tok_col) or "").strip()
                if not tok:
                    continue
                sz = float(row.get("size") or 0.0)
                out[tok] = sz
            except Exception:
                continue
    except Exception:
        logger.exception("Failed to fetch positions via client")
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


# Position value (excluding cash) via Data API
def _fetch_positions_value_usd(address: str) -> float:
    try:
        url = f"https://data-api.polymarket.com/value?user={address}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        # The API returns a dict with 'value' representing position value (per spec used here)
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
        logger.exception("Failed to fetch positions value for %s", address)
        return 0.0

# Direct on-chain USDC balance (cash only)
def _fetch_usdc_balance(address: str) -> float:
    try:
        rpc = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        web3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        usdc = web3.eth.contract(
            address=Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
            abi=erc20_abi,
        )
        bal = usdc.functions.balanceOf(Web3.to_checksum_address(address)).call()
        return float(bal) / 10**6
    except Exception:
        logger.exception("Failed to fetch USDC balance for %s", address)
        return 0.0

# Legacy log cleanup helper removed; rotating handler handles retention


async def main_async(test_mode: bool = False, debug_logging: bool = False) -> None:
    # Bootstrap logging (will finalize level after loading config)
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]

    # Always create logs directory and add timed rotating file handler
    os.makedirs("logs", exist_ok=True)
    # We need cfg to get dynamic paths, but cfg loads after logging. Use defaults for bootstrap
    file_path = "logs/mm_main.log"
    # Start fresh each run: truncate existing default log file
    try:
        if os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8"):
                pass
    except Exception:
        pass
    backups = 5
    fh = TimedRotatingFileHandler(file_path, when="midnight", backupCount=backups, encoding="utf-8")
    log_handlers.append(fh)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=log_handlers)

    # Log that file logging has started
    logger.info("Logging to rotating file: %s (daily, keep %d backups)", file_path, backups)
    # Suppress noisy third-party logs; keep our package logs
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    cfg = load_config()
    # If config overrides logging file/backups, update handlers now
    try:
        new_file = getattr(cfg, "log_file", file_path)
        new_backups = int(getattr(cfg, "log_rotation_backups", backups) or backups)
        if new_file != file_path or new_backups != backups:
            for h in list(log_handlers):
                if isinstance(h, TimedRotatingFileHandler):
                    logging.getLogger().removeHandler(h)
                    log_handlers.remove(h)
            # If log file path changed via config, truncate new file so each run starts fresh
            try:
                if os.path.exists(new_file):
                    with open(new_file, "w", encoding="utf-8"):
                        pass
            except Exception:
                pass
            fh2 = TimedRotatingFileHandler(new_file, when="midnight", backupCount=new_backups, encoding="utf-8")
            logging.getLogger().addHandler(fh2)
            log_handlers.append(fh2)
            file_path = new_file
            backups = new_backups
    except Exception:
        pass

    # Finalize log level: prefer Settings sheet, overridden only by --debug flag
    try:
        level_name = str(getattr(cfg, "log_level", "INFO")).upper()
        final_level = logging.DEBUG if debug_logging else getattr(logging, level_name, logging.INFO)
        logging.getLogger().setLevel(final_level)
        # Apply same to key module loggers
        logging.getLogger("mm.market_data").setLevel(final_level)
        # Log key risk/cap settings at startup for visibility
        logger.info(
            "Settings: max_position_shares=%s min_buy_price=%.2f order_layers=%d base_size_usd=%.2f",
            str(getattr(cfg, "max_position_shares", None)),
            float(getattr(cfg, "min_buy_price", 0.15)),
            int(getattr(cfg, "order_layers", 3)),
            float(getattr(cfg, "base_size_usd", 300.0)),
        )
    except Exception:
        pass
    state = StateStore()
    # Initialize Polymarket client once for consistent positions view
    try:
        pm_client = PolymarketClient(initialize_api=True)
    except Exception:
        pm_client = None  # Fallback to data-api path
    risk = RiskManager(
        soft_cap_delta_pct=cfg.soft_cap_delta_pct,
        hard_cap_delta_pct=cfg.hard_cap_delta_pct,
        daily_loss_limit_pct=cfg.daily_loss_limit_pct,
    )

    # Initial selection via SelectionManager (single source of truth)
    sel = SelectionManager(cfg.gamma_base_url, state_store=state, sheet_name=cfg.selected_sheet_name)
    initial_tokens, enriched = await asyncio.to_thread(sel.pull)
    logger.info("🎯 Initial market selection: %d markets loaded from sheet", len(initial_tokens))
    if initial_tokens:
        logger.info("Initial markets: %s", initial_tokens[:5])
        if len(initial_tokens) > 5:
            logger.info("... and %d more markets", len(initial_tokens) - 5)
    if len(initial_tokens) == 0:
        logger.info("Selected Markets empty; exiting")
        return

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

    # Track tokens where BUY placement is suspended by sheet flag
    suspended_tokens: set[str] = set()
    try:
        if "suspended" in enriched.columns:
            for _, row in enriched.iterrows():
                try:
                    is_susp = bool(row.get("suspended", False))
                except Exception:
                    is_susp = False
                if is_susp:
                    t1s = str(row.get("token1") or "").strip()
                    t2s = str(row.get("token2") or "").strip()
                    if t1s:
                        suspended_tokens.add(t1s)
                    if t2s:
                        suspended_tokens.add(t2s)
    except Exception:
        logger.exception("Failed to parse suspended flags from sheet")

    if suspended_tokens:
        try:
            logger.info("Sheet suspension active for %d tokens; will cancel existing buys and skip new BUY placements", len(suspended_tokens))
        except Exception:
            pass

    # Build token id list strictly from token1/token2
    token_ids: List[str] = list(initial_tokens)

    # Market data
    md = MarketData(cfg.clob_ws_url, cfg.clob_base_url)
    ws_task: asyncio.Task | None = None

    def _restart_ws(tokens: List[str]) -> None:
        nonlocal ws_task
        # Mark a cold-start/backfill window annotation for diagnostics
        try:
            logger.info("coldstart_window begin: tokens=%d", len(tokens))
        except Exception:
            pass
        if ws_task is not None:
            try:
                ws_task.cancel()
            except Exception:
                pass
            ws_task = None
        if tokens:
            md.backfill_prices(tokens)

            async def _ws_runner(tok_list: List[str]) -> None:
                try:
                    await md.run_ws(tok_list)
                except Exception as e:
                    # Treat normal/expected closes as info; others as warnings
                    msg = str(e)
                    if "NORMAL_CLOSURE" in msg or "ConnectionClosed" in type(e).__name__:
                        logger.info("Websocket closed: %s", msg)
                    else:
                        logger.warning("Websocket error: %s", msg)
            ws_task = asyncio.create_task(_ws_runner(tokens))
        else:
            logger.warning("No token_ids derived; skipping WS subscription")
        try:
            logger.info("coldstart_window end: tokens=%d", len(tokens))
        except Exception:
            pass

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
        tick=float(getattr(cfg, "price_tick", 0.01)),
        partial_fill_pct=50.0,
        order_max_age_sec=cfg.order_max_age_sec,
        requote_mid_ticks=cfg.requote_mid_ticks,
        requote_queue_levels=cfg.requote_queue_levels,
    )

    # On startup, proactively cancel any existing orders on suspended tokens (clears BUYs immediately)
    if suspended_tokens:
        for tok in sorted(suspended_tokens):
            try:
                orders.cancel_market_orders(asset_id=tok, reason="suspended")  # type: ignore[attr-defined]
                logger.info("Cancelled existing orders for suspended token %s", tok)
            except Exception:
                logger.warning("Failed to cancel existing orders for suspended token %s", tok)

    # Strategy per YES token id
    strategies: Dict[str, AvellanedaLite] = {}
    for token in token_ids:
        strategies[token] = AvellanedaLite(
            alpha_fair=cfg.alpha_fair,
            k_vol=cfg.k_vol,
            k_fee_ticks=cfg.k_fee_ticks,
            inv_gamma=cfg.inv_gamma,
            tick=float(getattr(cfg, "price_tick", 0.01)),
        )
    # Log strategy configuration parameters for analysis
    try:
        logger.debug(
            "AvellanedaLite params: alpha_fair=%s k_vol=%s k_fee_ticks=%s inv_gamma=%s order_layers=%s base_size_usd=%s max_size_usd=%s price_tick=%s min_buy_price=%s max_position_shares=%s requote_mid_ticks=%s order_max_age_sec=%s",
            getattr(cfg, "alpha_fair", None),
            getattr(cfg, "k_vol", None),
            getattr(cfg, "k_fee_ticks", None),
            getattr(cfg, "inv_gamma", None),
            getattr(cfg, "order_layers", None),
            getattr(cfg, "base_size_usd", None),
            getattr(cfg, "max_size_usd", None),
            getattr(cfg, "price_tick", None),
            getattr(cfg, "min_buy_price", None),
            getattr(cfg, "max_position_shares", None),
            getattr(cfg, "requote_mid_ticks", None),
            getattr(cfg, "order_max_age_sec", None),
        )
    except Exception:
        pass

    # Main loop: compute quotes and sync layered orders via engine
    last_quote_ts: Dict[str, float] = {}
    last_mid_seen: Dict[str, float] = {}
    last_fair_seen: Dict[str, float] = {}
    halt_event = asyncio.Event()
    cooldown_until: Dict[str, float] = {}
    # Throttle cancellations when position exceeds max cap to avoid spamming API
    overcap_cancel_until: Dict[str, float] = {}
    _last_heartbeat: float = 0.0
    backfill_last_at: Dict[str, float] = {}

    def _compute_mid(best_bid: float | None, best_ask: float | None, tick: float = 0.01) -> float | None:
        if best_bid is None and best_ask is None:
            return None
        if best_bid is not None and best_ask is not None:
            mid = (float(best_bid) + float(best_ask)) / 2.0
            if best_bid > best_ask:
                low, high = float(best_ask), float(best_bid)
                mid = min(max(mid, low), high)
        elif best_bid is not None:
            mid = float(best_bid)
        else:
            mid = float(best_ask)
        return float(min(0.99, max(0.01, mid)))

    # Selection supervisor (periodic re-pull) handled by existing SelectionManager instance

    async def selection_loop() -> None:
        nonlocal token_ids, token_to_market, token_to_token1, token_pairs, strategies, suspended_tokens
        try:
            logger.info(
                "Selection loop started - checking for market changes every %d seconds",
                int(getattr(cfg, "selection_loop_sec", 300))
            )
        except Exception:
            logger.info("Selection loop started - checking for market changes periodically")
        while not halt_event.is_set():
            try:
                # Single sheet read per cycle to honor SELECTION_LOOP_SEC
                try:
                    tokens_latest, enr = await asyncio.to_thread(sel.pull)
                except Exception:
                    tokens_latest, enr = (sel.active_tokens, enriched)
                to_add, to_remove = sel.diff(tokens_latest)
                if to_add or to_remove:
                    # Snapshot updates active_tokens and timestamps
                    try:
                        sel.snapshot(tokens_latest)
                    except Exception:
                        pass
                    logger.info("🔄 MARKET SELECTION CHANGE DETECTED - Updating trading configuration")
                    logger.info("Previous active markets: %d", len(token_ids))
                    # Cancel orders for removed tokens before reconfiguring
                    if to_remove:
                        for tok in to_remove:
                            try:
                                orders.cancel_market_orders(asset_id=tok, reason="removed_from_selection")  # type: ignore[attr-defined]
                                logger.info("Cancelled orders for removed token %s", tok)
                            except Exception:
                                logger.warning("Failed to cancel orders for removed token %s", tok)

                    # Update tokens and subscriptions
                    token_ids = list(tokens_latest)
                    logger.info("New active markets: %d", len(token_ids))
                    
                    # Use fresh enrichment to capture latest condition ids
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
                                tick=float(getattr(cfg, "price_tick", 0.01)),
                            )
                            new_strategies += 1
                    # Log current strategy params after selection changes
                    try:
                        logger.debug(
                            "AvellanedaLite params (post-selection): alpha_fair=%s k_vol=%s k_fee_ticks=%s inv_gamma=%s order_layers=%s base_size_usd=%s max_size_usd=%s price_tick=%s min_buy_price=%s max_position_shares=%s requote_mid_ticks=%s order_max_age_sec=%s",
                            getattr(cfg, "alpha_fair", None),
                            getattr(cfg, "k_vol", None),
                            getattr(cfg, "k_fee_ticks", None),
                            getattr(cfg, "inv_gamma", None),
                            getattr(cfg, "order_layers", None),
                            getattr(cfg, "base_size_usd", None),
                            getattr(cfg, "max_size_usd", None),
                            getattr(cfg, "price_tick", None),
                            getattr(cfg, "min_buy_price", None),
                            getattr(cfg, "max_position_shares", None),
                            getattr(cfg, "requote_mid_ticks", None),
                            getattr(cfg, "order_max_age_sec", None),
                        )
                    except Exception:
                        pass
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
                # Recompute suspension on every tick from latest sheet enrichment
                new_susp: set[str] = set()
                try:
                    if "suspended" in enr.columns:
                        for _, row in enr.iterrows():
                            try:
                                is_susp2 = bool(row.get("suspended", False))
                            except Exception:
                                is_susp2 = False
                            if is_susp2:
                                t1v2 = str(row.get("token1") or "").strip() if "token1" in enr.columns else ""
                                t2v2 = str(row.get("token2") or "").strip() if "token2" in enr.columns else ""
                                if t1v2:
                                    new_susp.add(t1v2)
                                if t2v2:
                                    new_susp.add(t2v2)
                except Exception:
                    logger.exception("Failed to recompute suspended set from sheet")
                added_susp = sorted(list(new_susp - suspended_tokens))
                removed_susp = sorted(list(suspended_tokens - new_susp))
                suspended_tokens = new_susp
                if added_susp:
                    logger.info("Suspension ENABLED for %d tokens; cancelling existing orders and gating BUYs", len(added_susp))
                    for tok in added_susp:
                        try:
                            orders.cancel_market_orders(asset_id=tok, reason="suspension_enabled")  # type: ignore[attr-defined]
                            logger.info("Cancelled existing orders for newly suspended token %s", tok)
                        except Exception:
                            logger.warning("Failed to cancel existing orders for suspended token %s", tok)
                if removed_susp:
                    logger.info("Suspension DISABLED for %d tokens; resuming BUY placements", len(removed_susp))
                else:
                    logger.info("Selection tick complete: no selection changes detected; active markets=%d suspended=%d", len(token_ids), len(suspended_tokens))
                
                # Log periodic status hourly based on timestamp since last change
                if (time.time() - sel.ts) > 3600:
                    logger.info("📊 Selection status: %d active markets, %d strategies, %d market mappings", 
                               len(token_ids), len(strategies), len(token_to_market))
                
                # Re-read interval from Settings each cycle to allow runtime tuning without restart
                try:
                    from mm.sheet_config import load_config as _reload_cfg  # local import to avoid cycles
                    new_cfg = _reload_cfg()
                    sleep_sec = int(getattr(new_cfg, "selection_loop_sec", getattr(cfg, "selection_loop_sec", 300)))
                except Exception:
                    sleep_sec = int(getattr(cfg, "selection_loop_sec", 300))
                await asyncio.sleep(max(1, sleep_sec))
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
    # Use configured browser address for merger (env may be unset)
    logger.info(
        "Launching merger loop: wallet=%s interval=%ds min=%.6f chunk=%.6f retries=%d backoff_ms=%d dry_run=%s",
        (cfg.browser_address or ""),
        int(cfg.merge_scan_interval_sec),
        float(cfg.min_merge_usdc),
        float(cfg.merge_chunk_usdc),
        int(cfg.merge_max_retries),
        int(cfg.merge_retry_backoff_ms),
        str(bool(cfg.merge_dry_run)),
    )
    merge_task = asyncio.create_task(merge_mgr.run_loop(cfg.browser_address))
    try:
        while True:
            # Pull live positions and NAV for inventory-aware quoting
            # Standardize wallet source from config
            wallet = cfg.browser_address or ""
            if wallet:
                if pm_client is not None:
                    positions_by_token, pos_value_usd, cash_usdc = await asyncio.gather(
                        asyncio.to_thread(_fetch_positions_by_token_via_client, pm_client),
                        asyncio.to_thread(_fetch_positions_value_usd, wallet),
                        asyncio.to_thread(_fetch_usdc_balance, wallet),
                    )
                else:
                    positions_by_token, pos_value_usd, cash_usdc = await asyncio.gather(
                        asyncio.to_thread(_fetch_positions_by_token, wallet),
                        asyncio.to_thread(_fetch_positions_value_usd, wallet),
                        asyncio.to_thread(_fetch_usdc_balance, wallet),
                    )
                total_bankroll_usd = float(max(0.0, pos_value_usd)) + float(max(0.0, cash_usdc))
            else:
                positions_by_token = {}
                pos_value_usd = 0.0
                cash_usdc = 0.0
                total_bankroll_usd = 0.0

            # Heartbeat every 5s to confirm loop is alive
            try:
                now_hb = time.monotonic()
                if now_hb - _last_heartbeat >= float(getattr(cfg, "heartbeat_sec", 5)):
                    logger.debug("MM tick: tokens=%d books=%d", len(token_ids), len(md.books))
                    # Heartbeat-level balances debug (reuse values fetched this tick)
                    try:
                        if wallet:
                            logger.debug(
                                "Balances: cash_usdc=%.2f positions_value=%.2f total_bankroll=%.2f",
                                float(cash_usdc),
                                float(pos_value_usd),
                                float(total_bankroll_usd),
                            )
                    except Exception:
                        pass
                    _last_heartbeat = now_hb
            except Exception:
                pass

            # Fetch live orders once per cycle for capacity checks
            try:
                live_orders_all = orders.get_orders()  # type: ignore[attr-defined]
            except Exception:
                live_orders_all = []

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
                    now_bf = time.monotonic()
                    last_ts = backfill_last_at.get(t1, 0.0)
                    throttle = float(getattr(cfg, "backfill_throttle_sec", 10))
                    if now_bf - last_ts > throttle:
                        try:
                            md.backfill_top_of_book([t1])
                        except Exception:
                            logger.exception("token1 REST backfill failed for %s", t1)
                        backfill_last_at[t1] = now_bf
                    # skip this pair for now; next loop should have levels
                    continue
                if bb1 is None and ba1 is None:
                    logger.debug("No best levels for token1=%s", t1)
                    continue
                mid1_val = _compute_mid(bb1, ba1)
                if mid1_val is None:
                    continue
                mid1 = float(mid1_val)
                mid2 = float(min(0.99, max(0.01, 1.0 - mid1)))
                # best bid/ask/mid logging already present elsewhere; avoid duplicates
                last_fair_seen[t1] = mid1
                if t2:
                    last_fair_seen[t2] = mid2

                desired_quotes: List[DesiredQuote] = []
                mid_by_token: Dict[str, float] = {}

                # Quote helper for a single token (collect desired quotes)
                def _quote_token(tok: str, fair_mid_val: float) -> None:
                    # Cooldown
                    now_ts = time.monotonic()
                    until = cooldown_until.get(tok)
                    if until is not None and now_ts < until:
                        logger.debug("Cooldown active for token %s for %.1fs", tok, until - now_ts)
                        return
                    bookx = md.books.get(tok)
                    if not bookx:
                        logger.debug("No orderbook for token=%s (market=%s)", tok, cid)
                        return
                    shares = positions_by_token.get(tok, 0.0)
                    # Hard guard: if position already at/over cap, cancel and skip BUY building entirely
                    try:
                        max_shares_guard = float(getattr(cfg, "max_position_shares", 500))
                        if float(shares) >= max_shares_guard:
                            now_cancel = time.monotonic()
                            next_ok = overcap_cancel_until.get(tok, 0.0)
                            if now_cancel >= next_ok:
                                try:
                                    orders.cancel_market_orders(asset_id=tok, reason="position_over_cap")  # type: ignore[attr-defined]
                                    overcap_cancel_until[tok] = now_cancel + float(getattr(cfg, "order_max_age_sec", 12))
                                    logger.info("Position already over cap for %s (%.2f >= %.2f). Cancelled open orders.", tok, float(shares), max_shares_guard)
                                except Exception:
                                    logger.warning("Failed to cancel open orders for over-cap token %s", tok)
                            else:
                                logger.debug("Over-cap cancel throttle active for %s", tok)
                    except Exception:
                        pass
                    # Inventory in USD as payout exposure at resolution (YES only here)
                    inventory_usd = float(shares) * 1.0
                    # Use risk_budget_usd consistently for normalization
                    try:
                        risk_budget = float(getattr(cfg, "risk_budget_usd", 1000.0))
                    except Exception:
                        risk_budget = 1000.0
                    inventory_norm = (inventory_usd / risk_budget) if risk_budget > 0 else 0.0
                    strat = strategies.get(tok) or AvellanedaLite(
                        alpha_fair=cfg.alpha_fair,
                        k_vol=cfg.k_vol,
                        k_fee_ticks=cfg.k_fee_ticks,
                        inv_gamma=cfg.inv_gamma,
                        tick=float(getattr(cfg, "price_tick", 0.01)),
                    )
                    strategies[tok] = strat
                    # Drive strategy sigma/microprice off token1 orderbook consistently
                    quote = strat.compute_quote(book1, inventory_norm, fair_hint=fair_mid_val, token_id=tok)
                    if quote is None:
                        return
                    # Apply risk multipliers to pricing (spread and reservation price) before layering
                    from mm.strategy import StrategyState
                    adj = risk.apply(
                        state=StrategyState(
                            fair=fair_mid_val,
                            sigma=0.0,
                            inventory_usd=inventory_usd,
                            bankroll_usd=total_bankroll_usd if total_bankroll_usd > 0 else 1.0,
                            time_to_resolution_sec=24 * 3600.0,
                            tick=float(cfg.price_tick),
                        ),
                        token_id=tok,
                        nav_usd=total_bankroll_usd,
                    )
                    try:
                        center0 = (float(quote.bid) + float(quote.ask)) / 2.0
                        delta_r0 = center0 - float(fair_mid_val)
                        h0 = max(0.0, (float(quote.ask) - float(quote.bid)) / 2.0)
                        logger.debug(
                            "risk_adjust_pre: fair=%.6f center=%.6f delta_r=%.6f h=%.6f gamma_mult=%.3f h_mult=%.3f",
                            float(fair_mid_val), float(center0), float(delta_r0), float(h0), float(getattr(adj, "gamma_multiplier", 1.0)), float(getattr(adj, "h_multiplier", 1.0))
                        )
                        delta_r1 = delta_r0 * float(adj.gamma_multiplier)
                        h1 = h0 * float(adj.h_multiplier)
                        new_bid = max(0.01, min(0.99, float(fair_mid_val) + delta_r1 - h1))
                        new_ask = max(0.01, min(0.99, float(fair_mid_val) + delta_r1 + h1))
                        logger.debug(
                            "risk_adjust_post: new_bid=%.6f new_ask=%.6f",
                            float(new_bid), float(new_ask)
                        )
                        quote = type(quote)(bid=float(new_bid), ask=float(new_ask))
                    except Exception:
                        pass
                    # Per-side USDC sizing with taper (balanced by inventory)
                    # Normalize inventory by the same risk budget for sizing skew
                    I_norm = 0.0
                    try:
                        if risk_budget > 0:
                            I_norm = max(-1.0, min(1.0, float(inventory_usd) / float(risk_budget)))
                    except Exception:
                        I_norm = 0.0
                    try:
                        K_total = float(getattr(cfg, "per_reprice_usdc", getattr(cfg, "base_size_usd", 300.0)))
                    except Exception:
                        K_total = 300.0
                    K_yes = 0.5 * K_total * (1.0 - I_norm)
                    K_no = 0.5 * K_total * (1.0 + I_norm)
                    taper = [0.40, 0.30, 0.20, 0.10]
                    # Build price ladders around computed bid/ask
                    yes_prices: list[float] = []
                    no_prices: list[float] = []
                    tsize_yes: list[float] = []
                    tsize_no: list[float] = []
                    tstep = float(getattr(cfg, "price_tick", 0.01))
                    # YES (BUY) ladder below bid
                    for i, w in enumerate(taper):
                        p = round(max(0.01, min(0.99, float(quote.bid) - i * tstep)) / tstep) * tstep
                        cap = w * K_yes
                        if cap > 0 and p > 0:
                            # BUY path: convert USD cap to YES shares
                            shares = cap / p
                            yes_prices.append(p)
                            tsize_yes.append(shares * p)  # engine expects USD size; track both via logs
                    # NO (SELL YES) ladder above ask
                    for i, w in enumerate(taper):
                        p = round(max(0.01, min(0.99, float(quote.ask) + i * tstep)) / tstep) * tstep
                        cap = w * K_no
                        if cap > 0 and p > 0:
                            # SELL YES path: convert USD cap to YES shares for consistency
                            shares = cap / p
                            no_prices.append(p)
                            tsize_no.append(shares * p)  # keep USD size for engine; log share calc
                    try:
                        logger.debug(
                            "sizing_inputs: risk_budget=%.2f inv_usd=%.2f inv_norm=%.4f K_total=%.2f K_yes=%.2f K_no=%.2f prices_yes=%s prices_no=%s",
                            float(risk_budget), float(inventory_usd), float(I_norm), float(K_total), float(K_yes), float(K_no),
                            [round(x, 4) for x in yes_prices], [round(x, 4) for x in no_prices]
                        )
                    except Exception:
                        pass
                    api_bid = bookx.best_bid()
                    api_ask = bookx.best_ask()
                    logger.debug(
                        "Quoting token=%s market=%s mid=%.4f bid=%.4f ask=%.4f layers=%d",
                        tok,
                        cid,
                        fair_mid_val,
                        (api_bid if api_bid is not None else -1.0),
                        (api_ask if api_ask is not None else -1.0),
                        cfg.order_layers,
                    )
                    # Build desired quotes for diffing engine
                    if tok in suspended_tokens:
                        logger.debug("BUYs suspended by sheet for token=%s; skipping BUY quotes", tok)
                    else:
                        # Compute remaining buy capacity in SHARES accounting for open BUY orders
                        try:
                            current_shares = float(positions_by_token.get(tok, 0.0))
                        except Exception:
                            current_shares = 0.0
                        try:
                            max_shares = float(getattr(cfg, "max_position_shares", 500))
                        except Exception:
                            max_shares = 500.0
                        # Sum remaining open BUY orders in shares for this token (original - matched)
                        open_buy_shares = 0.0
                        try:
                            for o in live_orders_all:
                                try:
                                    o_tok = str(o.get("asset_id") or o.get("token_id") or o.get("market") or "")
                                    o_side = str(o.get("side") or o.get("action") or o.get("order_side") or "").upper()
                                    if o_tok != tok or o_side != "BUY":
                                        continue
                                    o_price = float(o.get("price") or 0.0)
                                    # Prefer remaining size if available
                                    size_rem = None
                                    try:
                                        orig = float(o.get("original_size") or 0.0)
                                        filled = float(o.get("size_matched") or 0.0)
                                        rem = max(0.0, orig - filled)
                                        size_rem = rem if rem > 0.0 else None
                                    except Exception:
                                        size_rem = None
                                    if size_rem is None:
                                        size_rem = float(o.get("size") or o.get("remaining_size") or 0.0)
                                    if o_price > 0.0 and size_rem > 0.0:
                                        open_buy_shares += (size_rem / o_price)
                                except Exception:
                                    continue
                        except Exception:
                            open_buy_shares = 0.0
                        shares_budget = max(0.0, max_shares - current_shares - open_buy_shares)
                        try:
                            logger.debug(
                                "Cap check %s: cap=%.2f current=%.2f open_buy_shares=%.2f budget=%.2f",
                                tok, max_shares, current_shares, open_buy_shares, shares_budget
                            )
                        except Exception:
                            pass
                        # If no capacity remains, proactively cancel open BUYs (throttled) and skip placing new BUYs
                        if shares_budget <= 0.0:
                            try:
                                now_cancel = time.monotonic()
                                next_ok = overcap_cancel_until.get(tok, 0.0)
                                if now_cancel >= next_ok:
                                    orders.cancel_market_orders(asset_id=tok, reason="no_buy_capacity")  # type: ignore[attr-defined]
                                    overcap_cancel_until[tok] = now_cancel + float(getattr(cfg, "order_max_age_sec", 12))
                                    logger.info(
                                        "No BUY capacity (cap=%s, pos=%.2f, open_buys=%.2f). Cancelled open orders for %s",
                                        f"{max_shares:.0f}", current_shares, open_buy_shares, tok,
                                    )
                                else:
                                    logger.debug("No BUY capacity for %s; cancel throttle active", tok)
                            except Exception:
                                logger.warning("Failed to cancel open orders for %s when capacity is 0", tok)
                        # Minimum per-order notional in USD (exchange enforces >= 5)
                        try:
                            min_order_usd = float(getattr(cfg, "min_order_usd", 5.0))
                        except Exception:
                            min_order_usd = 5.0
                        # Parity bounds using opposite token bests (if available)
                        try:
                            other_tok = None
                            for a, b in token_pairs:
                                if a == tok:
                                    other_tok = b
                                    break
                                if b == tok:
                                    other_tok = a
                                    break
                            other_book = md.books.get(other_tok) if other_tok else None
                            other_bb = float(other_book.best_bid()) if (other_book and other_book.best_bid() is not None) else None
                            other_ba = float(other_book.best_ask()) if (other_book and other_book.best_ask() is not None) else None
                        except Exception:
                            other_bb = None
                            other_ba = None
                        for idx, (bp, usd_cap) in enumerate(zip(yes_prices, tsize_yes)):
                            if usd_cap > 0:
                                # Apply parity cap for YES bid: bid <= 1 - other_bb - tick
                                bp_eff = bp
                                try:
                                    if other_bb is not None:
                                        bp_eff = min(bp_eff, max(0.01, min(0.99, 1.0 - other_bb - tstep)))
                                    # Add one extra tick for profit versus current best bid (stay one tick below)
                                    if api_bid is not None:
                                        bp_eff = min(bp_eff, max(0.01, min(0.99, float(api_bid) - tstep)))
                                    # Round to tick after bounds
                                    bp_eff = max(0.01, min(0.99, round(bp_eff / tstep) * tstep))
                                except Exception:
                                    pass
                                # Enforce minimum notional
                                if float(usd_cap) < float(min_order_usd):
                                    try:
                                        logger.debug(
                                            "skip_buy_below_min_usd: token=%s lvl=%d price=%.4f usd_cap=%.2f min_usd=%.2f",
                                            tok, int(idx), float(bp_eff), float(usd_cap), float(min_order_usd)
                                        )
                                    except Exception:
                                        pass
                                    continue
                                # Safeguard: minimum buy price
                                if float(bp_eff) < float(getattr(cfg, "min_buy_price", 0.15)):
                                    logger.debug(
                                        "Skipping BUY below min_buy_price: token=%s price=%.4f min=%.4f",
                                        tok, float(bp_eff), float(getattr(cfg, "min_buy_price", 0.15))
                                    )
                                    continue
                                # If no remaining capacity considering open BUYs, stop placing new BUYs
                                if shares_budget <= 0.0:
                                    logger.debug("No BUY capacity remaining for token %s (budget=0)", tok)
                                    break
                                # Convert USD size to shares at bid price and clamp to remaining budget
                                intended_shares = float(usd_cap) / float(bp_eff) if float(bp_eff) > 0 else 0.0
                                if intended_shares <= 0.0:
                                    continue
                                place_shares = min(intended_shares, shares_budget)
                                desired_sz = float(place_shares) * float(bp_eff)
                                # Enforce minimum notional after clamping by capacity
                                if desired_sz < float(min_order_usd):
                                    try:
                                        logger.debug(
                                            "skip_buy_after_cap_below_min_usd: token=%s lvl=%d price=%.4f desired_usd=%.2f min_usd=%.2f",
                                            tok, int(idx), float(bp_eff), float(desired_sz), float(min_order_usd)
                                        )
                                    except Exception:
                                        pass
                                    continue
                                if desired_sz <= 0.0:
                                    continue
                                if place_shares < intended_shares:
                                    logger.debug(
                                        "Capping BUY due to remaining capacity: token=%s price=%.4f orig_shares=%.2f cap_shares=%.2f",
                                        tok, float(bp_eff), float(intended_shares), float(place_shares)
                                    )
                                desired_quotes.append(DesiredQuote(token_id=tok, side=Side.BUY, price=float(bp_eff), size=float(desired_sz), level=idx))
                                shares_budget -= place_shares
                                try:
                                    logger.debug(
                                        "place_plan_buy: token=%s lvl=%d price=%.4f usd_cap=%.2f shares=%.4f desired_usd=%.2f rem_shares_cap=%.2f",
                                        tok, int(idx), float(bp_eff), float(usd_cap), float(place_shares), float(desired_sz), float(shares_budget)
                                    )
                                except Exception:
                                    pass
                    # Compute SELL capacity based on current holdings minus open SELLs
                    open_sell_shares = 0.0
                    try:
                        for o in live_orders_all:
                            try:
                                o_tok = str(o.get("asset_id") or o.get("token_id") or o.get("market") or "")
                                o_side = str(o.get("side") or o.get("action") or o.get("order_side") or "").upper()
                                if o_tok != tok or o_side != "SELL":
                                    continue
                                o_price = float(o.get("price") or 0.0)
                                size_rem = None
                                try:
                                    orig = float(o.get("original_size") or 0.0)
                                    filled = float(o.get("size_matched") or 0.0)
                                    rem = max(0.0, orig - filled)
                                    size_rem = rem if rem > 0.0 else None
                                except Exception:
                                    size_rem = None
                                if size_rem is None:
                                    size_rem = float(o.get("size") or o.get("remaining_size") or 0.0)
                                # Convert USD to shares using order price
                                if o_price > 0.0 and size_rem > 0.0:
                                    open_sell_shares += (size_rem / o_price)
                            except Exception:
                                continue
                    except Exception:
                        open_sell_shares = 0.0
                    sell_shares_budget = max(0.0, float(positions_by_token.get(tok, 0.0)) - open_sell_shares)
                    for idx, (ap, usd_cap) in enumerate(zip(no_prices, tsize_no)):
                        if usd_cap > 0 and sell_shares_budget > 0.0:
                            # Enforce minimum notional for SELL before conversion
                            if float(usd_cap) < float(min_order_usd):
                                try:
                                    logger.debug(
                                        "skip_sell_below_min_usd: token=%s lvl=%d price=%.4f usd_cap=%.2f min_usd=%.2f",
                                        tok, int(idx), float(ap), float(usd_cap), float(min_order_usd)
                                    )
                                except Exception:
                                    pass
                                continue
                            # SELL YES parity bounds using other bests
                            ap_eff = ap
                            try:
                                if other_ba is not None:
                                    lower_bound = max(0.01, min(0.99, 1.0 - other_ba + tstep))
                                    ap_eff = max(ap_eff, lower_bound)
                                if other_bb is not None:
                                    upper_bound = max(0.01, min(0.99, 1.0 - other_bb - tstep))
                                    ap_eff = min(ap_eff, upper_bound)
                                # Add one extra tick for profit versus current best ask (stay one tick above)
                                if api_ask is not None:
                                    ap_eff = max(ap_eff, max(0.01, min(0.99, float(api_ask) + tstep)))
                                # Round to tick after bounds
                                ap_eff = max(0.01, min(0.99, round(ap_eff / tstep) * tstep))
                            except Exception:
                                pass
                            # SELL YES: gate by available shares; convert USD cap to shares and clamp
                            try:
                                assert float(ap_eff) > 0
                                intended_shares = float(usd_cap) / float(ap_eff)
                            except Exception:
                                intended_shares = 0.0
                            if intended_shares <= 0.0:
                                continue
                            place_shares = min(intended_shares, sell_shares_budget)
                            desired_usd = float(place_shares) * float(ap_eff)
                            if desired_usd < float(min_order_usd):
                                try:
                                    logger.debug(
                                        "skip_sell_after_cap_below_min_usd: token=%s lvl=%d price=%.4f desired_usd=%.2f min_usd=%.2f",
                                        tok, int(idx), float(ap_eff), float(desired_usd), float(min_order_usd)
                                    )
                                except Exception:
                                    pass
                                continue
                            if desired_usd <= 0.0:
                                continue
                            desired_quotes.append(DesiredQuote(token_id=tok, side=Side.SELL, price=float(ap_eff), size=float(desired_usd), level=idx))
                            sell_shares_budget -= place_shares
                            try:
                                logger.debug(
                                    "place_plan_sell: token=%s lvl=%d price=%.4f usd_cap=%.2f shares=%.4f desired_usd=%.2f rem_sell_shares=%.2f",
                                    tok, int(idx), float(ap_eff), float(usd_cap), float(place_shares), float(desired_usd), float(sell_shares_budget)
                                )
                            except Exception:
                                pass
                    mid_by_token[tok] = float(fair_mid_val)

                _quote_token(t1, mid1)
                if t2:
                    _quote_token(t2, mid2)
                # Sync desired quotes once per loop for both tokens
                try:
                    if not desired_quotes:
                        logger.debug("No desired quotes built this cycle (cooldowns? size=0?). Skipping engine sync.")
                        actions = SyncActions(placed=[], cancelled=[], replaced=[], errors=[])
                    else:
                        actions = engine.sync(desired_quotes, mid_by_token)
                    # Churn summary per loop
                    try:
                        logger.info(
                            "churn_summary placed=%d replaced=%d cancelled=%d errors=%d",
                            len(actions.placed), len(actions.replaced), len(actions.cancelled), len(actions.errors)
                        )
                    except Exception:
                        pass
                    # Aggregate placed USD by side for diagnostics
                    placed = actions.placed + actions.replaced
                    buy_usd = 0.0
                    sell_usd = 0.0
                    buy_cnt = 0
                    sell_cnt = 0
                    for a in placed:
                        try:
                            side = a.side.value
                            price = float(a.price)
                            size = float(a.size)
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
                        # Rate-limited summary: only log if last heartbeat passed threshold
                        if (time.monotonic() - _last_heartbeat) >= float(getattr(cfg, "heartbeat_sec", 5)):
                            logger.info("Placed quotes: BUY count=%d usd=%.2f, SELL count=%d usd=%.2f", buy_cnt, buy_usd, sell_cnt, sell_usd)
                    errs = actions.errors
                    if errs:
                        # Log first few errors and set cooldown only for affected tokens
                        for e in errs[:5]:
                            logger.warning("Order error token=%s side=%s price=%.4f size=%.2f type=%s err=%s", e.token, e.side.value, float(e.price), float(e.size), e.type, e.error)
                        now_bk = time.monotonic()
                        for e in errs:
                            tok_e = str(e.token or "")
                            if tok_e and str(e.type) == "nonretryable":
                                cooldown_until[tok_e] = now_bk + float(cfg.nonretryable_cooldown_sec)
                    for tok, m in mid_by_token.items():
                        last_quote_ts[tok] = time.time()
                        last_mid_seen[tok] = m
                except NonRetryableOrderError as nre:
                    logger.warning("Non-retryable order error: %s", nre)
                    # Apply cooldown/backoff for all tokens we attempted to quote this cycle
                    now_bk = time.monotonic()
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
                # Token2 parity divergence diagnostic if token2 book exists
                try:
                    b2 = md.books.get(t2).best_bid() if (t2 and md.books.get(t2)) else None  # type: ignore[union-attr]
                    a2 = md.books.get(t2).best_ask() if (t2 and md.books.get(t2)) else None  # type: ignore[union-attr]
                    if b2 is not None and a2 is not None:
                        mid2_ob = (float(b2) + float(a2)) / 2.0
                        if abs(mid2_ob - mid2) >= float(getattr(cfg, "price_tick", 0.01)):
                            logger.warning(
                                "token2_parity_divergence market=%s t1=%s t2=%s mid1=%.4f mirrored_mid2=%.4f ob_mid2=%.4f diff=%.4f",
                                cid, t1, (t2 or ""), mid1, mid2, mid2_ob, abs(mid2_ob - mid2)
                            )
                except Exception:
                    pass
                # best bid/ask/mid logging already present elsewhere; avoid duplicates
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


def start_market_maker(test_mode: bool = False, debug: bool = False) -> None:
    """Blocking entrypoint to start the market maker from external scripts.

    Args:
        test_mode: If True, runs in dry-run mode without sending real orders.
        debug: If True, enables DEBUG log level.
    """
    asyncio.run(main_async(test_mode=test_mode, debug_logging=debug))


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Market-Making Daemon")
    parser.add_argument("--test", action="store_true", help="Dry run: no orders sent, debug logs to file")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging without enabling dry-run")
    args = parser.parse_args()
    asyncio.run(main_async(test_mode=bool(args.test), debug_logging=bool(args.debug)))


if __name__ == "__main__":
    main()
