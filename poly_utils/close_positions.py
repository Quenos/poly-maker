from typing import Dict, Optional
import logging
import pandas as pd
from dotenv import load_dotenv
from poly_data.polymarket_client import PolymarketClient


logger = logging.getLogger(__name__)


def close_positions(
    limit_prices: Optional[Dict[str, Optional[float]]] = None,
    size_fraction: Optional[float] = None,
) -> int:
    """
    Close open positions using a single token->price map.

    Args:
        limit_prices: dict mapping token_id -> price (0..1) or None.
                      - If provided, only those token_ids are closed.
                      - If price is None for a token, use aggressive marketable default:
                        0.01 for SELL (closing longs), 0.99 for BUY (closing shorts).
                      - If not provided at all (None), close all positions using aggressive defaults.
        size_fraction: optional fraction of the position to close in (0, 1].
                       If omitted, closes the entire open size.

    Returns:
        Number of positions for which a close order was submitted.
    """
    load_dotenv()
    client = PolymarketClient()

    if size_fraction is not None:
        try:
            size_fraction = float(size_fraction)
        except Exception as exc:
            raise ValueError("size_fraction must be a float in (0, 1].") from exc
        if not (0.0 < size_fraction <= 1.0):
            raise ValueError("size_fraction must be in (0, 1].")

    df = client.get_all_positions()
    if df is None or df.empty:
        logger.info("No positions to close.")
        return 0

    # Normalize needed columns
    for col in ("size", "avgPrice", "curPrice"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "asset" not in df.columns or "size" not in df.columns:
        raise ValueError("Positions DataFrame must contain 'asset' and 'size' columns.")

    # Filter which tokens to act on
    if isinstance(limit_prices, dict) and limit_prices:
        wanted = set(str(k) for k in limit_prices.keys())
        df = df[df["asset"].astype(str).isin(wanted)].copy()

    # Keep nonzero positions (both longs and shorts)
    df = df[df["size"] != 0].copy()
    if df.empty:
        logger.info("No matching open positions.")
        return 0

    FALLBACK_SELL, FALLBACK_BUY = 0.01, 0.99

    def valid_price(p: float) -> bool:
        return 0.0 < float(p) < 1.0

    def clamp_quantity(qty: float) -> float:
        # Avoid micro dust and keep reasonable precision for shares
        if qty <= 0:
            return 0.0
        # round up tiny fractional to 1e-6 resolution
        return max(0.0, round(qty, 6))

    closed = 0
    for _, row in df.iterrows():
        token_id = str(row["asset"]).strip()
        pos = float(row["size"])  # positive=long, negative=short
        if not token_id or pos == 0.0:
            continue

        # Compute desired close size (fractional supported)
        base_size = abs(pos)
        desired = base_size * (size_fraction if size_fraction is not None else 1.0)
        shares = clamp_quantity(desired)
        if shares <= 0.0:
            logger.debug("Skip %s: computed shares=0 after fraction.", token_id)
            continue

        if pos > 0:
            # long -> SELL to flatten
            side = "SELL"
            if isinstance(limit_prices, dict) and token_id in limit_prices:
                p = limit_prices[token_id]
                price = FALLBACK_SELL if p is None else float(p)
            else:
                price = FALLBACK_SELL
        else:
            # short -> BUY to flatten
            side = "BUY"
            if isinstance(limit_prices, dict) and token_id in limit_prices:
                p = limit_prices[token_id]
                price = FALLBACK_BUY if p is None else float(p)
            else:
                price = FALLBACK_BUY

        if not valid_price(price):
            logger.warning("Skip %s: invalid price=%r", token_id, price)
            continue

        try:
            logger.info("Closing %s: %s %s @ %s (fraction=%s)", token_id, side, shares, price, size_fraction)
            # Signature assumed: create_order(token_id, side, price, size, is_market=False)
            resp = client.create_order(token_id, side, price, shares, False)
            logger.debug("create_order response: %s", resp)
            closed += 1
        except Exception as exc:
            logger.exception("Failed to close %s: %s", token_id, str(exc))

    logger.info("Submitted close orders for %s positions.", closed)
    return closed