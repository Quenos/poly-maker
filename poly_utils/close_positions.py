from typing import Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from poly_data.polymarket_client import PolymarketClient


def close_positions(limit_prices: Optional[Dict[str, Optional[float]]] = None) -> int:
    """
    Close open positions using a single token->price map.

    Args:
        limit_prices: dict mapping token_id -> price (0..1) or None.
                      - If provided, only those token_ids are closed.
                      - If price is None for a token, use aggressive marketable default:
                        0.01 for SELL (closing longs), 0.99 for BUY (closing shorts).
                      - If not provided at all (None), close all positions using aggressive defaults.

    Returns:
        Number of positions for which a close order was submitted.
    """
    load_dotenv()
    client = PolymarketClient()

    df = client.get_all_positions()
    if df is None or df.empty:
        print("No positions to close.")
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
        print("No matching open positions.")
        return 0

    FALLBACK_SELL, FALLBACK_BUY = 0.01, 0.99
    
    def valid_price(p: float) -> bool: 
        return 0.0 < float(p) < 1.0

    closed = 0
    for _, row in df.iterrows():
        token_id = str(row["asset"]).strip()
        pos = float(row["size"])
        if not token_id or pos == 0.0:
            continue

        if pos > 0:   # long -> SELL to flatten
            side, shares = "SELL", pos
            # choose price from dict or fallback
            if isinstance(limit_prices, dict) and token_id in limit_prices:
                p = limit_prices[token_id]
                price = FALLBACK_SELL if p is None else float(p)
            else:
                price = FALLBACK_SELL
        else:         # short -> BUY to flatten
            side, shares = "BUY", abs(pos)
            if isinstance(limit_prices, dict) and token_id in limit_prices:
                p = limit_prices[token_id]
                price = FALLBACK_BUY if p is None else float(p)
            else:
                price = FALLBACK_BUY

        if not valid_price(price):
            print(f"Skip {token_id}: invalid price={price!r}.")
            continue

        try:
            print(f"Closing {token_id}: {side} {shares} @ {price}")
            # Signature assumed: create_order(token_id, side, price, size, is_market=False)
            resp = client.create_order(token_id, side, price, shares, False)
            print(f"  response: {resp}")
            closed += 1
        except Exception as e:
            print(f"  failed for {token_id}: {e}")

    print(f"Done. Submitted close orders for {closed} positions.")
    return closed