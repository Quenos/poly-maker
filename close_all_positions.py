#!/usr/bin/env python3
"""
Close all open positions by placing aggressive SELL orders that should cross the
book immediately. This uses your PK/BROWSER_ADDRESS from .env.

Notes:
- Uses PolymarketClient.get_all_positions() which returns a DataFrame with
  columns including: asset (token id), size (shares), avgPrice, curPrice.
- Order size for the CLOB is nominated in USDC, so we approximate notional as
  size * curPrice.
- We set a conservative sell limit price (0.01) so the order will cross the
  best bid and fill at book prices.
"""

import os
from decimal import Decimal, ROUND_DOWN

import pandas as pd
from dotenv import load_dotenv

from poly_data.polymarket_client import PolymarketClient


def to_usdc_notional(shares: float, price: float) -> float:
    if shares <= 0 or price <= 0:
        return 0.0
    notional = Decimal(str(shares)) * Decimal(str(price))
    # Round down to 2 decimals for safety
    return float(notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def main() -> None:
    load_dotenv()

    client = PolymarketClient()
    df = client.get_all_positions()
    if df is None or df.empty:
        print("No positions to close.")
        return

    # Normalize numeric columns
    for col in ("size", "avgPrice", "curPrice"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep positions with positive size
    df = df[df.get("size", 0) > 0].copy()
    if df.empty:
        print("No positions to close.")
        return

    closed = 0
    for _, row in df.iterrows():
        token_id = str(row.get("asset", ""))
        shares = float(row.get("size", 0))
        cur_price = float(row.get("curPrice", 0))
        if not token_id or shares <= 0 or cur_price <= 0:
            continue

        notional = to_usdc_notional(shares, cur_price)
        if notional <= 0:
            continue

        # Aggressive price to ensure immediate cross; engine should fill at the
        # best bid, not at our limit when it crosses the spread
        price = 0.01
        try:
            print(f"Closing token {token_id}: SELL {notional} @ {price} (shares≈{shares}, mark≈{cur_price})")
            resp = client.create_order(token_id, "SELL", price, notional, False)
            print(f"  response: {resp}")
            closed += 1
        except Exception as e:
            print(f"  failed: {e}")

    print(f"Done. Submitted close orders for {closed} positions.")


if __name__ == "__main__":
    main()


