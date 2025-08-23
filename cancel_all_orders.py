#!/usr/bin/env python3
import os
from typing import Set

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON


def main() -> None:
    load_dotenv()
    pk = os.getenv("PK")
    funder = os.getenv("BROWSER_ADDRESS")
    if not pk or not funder:
        print("Missing PK or BROWSER_ADDRESS in environment")
        return

    client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=POLYGON, funder=funder)
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    orders = client.get_orders()
    if not orders:
        print("No open orders to cancel.")
        return

    market_ids: Set[str] = set()
    asset_ids: Set[str] = set()
    for o in orders:
        if isinstance(o, dict):
            if o.get("market") is not None:
                market_ids.add(str(o["market"]))
            if o.get("asset_id") is not None:
                asset_ids.add(str(o["asset_id"]))

    # Prefer cancelling by market when available, then fall back to asset_id
    cancelled = 0
    for mid in sorted(market_ids):
        try:
            client.cancel_market_orders(market=mid)
            print(f"Cancelled orders for market {mid}")
            cancelled += 1
        except Exception as e:
            print(f"Failed to cancel market {mid}: {e}")

    for aid in sorted(asset_ids):
        try:
            client.cancel_market_orders(asset_id=aid)
            print(f"Cancelled orders for asset {aid}")
            cancelled += 1
        except Exception as e:
            print(f"Failed to cancel asset {aid}: {e}")

    remaining = client.get_orders()
    print(f"Done. Batches attempted: {cancelled}. Remaining open orders: {len(remaining) if remaining else 0}")


if __name__ == "__main__":
    main()


