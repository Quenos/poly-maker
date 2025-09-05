#!/usr/bin/env python3
import os
import logging
import argparse
from typing import List

from dotenv import load_dotenv
from poly_data.polymarket_client import PolymarketClient


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="% (asctime)s % (levelname)s % (message)s".replace(" ", ""),
    )


def main() -> None:
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(description="List Polymarket positions using PolymarketClient")
    parser.add_argument(
        "--wallet",
        default=os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "",
        help="Wallet/proxy address override (defaults to BROWSER_WALLET/BROWSER_ADDRESS)",
    )
    parser.add_argument(
        "--fields",
        default="asset,market,size,avgPrice,curPrice,percentPnl,title,outcome",
        help="Comma-separated list of fields to display (only those present will be shown)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=int(os.getenv("POSITIONS_PREVIEW", "10")),
        help="Show only the top N rows in the preview log",
    )
    args = parser.parse_args()

    # Initialize client (no CLOB needed for positions)
    client = PolymarketClient(initialize_api=False)

    # Optional wallet override
    wallet_override: str = args.wallet.strip()
    if wallet_override:
        client.browser_wallet = wallet_override

    try:
        df = client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to fetch positions: %s", exc)
        return

    if df is None or df.empty:
        logging.info("No positions found for wallet: %s", getattr(client, "browser_wallet", "<unknown>"))
        return

    requested_fields: List[str] = [f.strip() for f in args.fields.split(",") if f.strip()]
    available_fields: List[str] = [f for f in requested_fields if f in df.columns]

    logging.info("Fetched %d positions for %s", len(df), getattr(client, "browser_wallet", "<unknown>"))
    logging.info("Available columns: %s", list(df.columns))

    if available_fields:
        preview_df = df[available_fields].copy()
    else:
        preview_df = df.copy()

    try:
        import pandas as pd  # type: ignore
        with pd.option_context("display.max_columns", None, "display.width", 180):
            logging.info("Preview (top %d):\n%s", args.top, preview_df.head(args.top).to_string(index=False))
    except Exception:  # noqa: BLE001
        # Fallback textual representation
        logging.info("First row: %s", preview_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
