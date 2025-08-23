#!/usr/bin/env python3
import os
import logging
import argparse
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="% (asctime)s % (levelname)s % (message)s".replace(" ", ""),
    )


def fetch_positions(
    wallet: str,
    limit: int = 50,
    size_threshold: float = 1.0,
    sort_by: str = "CURRENT",
    sort_direction: str = "DESC",
    market: Optional[str] = None,
    redeemable: Optional[bool] = None,
    mergeable: Optional[bool] = None,
    fetch_all: bool = False,
) -> List[Dict[str, Any]]:
    base_url = "https://data-api.polymarket.com/positions"
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    limit = max(1, min(500, int(limit)))

    while True:
        params: Dict[str, Any] = {
            "user": wallet,
            "limit": limit,
            "sizeThreshold": size_threshold,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
        }
        if market:
            params["market"] = market
        if redeemable is not None:
            params["redeemable"] = str(redeemable).lower()
        if mergeable is not None:
            params["mergeable"] = str(mergeable).lower()
        if fetch_all:
            params["offset"] = offset

        try:
            resp = requests.get(base_url, params=params, timeout=20)
            if not resp.ok:
                logging.error("positions GET failed %s -> %s %s", resp.url, resp.status_code, resp.text)
                break
            data = resp.json()
            page: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("data", [])
            if not page:
                break
            all_rows.extend(page)
            if not fetch_all or len(page) < limit:
                break
            offset += len(page)
        except Exception as e:
            logging.error("positions request error: %s", str(e))
            break

    return all_rows


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y"):  # truthy
        return True
    if v in ("0", "false", "no", "n"):   # falsy
        return False
    return None


def main() -> None:
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(description="List Polymarket positions via Data-API")
    parser.add_argument("--wallet", default=os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "", help="Wallet/proxy address")
    parser.add_argument("--limit", type=int, default=int(os.getenv("POSITIONS_LIMIT", "50")), help="Page size (1-500)")
    parser.add_argument("--size-threshold", type=float, default=float(os.getenv("SIZE_THRESHOLD", "1.0")), help="Minimum position size")
    parser.add_argument("--sort-by", default=os.getenv("SORT_BY", "CURRENT"), help="Sort by (e.g., TOKENS, CURRENT, INITIAL, CASHPNL)")
    parser.add_argument("--sort-direction", default=os.getenv("SORT_DIRECTION", "DESC"), choices=["ASC", "DESC"], help="Sort direction")
    parser.add_argument("--market", default=os.getenv("POSITIONS_MARKET", ""), help="Filter by condition IDs (comma-separated)")
    parser.add_argument("--redeemable", default=os.getenv("REDEEMABLE"), help="Filter redeemable (true/false)")
    parser.add_argument("--mergeable", default=os.getenv("MERGEABLE"), help="Filter mergeable (true/false)")
    parser.add_argument("--all", action="store_true", help="Fetch all pages using offset")
    args = parser.parse_args()

    wallet = args.wallet.strip()
    if not wallet:
        logging.error("Wallet address is required. Provide --wallet or set BROWSER_WALLET/BROWSER_ADDRESS in .env")
        return

    rows = fetch_positions(
        wallet=wallet,
        limit=args.limit,
        size_threshold=args.size_threshold,
        sort_by=args.sort_by,
        sort_direction=args.sort_direction,
        market=args.market.strip() or None,
        redeemable=parse_bool(args.redeemable),
        mergeable=parse_bool(args.mergeable),
        fetch_all=args.all,
    )

    logging.info("Fetched %d positions", len(rows))
    if not rows:
        return

    # Log a concise preview and the field names
    try:
        import pandas as pd  # local import to avoid hard dependency in case user removes it
        df = pd.DataFrame(rows)
        logging.info("Fields: %s", list(df.columns))
        with pd.option_context("display.max_columns", None, "display.width", 140):
            logging.info("Sample (top 10):\n%s", df.head(10).to_string(index=False))
    except Exception:
        # Fallback: log first item only
        logging.info("First row: %s", rows[0])


if __name__ == "__main__":
    main()
