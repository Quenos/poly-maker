#!/usr/bin/env python3
import logging
import os
from typing import Iterable, List, Sequence, Optional, Dict, Any
import concurrent.futures

import pandas as pd
import requests
from dotenv import load_dotenv

from poly_data.polymarket_client import PolymarketClient


def fetch_asset_metadata(asset_ids: Sequence[str]) -> pd.DataFrame:
    """Return DataFrame with columns [asset_id, question, outcome]."""
    rows: List[dict] = []
    # Query in chunks to keep URLs reasonable
    for i in range(0, len(asset_ids), 50):
        chunk = list(asset_ids)[i:i + 50]
        qs = ",".join(chunk)
        for url in (
            f"https://data-api.polymarket.com/assets?ids={qs}",
            f"https://clob.polymarket.com/assets?ids={qs}",
        ):
            try:
                r = requests.get(url, timeout=10)
                if not r.ok:
                    continue
                data = r.json()
                if isinstance(data, dict) and "assets" in data:
                    data = data["assets"]
                if not isinstance(data, list):
                    continue
                for a in data:
                    if not isinstance(a, dict):
                        continue
                    aid = str(
                        a.get("id")
                        or a.get("token_id")
                        or a.get("tokenId")
                        or ""
                    )
                    if not aid:
                        continue
                    question = (
                        a.get("question")
                        or a.get("market_question")
                        or a.get("title")
                        or ""
                    )
                    outcome = a.get("outcome") or a.get("name") or ""
                    rows.append(
                        {
                            "asset_id": aid,
                            "question": question,
                            "outcome": outcome,
                        }
                    )
                break
            except Exception:
                continue
    if rows:
        df_rows = pd.DataFrame(rows).drop_duplicates(
            subset=["asset_id"]
        )  # type: ignore[no-any-return]
        return df_rows
    cols = [
        "asset_id",
        "question",
        "outcome",
    ]
    return pd.DataFrame(columns=cols)  # type: ignore[no-any-return]


def verify_onchain_positions(
    client: PolymarketClient,
    asset_ids: Iterable[str],
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Check ERC-1155 ConditionalTokens balances; return non-zero positions.

    Returns DataFrame with columns:
    - asset
    - raw_position
    - shares (decimal, 1e6)
    """
    records: List[Dict[str, Any]] = []
    assets_list = list(asset_ids)
    total = len(assets_list)
    if total == 0:
        return pd.DataFrame(records)

    if max_workers is None:
        try:
            max_workers = int(os.getenv("SWEEP_WORKERS", "16"))
        except Exception:
            max_workers = 16

    def task(aid: str) -> Optional[Dict[str, Any]]:
        try:
            token_id_int = int(str(aid))
        except ValueError:
            return None
        try:
            raw_position, shares = client.get_position(token_id_int)
        except Exception:
            return None
        if shares and shares > 0:
            return {
                "asset": str(aid),
                "raw_position": int(raw_position),
                "shares": float(shares),
            }
        return None

    processed = 0
    log_every = max(10, total // 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, aid) for aid in assets_list]
        for fut in concurrent.futures.as_completed(futures):
            processed += 1
            try:
                res = fut.result()
                if res is not None:
                    records.append(res)
            except Exception:
                pass
            if processed % log_every == 0 or processed == total:
                logging.debug(
                    "On-chain progress: %d/%d (found %d)",
                    processed,
                    total,
                    len(records),
                )

    return pd.DataFrame(records)


def collect_candidate_assets(
    client: PolymarketClient,
) -> List[str]:
    """
    Collect candidate asset ids from API positions; fallback to sheet tokens.
    """
    candidates: List[str] = []
    try:
        api_df = client.get_all_positions()
        if (
            api_df is not None
            and not api_df.empty
            and "asset" in api_df.columns
        ):
            unique_assets = set(api_df["asset"].astype(str))
            candidates = sorted(unique_assets)  # type: ignore[assignment]
    except Exception:
        pass

    if candidates:
        logging.debug("Positions API provided %d assets", len(candidates))
        return candidates

    # Optional fallback sweep of all markets (expensive!)
    sweep_all = os.getenv("SWEEP_FULL_MARKETS", "0").lower() in ("1", "true", "yes")
    if not sweep_all:
        logging.debug("Positions API empty; falling back to sheet tokens (limited sweep)")

    try:
        # Lazy import to avoid a hard dependency when not available
        from poly_utils.google_utils import get_spreadsheet

        spreadsheet = get_spreadsheet(read_only=True)
        wk_full = spreadsheet.worksheet("Full Markets")
        import pandas as _pd

        markets_df = _pd.DataFrame(wk_full.get_all_records())
        tokens: List[str] = []
        for col in ("token1", "token2"):
            if col in markets_df.columns:
                tokens.extend(markets_df[col].astype(str).tolist())
        filtered_tokens = (t for t in tokens if t and t != "nan")
        candidates = sorted(set(filtered_tokens))
        # Apply default limit when not forcing full sweep
        if not sweep_all:
            try:
                default_limit = int(os.getenv("SWEEP_LIMIT", "300"))
            except Exception:
                default_limit = 300
            if len(candidates) > default_limit:
                candidates = candidates[:default_limit]
                logging.debug("Applying default sheet fallback limit=%d", default_limit)
    except Exception:
        candidates = []
    return candidates


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    load_dotenv()
    setup_logging()

    logging.debug("Starting list_open_positions")
    logging.debug(
        "RPC=%s wallet=%s",
        os.getenv("POLYGON_RPC_URL", ""),
        os.getenv("BROWSER_ADDRESS", ""),
    )

    # Use lightweight mode to avoid API setup hangs; we only need on-chain reads
    client = PolymarketClient(initialize_api=False)
    logging.debug("PolymarketClient initialized (API skipped)")

    # Collect candidate assets and verify on-chain balances
    asset_ids: List[str] = collect_candidate_assets(client)
    logging.debug("Candidate assets: %d", len(asset_ids))
    if not asset_ids:
        logging.info("No candidate assets found to check on-chain.")
        return

    # Optional limit to avoid very long sweeps
    limit_env = os.getenv("SWEEP_LIMIT")
    if limit_env:
        try:
            limit = int(limit_env)
            asset_ids = asset_ids[:max(0, limit)]
            logging.debug("Applying SWEEP_LIMIT=%d", limit)
        except Exception:
            pass

    logging.debug("Verifying on-chain balances for candidates...")
    onchain_df = verify_onchain_positions(client, asset_ids)
    logging.debug("Found non-zero positions: %d", len(onchain_df))
    if onchain_df is None or onchain_df.empty:
        logging.info("No positions found on-chain.")
        return

    # Enrich with metadata and any available pricing from API positions
    meta = fetch_asset_metadata(list(onchain_df["asset"]))
    result = onchain_df.merge(
        meta,
        left_on="asset",
        right_on="asset_id",
        how="left",
    )

    # Try to augment with API position pricing to compute value
    try:
        api_df = client.get_all_positions()
        if api_df is not None and not api_df.empty:
            numeric_cols = ("size", "avgPrice", "curPrice", "percentPnl")
            for col in numeric_cols:
                if col in api_df.columns:
                    api_df[col] = pd.to_numeric(api_df[col], errors="coerce")
            api_df["asset"] = api_df["asset"].astype(str)
            enrich_cols = ["asset", "avgPrice", "curPrice", "percentPnl"]
            result = result.merge(api_df[enrich_cols], on="asset", how="left")
    except Exception:
        pass

    if {"shares", "curPrice"}.issubset(result.columns):
        result["value_usdc"] = result["shares"] * result["curPrice"]

    cols = [
        "asset",
        "question",
        "outcome",
        "shares",
        "avgPrice",
        "curPrice",
        "percentPnl",
        "value_usdc",
    ]
    existing = [c for c in cols if c in result.columns]

    if "value_usdc" in result.columns:
        result = result.sort_values("value_usdc", ascending=False)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
    ):
        table_str = result[existing].to_string(index=False)
        logging.info("\n%s", table_str)


if __name__ == "__main__":
    main()
