#!/usr/bin/env python3
import logging
import os
from typing import Dict, Iterable, List, Tuple, Optional
import math
import argparse
import numpy as np

import pandas as pd
import requests
from dotenv import load_dotenv

from data_updater.google_utils import get_spreadsheet
from store_selected_markets import write_sheet


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i+size]


def _to_float(value) -> float:
    try:
        if value is None:
            return 0.0
        # Many Gamma fields are strings; coerce robustly
        return float(str(value).replace(",", "").strip())
    except Exception:
        return 0.0


def fetch_gamma_markets_by_condition_ids(condition_ids: List[str], batch_size: int = 100) -> Dict[str, Tuple[float, float, float, float, float]]:
    """Return mapping condition_id -> (volume, liquidity, volume24hr, volume1wk, volume1mo) from Gamma /markets.

    Parses exactly 'volume', 'liquidity', 'volume24hr', 'volume1wk', and 'volume1mo'.
    """
    url = "https://gamma-api.polymarket.com/markets"
    headers = {
        "User-Agent": "poly-maker/market-size/1.0",
        "Accept": "application/json",
    }
    result: Dict[str, Tuple[float, float, float, float, float]] = {}

    uniq_ids = sorted(set([str(x).strip() for x in condition_ids if str(x).strip()]))
    logger.info("Gamma fetch: %s condition_ids", len(uniq_ids))

    for chunk in _chunk(uniq_ids, batch_size):
        # condition_ids is a repeated query parameter per docs
        params_list: List[Tuple[str, str]] = [("condition_ids", cid) for cid in chunk]
        # Provide a generous limit to avoid server-side truncation
        params_list.append(("limit", str(len(chunk))))
        try:
            logger.debug("GET %s ids=%s", url, len(chunk))
            resp = requests.get(url, params=params_list, headers=headers, timeout=20)
            if not resp.ok:
                logger.warning("Gamma GET not OK %s: %s", resp.status_code, resp.text[:200])
                continue
            data = resp.json()
            # Gamma commonly returns a list of market objects
            items: List[dict]
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "markets" in data:
                items = data.get("markets", [])
            else:
                items = [data]

            matched = 0
            for m in items:
                if not isinstance(m, dict):
                    continue
                cid = str(m.get("conditionId") or "").strip()
                if not cid:
                    continue
                if cid not in chunk:
                    # Only record rows we asked for
                    continue
                vol = _to_float(m.get("volume"))
                liq = _to_float(m.get("liquidity"))
                vol_24h = _to_float(m.get("volume24hr"))
                vol_1wk = _to_float(m.get("volume1wk"))
                vol_1mo = _to_float(m.get("volume1mo"))
                result[cid] = (vol, liq, vol_24h, vol_1wk, vol_1mo)
                matched += 1
            logger.info("Gamma chunk done: requested=%s matched=%s total=%s", len(chunk), matched, len(result))
        except Exception:
            logger.exception("Gamma request failed for chunk of size %s", len(chunk))
            continue

    return result


def fetch_all_gamma_markets(limit: int = 500, max_offset: Optional[int] = None) -> List[dict]:
    """Fetch all markets from Gamma via pagination.

    Returns a list of raw market dicts.
    """
    url = "https://gamma-api.polymarket.com/markets"
    headers = {
        "User-Agent": "poly-maker/market-size/1.0",
        "Accept": "application/json",
    }
    out: List[dict] = []
    offset = 0
    while True:
        try:
            params = {"limit": str(limit), "offset": str(offset)}
            logger.debug("GET %s?limit=%s&offset=%s", url, limit, offset)
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if not resp.ok:
                logger.warning("Gamma page not OK %s: %s", resp.status_code, resp.text[:200])
                break
            data = resp.json()
            items: List[dict]
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "markets" in data:
                items = data.get("markets", [])
            else:
                items = [data]
            # Stop when no items or short page
            if not items:
                logger.info("Gamma pagination done at offset=%s (no items)", offset)
                break
            out.extend([m for m in items if isinstance(m, dict)])
            logger.info("Fetched gamma markets: page=%s size=%s total=%s", offset // limit, len(items), len(out))
            offset += limit
            # If server returned fewer than requested, we're done
            if len(items) < limit:
                logger.info("Gamma pagination done at offset=%s (short page size=%s)", offset, len(items))
                break
            # Optional safety cap if provided
            if max_offset is not None and max_offset >= 0 and offset > max_offset:
                logger.info("Reached max_offset cap=%s; stopping", max_offset)
                break
        except Exception:
            logger.exception("Gamma pagination error at offset=%s", offset)
            break
    return out


def build_market_size_df(
    apply_filter: bool = True,
) -> pd.DataFrame:
    markets = fetch_all_gamma_markets()
    if not markets:
        return pd.DataFrame(columns=[
            "market",
            "condition_id",
            "volume",
            "liquidity",
            "volume24hr",
            "avg_volume_1wk",
            "avg_volume_1mo",
            "trend_score",
            "mm_score",
        ])

    # Build base frame from Gamma markets
    df = pd.DataFrame(markets)
    # Normalize expected columns; default to empty strings or zeros
    condition_id = df.get("conditionId", pd.Series([""] * len(df))).astype(str)
    market_name = df.get("question", pd.Series([""] * len(df))).astype(str)
    base = pd.DataFrame({
        "condition_id": condition_id,
        "market": market_name,
    })

    # Extract metrics
    vol_series = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0).astype(float)
    liq_series = pd.to_numeric(df.get("liquidity", 0), errors="coerce").fillna(0.0).astype(float)
    d24_series = pd.to_numeric(df.get("volume24hr", 0), errors="coerce").fillna(0.0).astype(float)
    wk_series = pd.to_numeric(df.get("volume1wk", 0), errors="coerce").fillna(0.0).astype(float)
    mo_series = pd.to_numeric(df.get("volume1mo", 0), errors="coerce").fillna(0.0).astype(float)
    best_bid_series = pd.to_numeric(df.get("bestBid", 0), errors="coerce").fillna(0.0).astype(float)
    best_ask_series = pd.to_numeric(df.get("bestAsk", 0), errors="coerce").fillna(0.0).astype(float)
    description_series = df.get("description", pd.Series([""] * len(df))).astype(str)
    # Extract token ids if present (Gamma sometimes exposes as comma-separated string or list)
    clob_ids_raw = df.get("clobTokenIds")
    token1_list: list[str] = []
    token2_list: list[str] = []
    if clob_ids_raw is not None:
        for v in clob_ids_raw.tolist():
            t1, t2 = "", ""
            try:
                if isinstance(v, list):
                    if len(v) > 0:
                        t1 = str(v[0])
                    if len(v) > 1:
                        t2 = str(v[1])
                elif isinstance(v, str):
                    parts = [p.strip() for p in v.split(",") if p.strip()]
                    if len(parts) > 0:
                        t1 = parts[0]
                    if len(parts) > 1:
                        t2 = parts[1]
            except Exception:
                t1, t2 = "", ""
            token1_list.append(t1)
            token2_list.append(t2)
    else:
        token1_list = [""] * len(df)
        token2_list = [""] * len(df)

    out = base.copy()
    out["volume"] = vol_series
    out["liquidity"] = liq_series
    out["volume24hr"] = d24_series
    out["best_bid"] = best_bid_series
    out["best_ask"] = best_ask_series
    out["description"] = description_series
    out["token1"] = pd.Series(token1_list, index=out.index)
    out["token2"] = pd.Series(token2_list, index=out.index)
    # Weekly/monthly totals with fallbacks from 24h volume to avoid zeroed scores
    weekly_total = wk_series.copy()
    monthly_total = mo_series.copy()
    # If weekly is zero but 24h exists, approximate weekly from 24h * 7
    weekly_zero_mask = (weekly_total <= 0) & (d24_series > 0)
    weekly_total.loc[weekly_zero_mask] = (d24_series * 7.0).loc[weekly_zero_mask]
    # If monthly is zero but weekly exists, backfill from weekly; else from 24h * 30
    monthly_zero_mask = monthly_total <= 0
    monthly_total.loc[monthly_zero_mask & (weekly_total > 0)] = weekly_total.loc[monthly_zero_mask & (weekly_total > 0)]
    monthly_total.loc[monthly_zero_mask & (weekly_total <= 0) & (d24_series > 0)] = (d24_series * 30.0).loc[monthly_zero_mask & (weekly_total <= 0) & (d24_series > 0)]
    out["avg_volume_1wk"] = weekly_total / 7.0
    out["avg_volume_1mo"] = monthly_total / 30.0

    # Derived metrics and filter
    trend_raw = weekly_total / monthly_total
    # Replace +/- inf with NaN, then fill
    trend_clean = trend_raw.replace([np.inf, -np.inf], pd.NA).fillna(0.0)
    out["trend_score"] = trend_clean
    # MM Score = Liquidity * sqrt(Weekly Volume)
    out["mm_score"] = out["liquidity"] * weekly_total.apply(lambda x: math.sqrt(x) if x > 0 else 0.0)
    nonzero_mm = int((out["mm_score"] > 0).sum())
    logger.info("Computed mm_score: nonzero=%s of %s", nonzero_mm, len(out))

    # Apply filter (optional): keep only rows that meet the criteria
    # mm_score >= 1000000 and 0.15 <= best_bid <= 0.85
    pre_rows = len(out)
    if apply_filter:
        kept_mask = ((out["mm_score"] >= 1000000.0) & (out["best_bid"] >= 0.15) & (out["best_bid"] <= 0.85))
        kept = out[kept_mask].copy()
        logger.info("Applied criteria filter: kept=%s dropped=%s", len(kept), pre_rows - len(kept))
        out = kept
    else:
        logger.info("Filter disabled (--all): returning all %s rows", pre_rows)

    # Drop markets without a condition_id
    out = out[out["condition_id"].astype(str) != ""].copy()
    return out.sort_values("mm_score", ascending=False).reset_index(drop=True)


def _col_to_a1(col_index: int) -> str:
    """Convert 1-based column index to A1 letter(s)."""
    label = ""
    x = int(col_index)
    while x > 0:
        x, r = divmod(x - 1, 26)
        label = chr(65 + r) + label
    return label


def apply_row_coloring(spreadsheet, sheet_title: str, df: pd.DataFrame) -> None:
    """Coloring removed per request."""
    try:
        spreadsheet.worksheet(sheet_title)
    except Exception:
        logger.warning("Worksheet '%s' not found; skipping", sheet_title)
        return
    return


def main() -> None:
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(description="Update Market Size sheet from Gamma")
    parser.add_argument("--all", dest="show_all", action="store_true", help="Do not apply filtering; return all rows ranked by mm_score")
    args = parser.parse_args()

    logger.info("Updating Market Size using Gamma (all=%s)", args.show_all)

    spreadsheet = get_spreadsheet()
    ms_df = build_market_size_df(
        apply_filter=(not args.show_all),
    )

    try:
        write_sheet(spreadsheet, "Market Size", ms_df)
        logger.info("Wrote %d rows to sheet 'Market Size'", len(ms_df))
        # If not --all, build and write Filtered Markets using remove_markets_to_avoid
        if not args.show_all:
            try:
                from ai.filter_markets import remove_markets_to_avoid  # local import to avoid heavy deps on load
            except Exception:
                logger.exception("Could not import remove_markets_to_avoid; skipping 'Filtered Markets'")
            else:
                # Construct the input DataFrame expected by remove_markets_to_avoid
                # It expects at least columns: question, token1, token2 (and may use rules)
                pairs_df = pd.DataFrame({
                    "question": ms_df.get("market", pd.Series([""] * len(ms_df))).astype(str),
                    "token1": ms_df.get("token1", pd.Series([""] * len(ms_df))).astype(str),
                    "token2": ms_df.get("token2", pd.Series([""] * len(ms_df))).astype(str),
                    "rules": ms_df.get("description", pd.Series([""] * len(ms_df))).astype(str),
                })
                filtered = remove_markets_to_avoid(pairs_df=pairs_df)
                # Merge back extra Market Size fields for the kept questions
                if not filtered.empty:
                    # Only merge back data for the questions that passed the AI filter
                    # and ensure we maintain the criteria filter
                    filtered_questions = set(filtered["question"].tolist())
                    ms_filtered = ms_df[ms_df["market"].isin(filtered_questions)].copy()
                    
                    merged = filtered.merge(
                        ms_filtered,
                        left_on="question",
                        right_on="market",
                        how="left",
                        suffixes=("", "_ms"),
                    )
                    # Keep a tidy set of columns
                    cols = [
                        "question", "token1", "token2", "rules",
                        "condition_id", "liquidity", "volume24hr", "avg_volume_1wk", "avg_volume_1mo",
                        "trend_score", "mm_score", "best_bid", "best_ask", "description",
                    ]
                    present = [c for c in cols if c in merged.columns]
                    out_df = merged[present].copy()
                else:
                    out_df = pd.DataFrame(columns=[
                        "question", "token1", "token2", "rules",
                        "condition_id", "liquidity", "volume24hr", "avg_volume_1wk", "avg_volume_1mo",
                        "trend_score", "mm_score", "best_bid", "best_ask", "description",
                    ])
                write_sheet(spreadsheet, "Filtered Markets", out_df)
                logger.info("Wrote %d rows to sheet 'Filtered Markets'", len(out_df))
    except Exception:
        logger.exception("Failed to write 'Market Size' sheet")


if __name__ == "__main__":
    main()
