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
import re

from data_updater.google_utils import get_spreadsheet
from store_selected_markets import write_sheet, read_sheet


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def clean_token_string(token: str) -> str:
    """
    Clean token strings by removing special characters and ensuring they remain as strings.
    
    Args:
        token: Raw token string that may contain special characters
        
    Returns:
        Cleaned token string suitable for Google Sheets (prevents scientific notation)
    """
    if not token or not isinstance(token, str):
        return ""
    
    # Remove common special characters that might appear at start/end
    # This includes quotes, brackets, extra whitespace, etc.
    cleaned = token.strip()
    
    # Remove quotes, brackets, and other common delimiters
    cleaned = re.sub(r'^["\'\[\](){}]+', '', cleaned)
    cleaned = re.sub(r'["\'\[\](){}]+$', '', cleaned)
    
    # Remove any remaining special characters that aren't alphanumeric
    # Keep only digits and letters (token IDs should be hex strings)
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', cleaned)
    
    # Ensure it's a valid token ID (should be a long hex string)
    if len(cleaned) < 10:  # Token IDs are typically very long
        return ""
    
    # Force to string and ensure it's not empty
    result = str(cleaned).strip()
    return result if result else ""


def check_market_size_criteria(market_row: pd.Series) -> bool:
    """
    Check if a market meets the minimum market size criteria.
    Markets that don't meet these criteria should be removed even if currently traded.
    
    Args:
        market_row: Single row from market DataFrame with market metrics
        
    Returns:
        True if market meets criteria, False otherwise
    """
    try:
        # Extract metrics with safe defaults
        liquidity = float(market_row.get("liquidity", 0))
        volume_7d = float(market_row.get("avg_volume_1wk", 0)) * 7  # Convert daily average to weekly
        best_bid = float(market_row.get("best_bid", 0))
        
        # Market size criteria (same as in build_market_size_df)
        mm_score = liquidity * (volume_7d ** 0.5) if volume_7d > 0 else 0
        
        # Criteria thresholds
        min_liquidity = 1000000.0  # $1M minimum liquidity
        min_weekly_volume = 50000.0  # $50K minimum weekly volume
        min_mm_score = 1000000.0  # Minimum market making score
        min_bid = 0.15  # Minimum bid price
        max_bid = 0.85  # Maximum bid price
        
        # Check all criteria
        meets_criteria = (
            liquidity >= min_liquidity and  # noqa: W504
            volume_7d >= min_weekly_volume and  # noqa: W504
            mm_score >= min_mm_score and  # noqa: W504
            min_bid <= best_bid <= max_bid
        )
        
        if not meets_criteria:
            logger.info(f"Market '{market_row.get('market', 'Unknown')}' fails criteria: "
                       f"liquidity=${liquidity:,.0f} (min: ${min_liquidity:,.0f}), "
                       f"volume_7d=${volume_7d:,.0f} (min: ${min_weekly_volume:,.0f}), "
                       f"mm_score={mm_score:,.0f} (min: {min_mm_score:,.0f}), "
                       f"best_bid={best_bid:.3f} (range: {min_bid:.3f}-{max_bid:.3f})")
        
        return meets_criteria
        
    except Exception as e:
        logger.warning(f"Error checking market size criteria: {e}")
        return False  # Fail safe - remove if we can't verify


def protect_currently_traded_markets(filtered_df: pd.DataFrame, selected_markets_df: pd.DataFrame, 
                                   ms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Protect markets that are currently in Selected Markets from being removed by AI filtering.
    This prevents disruption to active trading positions.
    
    EXCEPTION: Markets that no longer meet market size criteria are removed even if currently traded.
    
    Args:
        filtered_df: DataFrame of markets that passed AI filtering
        selected_markets_df: DataFrame of currently selected markets being traded
        ms_df: Original market size DataFrame with full market data
        
    Returns:
        DataFrame with currently traded markets protected from AI removal (except those failing criteria)
    """
    if selected_markets_df.empty:
        logger.info("No currently selected markets to protect")
        return filtered_df
    
    # Get currently traded market questions
    current_questions = set()
    if "question" in selected_markets_df.columns:
        current_questions = set(selected_markets_df["question"].astype(str).tolist())
    elif "market" in selected_markets_df.columns:
        current_questions = set(selected_markets_df["market"].astype(str).tolist())
    
    if not current_questions:
        logger.info("No currently traded market questions found")
        return filtered_df
    
    logger.info(f"Found {len(current_questions)} currently traded markets to evaluate")
    
    # Find markets that were filtered out by AI but are currently traded
    filtered_questions = set(filtered_df["question"].tolist())
    protected_markets = []
    removed_criteria_failures = []
    
    for question in current_questions:
        if question not in filtered_questions:
            # This market was filtered out by AI but is currently traded
            # Find it in the original market size data
            original_row = ms_df[ms_df["market"] == question]
            if not original_row.empty:
                market_data = original_row.iloc[0]
                
                # Check if market still meets size criteria
                if check_market_size_criteria(market_data):
                    # Market meets criteria - protect it from AI removal
                    protected_row = market_data.copy()
                    # Ensure the row has the expected column names for the filtered output
                    if "question" not in protected_row:
                        protected_row["question"] = protected_row.get("market", question)
                    protected_markets.append(protected_row)
                    logger.warning(f"🛡️  Protecting currently traded market from AI removal: {question}")
                else:
                    # Market no longer meets criteria - allow removal
                    removed_criteria_failures.append(question)
                    logger.warning(f"⚠️  Allowing removal of currently traded market (fails criteria): {question}")
            else:
                logger.warning(f"⚠️  Currently traded market '{question}' not found in market size data")
    
    # Log summary of protection decisions
    if protected_markets:
        logger.info(f"✅ Protecting {len(protected_markets)} currently traded markets that meet criteria")
    
    if removed_criteria_failures:
        logger.warning(f"🗑️  Allowing removal of {len(removed_criteria_failures)} currently traded markets that fail criteria:")
        for market in removed_criteria_failures[:5]:  # Show first 5
            logger.warning(f"  - {market}")
        if len(removed_criteria_failures) > 5:
            logger.warning(f"  ... and {len(removed_criteria_failures) - 5} more")
    
    if protected_markets:
        # Add protected markets back to filtered DataFrame
        protected_df = pd.DataFrame(protected_markets)
        
        # Ensure protected markets have all required columns including condition_id
        required_cols = ["question", "token1", "token2", "condition_id", "rules"]
        for col in required_cols:
            if col not in protected_df.columns:
                if col == "rules" and "description" in protected_df.columns:
                    protected_df[col] = protected_df["description"]
                elif col == "condition_id" and "condition_id" in protected_df.columns:
                    protected_df[col] = protected_df["condition_id"]
                else:
                    protected_df[col] = ""
        
        # Clean token strings in protected markets
        protected_df["token1"] = protected_df["token1"].apply(clean_token_string)
        protected_df["token2"] = protected_df["token2"].apply(clean_token_string)
        protected_df["token1"] = protected_df["token1"].astype(str)
        protected_df["token2"] = protected_df["token2"].astype(str)
        
        # Merge with filtered results
        final_df = pd.concat([filtered_df, protected_df], ignore_index=True)
        logger.info(f"Final filtered markets: {len(final_df)} (AI filtered: {len(filtered_df)}, Protected: {len(protected_markets)}, Removed by criteria: {len(removed_criteria_failures)})")
        
        return final_df
    else:
        logger.info("✅ No currently traded markets need protection (all either passed AI filtering or failed criteria)")
        return filtered_df


def get_ai_evaluation_data(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame showing the raw AI evaluation results for all markets.
    
    Args:
        original_df: Original DataFrame with all markets before AI filtering
        filtered_df: DataFrame with markets that passed AI filtering
        
    Returns:
        DataFrame with columns: question, token1, token2, ai_decision, ai_reason, passed_filtering
    """
    try:
        # Get all original questions
        filtered_questions = set(filtered_df["question"].astype(str).tolist())
        
        # Create evaluation DataFrame
        evaluation_data = []
        
        for _, row in original_df.iterrows():
            question = str(row.get("question", ""))
            token1 = str(row.get("token1", ""))
            token2 = str(row.get("token2", ""))
            
            # Determine AI decision based on filtering result
            if question in filtered_questions:
                ai_decision = "ELIGIBLE"
                ai_reason = "Passed AI filtering criteria"
                passed_filtering = True
            else:
                ai_decision = "AVOID"
                ai_reason = "Marked as AVOID by AI (insider-prone, ambiguous, manipulable, etc.)"
                passed_filtering = False
            
            evaluation_data.append({
                "question": question,
                "token1": token1,
                "token2": token2,
                "ai_decision": ai_decision,
                "ai_reason": ai_reason,
                "passed_filtering": passed_filtering
            })
        
        # Create DataFrame and sort by decision (ELIGIBLE first, then AVOID)
        evaluation_df = pd.DataFrame(evaluation_data)
        evaluation_df = evaluation_df.sort_values(["passed_filtering", "question"], ascending=[False, True]).reset_index(drop=True)
        
        logger.info(f"Created AI evaluation data for {len(evaluation_df)} markets")
        logger.info(f"AI decisions: {len(evaluation_df[evaluation_df['ai_decision'] == 'ELIGIBLE'])} ELIGIBLE, {len(evaluation_df[evaluation_df['ai_decision'] == 'AVOID'])} AVOID")
        
        return evaluation_df
        
    except Exception as e:
        logger.error(f"Error creating AI evaluation data: {e}")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=["question", "token1", "token2", "ai_decision", "ai_reason", "passed_filtering"])


def log_filtering_changes(original_df: pd.DataFrame, filtered_df: pd.DataFrame, 
                         selected_markets_df: pd.DataFrame) -> dict:
    """
    Log all changes made by AI filtering for transparency.
    
    Returns:
        Dictionary with summary of filtering changes
    """
    original_questions = set(original_df["question"].tolist())
    filtered_questions = set(filtered_df["question"].tolist())
    
    # Get currently traded markets
    current_questions = set()
    if "question" in selected_markets_df.columns:
        current_questions = set(selected_markets_df["question"].astype(str).tolist())
    elif "market" in selected_markets_df.columns:
        current_questions = set(selected_markets_df["market"].astype(str).tolist())
    
    # What was removed
    removed = original_questions - filtered_questions
    removed_current = removed & current_questions
    removed_new = removed - current_questions
    
    # What was kept
    kept = filtered_questions
    kept_current = kept & current_questions
    kept_new = kept - current_questions
    
    logger.info("=== AI Filtering Summary ===")
    logger.info(f"Total markets: {len(original_questions)}")
    logger.info(f"Markets kept: {len(kept)} ({len(kept_current)} current, {len(kept_new)} new)")
    logger.info(f"Markets removed: {len(removed)} ({len(removed_current)} current, {len(removed_new)} new)")
    
    if removed_current:
        logger.warning("⚠️  AI wants to remove currently traded markets:")
        for market in list(removed_current)[:10]:  # Show first 10
            logger.warning(f"  - {market}")
        if len(removed_current) > 10:
            logger.warning(f"  ... and {len(removed_current) - 10} more")
    
    return {
        "removed_current": removed_current,
        "removed_new": removed_new,
        "kept_current": kept_current,
        "kept_new": kept_new
    }


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
                        t1 = clean_token_string(str(v[0]))
                    if len(v) > 1:
                        t2 = clean_token_string(str(v[1]))
                elif isinstance(v, str):
                    parts = [p.strip() for p in v.split(",") if p.strip()]
                    if len(parts) > 0:
                        t1 = clean_token_string(parts[0])
                    if len(parts) > 1:
                        t2 = clean_token_string(parts[1])
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
    
    # Ensure token columns are explicitly strings to prevent scientific notation
    out["token1"] = out["token1"].astype(str)
    out["token2"] = out["token2"].astype(str)
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
                # Read currently selected markets to protect them from AI removal
                try:
                    selected_markets_df = read_sheet(spreadsheet, "Selected Markets")
                    logger.info(f"Found {len(selected_markets_df)} currently selected markets to protect")
                except Exception:
                    logger.warning("Could not read Selected Markets sheet; no protection will be applied")
                    selected_markets_df = pd.DataFrame()
                
                # Run AI filtering and capture raw evaluation results
                filtered = remove_markets_to_avoid(pairs_df=pairs_df)
                
                # Get raw AI evaluation data for transparency
                ai_evaluation_df = get_ai_evaluation_data(pairs_df, filtered)
                
                # Apply Grandfather Clause: protect currently traded markets from AI removal
                if not selected_markets_df.empty:
                    filtered = protect_currently_traded_markets(filtered, selected_markets_df, ms_df)
                    
                    # Log filtering changes for transparency
                    log_filtering_changes(pairs_df, filtered, selected_markets_df)
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
                    
                    # Ensure essential columns are always first in the correct order
                    essential_cols = ["question", "token1", "token2", "condition_id"]
                    
                    # Additional columns in preferred order
                    additional_cols = [
                        "rules", "liquidity", "volume24hr", "avg_volume_1wk", "avg_volume_1mo",
                        "trend_score", "mm_score", "best_bid", "best_ask", "description",
                    ]
                    
                    # Build final column order: essential first, then additional if present
                    final_cols = []
                    
                    # Add essential columns first (in order)
                    for col in essential_cols:
                        if col in merged.columns:
                            final_cols.append(col)
                        else:
                            logger.warning(f"Essential column '{col}' missing from merged data")
                    
                    # Add additional columns if present
                    for col in additional_cols:
                        if col in merged.columns:
                            final_cols.append(col)
                    
                    # Create output DataFrame with guaranteed column order
                    out_df = merged[final_cols].copy()
                    
                    # Clean token strings in the final output before writing to sheet
                    out_df["token1"] = out_df["token1"].apply(clean_token_string)
                    out_df["token2"] = out_df["token2"].apply(clean_token_string)
                    
                    # Ensure token columns are explicitly strings to prevent scientific notation
                    out_df["token1"] = out_df["token1"].astype(str)
                    out_df["token2"] = out_df["token2"].astype(str)
                    
                    # Sort by mm_score in descending order (best markets first)
                    if "mm_score" in out_df.columns:
                        # Handle any NaN values in mm_score before sorting
                        if out_df["mm_score"].isna().any():
                            logger.warning(f"Found {out_df['mm_score'].isna().sum()} markets with NaN mm_score - filling with 0")
                            out_df["mm_score"] = out_df["mm_score"].fillna(0)
                        
                        # Sort by mm_score in descending order
                        out_df = out_df.sort_values("mm_score", ascending=False).reset_index(drop=True)
                        
                        # Log sorting results
                        max_score = out_df["mm_score"].max()
                        min_score = out_df["mm_score"].min()
                        logger.info(f"✅ Sorted {len(out_df)} markets by mm_score (highest: {max_score:,.0f}, lowest: {min_score:,.0f})")
                        
                        # Show top 3 markets for verification
                        if len(out_df) > 0:
                            top_markets = out_df.head(3)
                            logger.info("🏆 Top 3 markets by mm_score:")
                            for i, (_, row) in enumerate(top_markets.iterrows(), 1):
                                logger.info(f"  {i}. {row.get('question', 'Unknown')[:50]}... (score: {row['mm_score']:,.0f})")
                    else:
                        logger.warning("⚠️  mm_score column not found - cannot sort by market making score")
                else:
                    # Create empty DataFrame with consistent column ordering
                    out_df = pd.DataFrame(columns=[
                        "question", "token1", "token2", "condition_id",
                        "rules", "liquidity", "volume24hr", "avg_volume_1wk", "avg_volume_1mo",
                        "trend_score", "mm_score", "best_bid", "best_ask", "description",
                    ])
                write_sheet(spreadsheet, "Filtered Markets", out_df)
                logger.info("Wrote %d rows to sheet 'Filtered Markets'", len(out_df))
                
                # Write AI Evaluation sheet for transparency
                write_sheet(spreadsheet, "AI Evaluation", ai_evaluation_df)
                logger.info("Wrote %d rows to sheet 'AI Evaluation'", len(ai_evaluation_df))
    except Exception:
        logger.exception("Failed to write 'Market Size' sheet")


if __name__ == "__main__":
    main()
