#!/usr/bin/env python3
import os
import logging
from typing import Dict, List

import requests
import pandas as pd
from decimal import Decimal, InvalidOperation, localcontext
from dotenv import load_dotenv
from poly_utils.google_utils import get_spreadsheet
from gspread_dataframe import set_with_dataframe


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _stringify_token_id(value) -> str:
    """Return a stable string for token_id without scientific notation.

    - Empty/NaN -> ""
    - Numeric values -> integer-like string without exponent (e.g., 1.23E+21 -> "1230000000000000000000")
    - Other values -> str(value)
    """
    if pd.isna(value):
        return ""
    # Fast-path for strings
    if isinstance(value, str):
        return value
    # Ints are safe to stringify directly
    if isinstance(value, int):
        return str(value)
    # Handle floats or other numeric-like values via Decimal to avoid scientific notation
    try:
        with localcontext() as ctx:
            ctx.prec = 50
            d = Decimal(str(value))
            # Quantize to an integer if it's integral, otherwise keep as is (but token_ids should be integral)
            if d == d.to_integral_value():
                d = d.to_integral_value()
            return format(d, 'f')
    except (InvalidOperation, ValueError, TypeError):
        return str(value)


def fetch_activity_trades(wallet: str, per_page_limit: int = 500) -> pd.DataFrame:
    """Fetch all TRADE activity rows for a wallet from the Data-API /activity endpoint with pagination."""
    base_url = "https://data-api.polymarket.com/activity"
    all_rows: List[dict] = []
    offset = 0
    limit = min(500, max(1, int(per_page_limit)))
    while True:
        params: Dict[str, object] = {
            "user": wallet,
            "type": "TRADE",
            "limit": limit,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        try:
            r = requests.get(base_url, params=params, timeout=20)
            if not r.ok:
                logging.warning("/activity GET failed %s -> %s %s", r.url, r.status_code, r.text)
                break
            data = r.json()
            page = data if isinstance(data, list) else data.get("data", [])
            if not page:
                break
            for row in page:
                if isinstance(row, dict) and str(row.get("type", "")).upper() == "TRADE":
                    all_rows.append(row)
            offset += len(page)
            if len(page) < limit:
                break
        except Exception as e:
            logging.warning("/activity request error: %s", str(e))
            break

    if not all_rows:
        return pd.DataFrame(columns=["asset", "timestamp", "size", "price", "side"])  # consistent schema
    return pd.DataFrame(all_rows)


def fetch_reward_activities(wallet: str, per_page_limit: int = 500) -> pd.DataFrame:
    """Fetch all REWARD activity rows for a wallet from the Data-API /activity endpoint with pagination."""
    base_url = "https://data-api.polymarket.com/activity"
    all_rows: List[dict] = []
    offset = 0
    limit = min(500, max(1, int(per_page_limit)))
    while True:
        params: Dict[str, object] = {
            "user": wallet,
            "type": "REWARD",
            "limit": limit,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        try:
            r = requests.get(base_url, params=params, timeout=20)
            if not r.ok:
                logging.debug("/activity REWARD GET %s -> %s %s", r.url, r.status_code, r.text)
                break
            data = r.json()
            page = data if isinstance(data, list) else data.get("data", [])
            if not page:
                break
            for row in page:
                if isinstance(row, dict) and str(row.get("type", "")).upper() == "REWARD":
                    all_rows.append(row)
            offset += len(page)
            if len(page) < limit:
                break
        except Exception as e:
            logging.debug("/activity REWARD request error: %s", str(e))
            break

    if not all_rows:
        return pd.DataFrame(columns=["market", "amount", "timestamp"])  # consistent schema
    return pd.DataFrame(all_rows)


def compute_earnings_by_market(reward_df: pd.DataFrame) -> pd.DataFrame:
    """Map reward activity rows to per-market earnings.

    Returns DataFrame with columns: market, Earnings
    """
    if reward_df is None or reward_df.empty:
        return pd.DataFrame(columns=["market", "Earnings"])  # empty schema

    # Identify market/title column heuristically
    market_col = None
    for c in ("title", "market", "question"):
        if c in reward_df.columns:
            market_col = c
            break
    if not market_col:
        # If we cannot attribute to markets, return empty; earnings stay 0
        return pd.DataFrame(columns=["market", "Earnings"])  # empty

    # Identify amount/value column heuristically
    amount_series = None
    for c in ("amount", "value", "quantity", "reward", "usdc_amount", "usdcAmount", "usdc"):
        if c in reward_df.columns:
            amount_series = reward_df[c]
            break
    if amount_series is None:
        return pd.DataFrame(columns=["market", "Earnings"])  # empty

    amounts = pd.to_numeric(amount_series, errors="coerce").fillna(0.0).astype(float)
    markets = reward_df[market_col].astype(str)
    df = pd.DataFrame({"market": markets, "Earnings": amounts})
    df = df[df["market"] != ""]
    grouped = df.groupby("market", as_index=False)["Earnings"].sum()
    return grouped

def map_activity_to_trades(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize activity rows to trades with the required columns for PnL.

    Output columns: token_id, timestamp, side, shares, price
    """
    if activity_df is None or activity_df.empty:
        return pd.DataFrame(columns=["token_id", "timestamp", "side", "shares", "price"])  # empty schema

    # Resolve columns robustly
    token_col = None
    for c in ("asset", "token_id", "tokenId", "asset_id"):
        if c in activity_df.columns:
            token_col = c
            break
    size_col = "size" if "size" in activity_df.columns else None
    price_col = "price" if "price" in activity_df.columns else None
    side_col = "side" if "side" in activity_df.columns else None
    ts_col = None
    for c in ("timestamp", "ts", "created_at", "createdAt"):
        if c in activity_df.columns:
            ts_col = c
            break

    if not (token_col and size_col and price_col and side_col and ts_col):
        return pd.DataFrame(columns=["token_id", "timestamp", "side", "shares", "price"])  # schema with no rows

    trades = pd.DataFrame({
        "token_id": activity_df[token_col].astype(str),
        "shares": pd.to_numeric(activity_df[size_col], errors="coerce").fillna(0.0).astype(float),
        "price": pd.to_numeric(activity_df[price_col], errors="coerce").fillna(0.0).astype(float),
        "timestamp": pd.to_numeric(activity_df[ts_col], errors="coerce").fillna(0).astype(int),
        "side": activity_df[side_col].astype(str).str.upper(),
    })
    trades = trades[(trades["token_id"] != "") & (trades["shares"] > 0) & (trades["price"] > 0)]
    trades = trades.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
    return trades


def build_pnl_rows_from_activity(activity: pd.DataFrame) -> pd.DataFrame:
    """Map raw activity rows to the PnL sheet schema without computing PnL.

    Columns: type, token_id, market, side, open_date, date, open_price, price, shares, pnl, fees
    """
    columns = [
        "type",
        "token_id",
        "market",
        "yes/no",
        "side",
        "open_date",
        "date",
        "open_price",
        "price",
        "shares",
        "pnl",
        "fees",
    ]
    if activity is None or activity.empty:
        return pd.DataFrame(columns=columns)

    idx = activity.index

    def col(name: str, default_value):
        return activity[name] if name in activity.columns else pd.Series([default_value] * len(idx), index=idx)

    token_series = col("asset", "").astype(str)
    market_series = col("title", "").astype(str)
    outcome_series = col("outcome", "").astype(str)
    side_series = col("side", "").astype(str).str.upper()
    ts_series = pd.to_numeric(col("timestamp", 0), errors="coerce").fillna(0).astype(int)
    price_series = pd.to_numeric(col("price", 0), errors="coerce").fillna(0.0).astype(float)
    shares_series = pd.to_numeric(col("size", 0), errors="coerce").fillna(0.0).astype(float)
    # Signed shares: SELL is negative
    shares_signed = shares_series.where(side_series != "SELL", -shares_series)

    dates = pd.to_datetime(ts_series, unit="s")
    # Notional value (USDC): buys negative (cash out), sells positive (cash in)
    notional_series = -shares_signed * price_series
    pnl_df = pd.DataFrame({
        "type": "trades",
        "token_id": token_series,
        "market": market_series,
        "yes/no": outcome_series,
        "side": side_series,
        "open_date": dates,
        "date": dates,
        "open_price": price_series,
        "price": notional_series,
        "shares": shares_signed,
        "pnl": 0.0,
        "fees": 0.0,
    })
    return pnl_df[columns]


def get_best_price(token_id: str, side: str) -> float:
    """Fetch best price from CLOB for a single token/side.

    Args:
        token_id: ERC1155 token id
        side: "buy" for best bid, "sell" for best ask

    Returns:
        float price (0.0 on failure)
    """
    try:
        r = requests.get(
            "https://clob.polymarket.com/price",
            params={"token_id": str(token_id), "side": side},
            timeout=10,
        )
        if not r.ok:
            logging.debug("price GET %s -> %s %s", r.url, r.status_code, r.text)
            return 0.0
        data = r.json()
        p = data.get("price") if isinstance(data, dict) else None
        return float(p) if p is not None else 0.0
    except Exception as e:
        logging.debug("price GET error for %s/%s: %s", token_id, side, str(e))
        return 0.0


def main() -> None:
    load_dotenv()
    setup_logging()

    wallet = (os.getenv("BROWSER_WALLET") or os.getenv("BROWSER_ADDRESS") or "").strip()
    if not wallet:
        logging.error("BROWSER_WALLET not set in environment")
        return

    logging.info("Fetching all activity trades for %s", wallet)
    activity = fetch_activity_trades(wallet, per_page_limit=500)
    logging.info("Fetched %d trade activity rows", len(activity))
    if activity.empty:
        logging.info("No trades found for wallet")
        return

    pnl_rows = build_pnl_rows_from_activity(activity)
    # Sort by date before writing
    try:
        pnl_rows["date"] = pd.to_datetime(pnl_rows["date"], errors="coerce")
    except Exception:
        pass
    pnl_rows = pnl_rows.sort_values("date").reset_index(drop=True)
    # Write to Google Sheet 'PnL' (merge with existing and rewrite to avoid append issues)
    try:
        spreadsheet = get_spreadsheet()
        # Determine existence without relying on exception paths
        try:
            sheet_titles = [ws.title for ws in spreadsheet.worksheets()]
        except Exception:
            sheet_titles = []

        cols = [
            "type",
            "token_id",
            "market",
            "yes/no",
            "side",
            "open_date",
            "date",
            "open_price",
            "price",
            "shares",
            "pnl",
            "fees",
        ]

        if "Trade History" in sheet_titles:
            wk = spreadsheet.worksheet("Trade History")
            try:
                existing_df = pd.DataFrame(wk.get_all_records())
            except Exception:
                existing_df = pd.DataFrame()
            if existing_df.empty:
                existing_df = pd.DataFrame(columns=cols)
            else:
                # Align to expected columns; add any missing ones
                existing_df = existing_df.reindex(columns=cols)
                # Normalize date types to datetime for stable sorting/merge
                for dc in ("open_date", "date"):
                    if dc in existing_df.columns:
                        try:
                            existing_df[dc] = pd.to_datetime(existing_df[dc], errors="coerce")
                        except Exception:
                            pass
                # Ensure token_id is string to avoid scientific notation in Sheets
                if "token_id" in existing_df.columns:
                    existing_df["token_id"] = existing_df["token_id"].apply(_stringify_token_id)
            # Also normalize token_id in newly built rows
            if "token_id" in pnl_rows.columns:
                pnl_rows["token_id"] = pnl_rows["token_id"].apply(_stringify_token_id)
            combined = pd.concat([existing_df, pnl_rows], ignore_index=True)

            def build_keys(df: pd.DataFrame) -> pd.Series:
                if df.empty:
                    return pd.Series([], dtype=str)
                token = df.get("token_id", pd.Series([""] * len(df))).astype(str)
                side = df.get("side", pd.Series([""] * len(df))).astype(str)
                open_price = pd.to_numeric(df.get("open_price", 0.0), errors="coerce").fillna(0.0).round(6)
                shares = pd.to_numeric(df.get("shares", 0.0), errors="coerce").fillna(0.0).round(6)
                try:
                    open_ts = pd.to_datetime(df["open_date"]).astype("int64") // 10**9
                except Exception:
                    open_ts = pd.to_datetime(df.get("open_date", 0), errors="coerce").astype("int64").fillna(0).astype(int) // 10**9
                return token + "|" + side + "|" + open_price.astype(str) + "|" + shares.astype(str) + "|" + open_ts.astype(str)

            keys = build_keys(combined)
            # Prefer the newest computation for a given trade signature
            deduped = combined.loc[~keys.duplicated(keep="last")].copy()
            # Ensure 'date' is datetime before sorting (can be NaT for some rows)
            if "date" in deduped.columns:
                try:
                    deduped["date"] = pd.to_datetime(deduped["date"], errors="coerce")
                except Exception:
                    pass
            deduped = deduped.sort_values("date").reset_index(drop=True)
            # Final guard: keep token_id as string before writing
            if "token_id" in deduped.columns:
                deduped["token_id"] = deduped["token_id"].apply(_stringify_token_id)
            set_with_dataframe(wk, deduped[cols], include_index=False, include_column_header=True, resize=True)
            logging.info("Wrote %d rows to Google Sheet 'Trade History' (merged)", len(deduped))
        else:
            wk = spreadsheet.add_worksheet(title="Trade History", rows=1000, cols=20)
            # Ensure token_id is string on first write too
            if "token_id" in pnl_rows.columns:
                pnl_rows["token_id"] = pnl_rows["token_id"].apply(_stringify_token_id)
            set_with_dataframe(wk, pnl_rows, include_index=False, include_column_header=True, resize=True)
            logging.info("Wrote %d rows to new Google Sheet 'Trade History'", len(pnl_rows))
    except Exception as e:
        logging.info("Failed to write trades to Google Sheet: %s", str(e))

    # Build and write Summary: one row per market with cumulative shares and selection flag
    try:
        spreadsheet = get_spreadsheet()
        # Re-read PnL to compute cumulative shares and realized/unrealized PnL across all entries
        try:
            wk_pnl = spreadsheet.worksheet("Trade History")
            pnl_all_df = pd.DataFrame(wk_pnl.get_all_records())
        except Exception:
            pnl_all_df = pnl_rows.copy()
        if pnl_all_df.empty:
            logging.info("PnL empty; skipping Summary update")
            return
        # Normalize token_id to string for any downstream usage (e.g., pricing lookups)
        if "token_id" in pnl_all_df.columns:
            pnl_all_df["token_id"] = pnl_all_df["token_id"].apply(_stringify_token_id)
        # Ensure numeric fields
        pnl_all_df["shares"] = pd.to_numeric(pnl_all_df.get("shares", 0), errors="coerce").fillna(0.0)
        pnl_all_df["price"] = pd.to_numeric(pnl_all_df.get("price", 0), errors="coerce").fillna(0.0)
        # Aggregations per market AND outcome (treat YES/NO separately)
        group_keys = ["market", "yes/no"] if "yes/no" in pnl_all_df.columns else ["market"]
        summary = pnl_all_df.groupby(group_keys, dropna=False)["shares"].sum().reset_index()
        if "yes/no" in summary.columns:
            summary = summary.rename(columns={"shares": "position_size", "yes/no": "yes_no"})
        else:
            summary = summary.rename(columns={"shares": "position_size"})

        realized = pnl_all_df.groupby(group_keys, dropna=False)["price"].sum().reset_index()
        if "yes/no" in realized.columns:
            realized = realized.rename(columns={"price": "Realized PnL", "yes/no": "yes_no"})
            summary = summary.merge(realized, on=["market", "yes_no"], how="left")
        else:
            realized = realized.rename(columns={"price": "Realized PnL"})
            summary = summary.merge(realized, on="market", how="left")

        # Ensure all markets present in data are listed, even if position_size is exactly 0
        try:
            all_markets = (
                pnl_all_df.get("market")
                .astype(str)
                .dropna()
                .unique()
            )
            if len(all_markets) > 0:
                summary = summary.set_index("market")
                for m in all_markets:
                    if str(m) not in summary.index:
                        summary.loc[str(m)] = {
                            "position_size": 0.0,
                            "Realized PnL": 0.0,
                        }
                summary = summary.reset_index()
        except Exception:
            pass

        # Compute Unrealized PnL from best prices for markets with open positions
        try:
            # Map (market, yes/no) -> representative token_id and side context using the latest trade
            pnl_all_df["date_ts"] = pd.to_datetime(pnl_all_df.get("date")).astype("int64") // 10**9
            if "yes/no" in pnl_all_df.columns:
                latest_by_key = pnl_all_df.sort_values("date_ts").groupby(["market", "yes/no"]).tail(1)
            else:
                latest_by_key = pnl_all_df.sort_values("date_ts").groupby(["market"]).tail(1)
        except Exception:
            if "yes/no" in pnl_all_df.columns:
                latest_by_key = pnl_all_df.groupby(["market", "yes/no"]).head(1)
            else:
                latest_by_key = pnl_all_df.groupby(["market"]).head(1)

        market_to_token = {}
        token_col = "token_id" if "token_id" in latest_by_key.columns else None
        for _, r in latest_by_key.iterrows():
            m = str(r.get("market", ""))
            y = str(r.get("yes/no", "")) if "yes/no" in latest_by_key.columns else ""
            if not m:
                continue
            if token_col and str(r.get(token_col, "")):
                market_to_token[(m, y)] = str(r.get(token_col, ""))

        # Merge position_size into a frame to compute unrealized
        key_cols = ["market", "yes_no"] if "yes_no" in summary.columns else ["market"]
        pos_df = summary[key_cols + ["position_size"]].copy()
        pos_df["best_price"] = 0.0
        pos_df["Unrealized PnL"] = 0.0

        for i, row in pos_df.iterrows():
            mkt = str(row["market"])
            size = float(row["position_size"]) if row["position_size"] is not None else 0.0
            yes_no_val = str(row.get("yes_no", "")) if "yes_no" in pos_df.columns else ""
            token = market_to_token.get((mkt, yes_no_val)) if ("yes_no" in pos_df.columns) else market_to_token.get(mkt)
            if not token:
                continue
            side = "sell" if size > 0 else "buy"  # long -> sell to close, short -> buy to close
            best = get_best_price(token, side)
            pos_df.at[i, "best_price"] = best
            # Unrealized is negative of liquidation value per request
            pos_df.at[i, "Unrealized PnL"] = -(best * size)

        if "yes_no" in summary.columns:
            summary = summary.merge(pos_df[["market", "yes_no", "Unrealized PnL"]], on=["market", "yes_no"], how="left")
        else:
            summary = summary.merge(pos_df[["market", "Unrealized PnL"]], on="market", how="left")
        summary["Unrealized PnL"] = pd.to_numeric(summary.get("Unrealized PnL", 0.0), errors="coerce").fillna(0.0)

        # Fetch and merge Earnings (rewards) per market; default to 0 when absent
        try:
            rewards_df = fetch_reward_activities(wallet, per_page_limit=500)
        except Exception:
            rewards_df = pd.DataFrame()
        earnings_by_market = compute_earnings_by_market(rewards_df)
        if not earnings_by_market.empty:
            summary = summary.merge(earnings_by_market, on="market", how="left")
        if "Earnings" not in summary.columns:
            summary["Earnings"] = 0.0
        summary["Earnings"] = pd.to_numeric(summary.get("Earnings", 0.0), errors="coerce").fillna(0.0)

        # PnL includes Realized, Unrealized and Earnings
        summary["Pnl"] = summary.get("Realized PnL", 0.0) + summary.get("Unrealized PnL", 0.0) + summary.get("Earnings", 0.0)

        # Preserve existing yes_no from grouping; only backfill if missing
        if "yes_no" not in summary.columns or summary["yes_no"].isna().all():
            token_to_outcome = {}
            try:
                wk_all_mkts = spreadsheet.worksheet("All Markets")
                all_mkts_df = pd.DataFrame(wk_all_mkts.get_all_records())
                if not all_mkts_df.empty:
                    for _, r in all_mkts_df.iterrows():
                        t1 = str(r.get("token1", ""))
                        t2 = str(r.get("token2", ""))
                        a1 = str(r.get("answer1", ""))
                        a2 = str(r.get("answer2", ""))
                        if t1:
                            token_to_outcome[t1] = a1
                        if t2:
                            token_to_outcome[t2] = a2
            except Exception:
                pass

            yes_no_col = []
            try:
                token_shares = pnl_all_df.groupby(["market", "token_id"], dropna=False)["shares"].sum().reset_index()
                idx = (
                    token_shares.assign(abs_shares=token_shares["shares"].abs())
                    .sort_values(["market", "abs_shares"]) 
                    .groupby("market")
                    .tail(1)
                )
                m2t = {str(r["market"]): str(r["token_id"]) for _, r in idx.iterrows()}
                for _, r in summary.iterrows():
                    m = str(r.get("market", ""))
                    tok = m2t.get(m, "")
                    yes_no_col.append(token_to_outcome.get(tok, ""))
            except Exception:
                yes_no_col = [""] * len(summary)
            summary["yes_no"] = yes_no_col

        # Determine if market is in Selected Markets sheet
        try:
            wk_sel = spreadsheet.worksheet("Selected Markets")
            selected_df = pd.DataFrame(wk_sel.get_all_records())
        except Exception:
            selected_df = pd.DataFrame()

        selected_set = set()
        if not selected_df.empty:
            # Prefer 'question' column; fallback to 'market' if present
            if "question" in selected_df.columns:
                selected_set = set(str(x) for x in selected_df["question"].astype(str).tolist())
            elif "market" in selected_df.columns:
                selected_set = set(str(x) for x in selected_df["market"].astype(str).tolist())

        summary["marketInSelected"] = summary["market"].astype(str).apply(
            lambda m: "yes" if m in selected_set else "no"
        )

        # Write Summary sheet
        try:
            wk_summary = spreadsheet.worksheet("Position Summary")
            # Upsert: overwrite rows for markets we computed, keep others
            try:
                existing_sum = pd.DataFrame(wk_summary.get_all_records())
            except Exception:
                existing_sum = pd.DataFrame()
            if not existing_sum.empty and "market" in existing_sum.columns:
                if "position_size" not in existing_sum.columns:
                    existing_sum["position_size"] = 0.0
                if "Realized PnL" not in existing_sum.columns:
                    existing_sum["Realized PnL"] = 0.0
                if "Unrealized PnL" not in existing_sum.columns:
                    existing_sum["Unrealized PnL"] = 0.0
                if "Earnings" not in existing_sum.columns:
                    existing_sum["Earnings"] = 0.0
                if "Pnl" not in existing_sum.columns:
                    existing_sum["Pnl"] = 0.0
                if "marketInSelected" not in existing_sum.columns:
                    existing_sum["marketInSelected"] = "no"
                if "yes_no" not in existing_sum.columns:
                    existing_sum["yes_no"] = ""
                existing_sum = existing_sum[~existing_sum["market"].astype(str).isin(summary["market"].astype(str))]
                out_df = pd.concat([
                    existing_sum[["market", "position_size", "yes_no", "Realized PnL", "Unrealized PnL", "Earnings", "Pnl", "marketInSelected"]],
                    summary[["market", "position_size", "yes_no", "Realized PnL", "Unrealized PnL", "Earnings", "Pnl", "marketInSelected"]]
                ], ignore_index=True)
            else:
                out_df = summary[["market", "position_size", "yes_no", "Realized PnL", "Unrealized PnL", "Earnings", "Pnl", "marketInSelected"]]
        except Exception:
            wk_summary = spreadsheet.add_worksheet(title="Position Summary", rows=500, cols=10)
            out_df = summary[["market", "position_size", "yes_no", "Realized PnL", "Unrealized PnL", "Earnings", "Pnl", "marketInSelected"]]

        set_with_dataframe(
            wk_summary,
            out_df,
            include_index=False,
            include_column_header=True,
            resize=True,
        )
        logging.info("Wrote %d rows to Google Sheet 'Summary'", len(out_df))
    except Exception as e:
        logging.info("Failed to write Summary to Google Sheet: %s", str(e))


if __name__ == "__main__":
    main()
