import os
import time
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import datetime as dt

# Ensure local project imports work when run as a script
try:
    import sys
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except Exception:
    pass

from dotenv import load_dotenv  # type: ignore

# Reuse existing project functions for market data and order books
from data_updater.trading_utils import get_clob_client  # type: ignore
from data_updater.find_markets import get_order_book_with_retry  # type: ignore
from ai.filter_markets import get_all_markets  # type: ignore


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))


def ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            title TEXT,
            category TEXT,
            token1 TEXT,
            token2 TEXT,
            confidence REAL,
            p_model REAL,
            reward_min_size REAL,
            reward_paid INTEGER,
            significant INTEGER
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_markets_tokens ON markets(token1, token2)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_outcomes (
            token_id TEXT PRIMARY KEY,
            outcome TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_snapshots (
            ts_utc TEXT,
            token_id TEXT,
            p_actual REAL,
            PRIMARY KEY (ts_utc, token_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_prices_hourly (
            ts_utc TEXT,
            token1 TEXT,
            token2 TEXT,
            p_token1 REAL,
            p_token2 REAL,
            p_yes REAL,
            p_no REAL,
            PRIMARY KEY (ts_utc, token1, token2)
        )
        """
    )
    conn.commit()
    return conn


def load_markets_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logger.warning("CSV not found at %s; returning empty DataFrame", csv_path)
        return pd.DataFrame()
    try:
        if csv_path.lower().endswith(".json"):
            with open(csv_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return pd.DataFrame.from_records(data["results"])  # type: ignore[arg-type]
            return pd.DataFrame.from_records(data) if isinstance(data, list) else pd.DataFrame()
        return pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed to load markets CSV: %s", csv_path)
        return pd.DataFrame()


def upsert_markets(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    cols = [c for c in [
        "question", "title", "category", "token1", "token2",
        "confidence", "p_model", "reward_min_size", "reward_paid", "significant"
    ] if c in df.columns]
    batch = df[cols].copy()
    batch["reward_paid"] = batch.get("reward_paid", False).astype(int)
    sql = (
        "INSERT OR IGNORE INTO markets (question, title, category, token1, token2, confidence, p_model, reward_min_size, reward_paid, significant) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    values = [
        (
            str(r.get("question", "")),
            str(r.get("title", "")),
            str(r.get("category", "")),
            str(r.get("token1", "")),
            str(r.get("token2", "")),
            float(r.get("confidence", 0.0)) if r.get("confidence") is not None else None,
            float(r.get("p_model", 0.0)) if r.get("p_model") is not None else None,
            float(r.get("reward_min_size", 0.0)) if r.get("reward_min_size") is not None else None,
            int(1 if r.get("reward_paid") else 0),
            int(1 if r.get("significant") else 0),
        )
        for _, r in batch.iterrows()
    ]
    conn.executemany(sql, values)
    conn.commit()


def build_token_outcome_map() -> Dict[str, str]:
    """Build token_id -> outcome label map by fetching current markets."""
    out: Dict[str, str] = {}
    try:
        df = get_all_markets()
        if df is None or df.empty:
            return out
        for _, row in df.iterrows():
            tokens = row.get("tokens", [])
            if isinstance(tokens, list):
                for t in tokens:
                    if isinstance(t, dict):
                        tid = t.get("token_id")
                        outc = t.get("outcome") or t.get("answer") or t.get("title")
                        if tid:
                            out[str(tid)] = str(outc or "")
        return out
    except Exception:
        logger.warning("Failed to build token_outcome map")
        return out


def upsert_token_outcomes(conn: sqlite3.Connection, token_to_outcome: Dict[str, str]) -> None:
    if not token_to_outcome:
        return
    sql = "INSERT OR REPLACE INTO token_outcomes (token_id, outcome) VALUES (?, ?)"
    conn.executemany(sql, [(tid, token_to_outcome.get(tid, "")) for tid in token_to_outcome.keys()])
    conn.commit()


def fetch_midpoint_for_token(client, token_id: str) -> float:
    try:
        book = get_order_book_with_retry(client, token_id)
        raw_bids = getattr(book, "bids", []) or []
        raw_asks = getattr(book, "asks", []) or []

        def price(e) -> float:
            try:
                if isinstance(e, dict):
                    return float(e.get("price", 0.0))
                return float(getattr(e, "price", 0.0))
            except Exception:
                return 0.0

        bids = [price(e) for e in raw_bids if price(e) > 0.0]
        asks = [price(e) for e in raw_asks if price(e) > 0.0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        if bids:
            return max(bids)
        if asks:
            return min(asks)
        return 0.0
    except Exception:
        logger.debug("Order book fetch failed for token=%s", token_id)
        return 0.0


def snapshot_prices(
    conn: sqlite3.Connection,
    ts_utc: str,
    markets_df: pd.DataFrame,
    token_outcomes: Dict[str, str]
) -> None:
    client = get_clob_client()
    if client is None:
        logger.warning("CLOB client unavailable; skipping price snapshot")
        return

    tokens_needed: List[str] = []
    for _, r in markets_df.iterrows():
        t1 = str(r.get("token1", ""))
        t2 = str(r.get("token2", ""))
        if t1:
            tokens_needed.append(t1)
        if t2:
            tokens_needed.append(t2)
    tokens_needed = list({t for t in tokens_needed if t})

    token_to_price: Dict[str, float] = {}
    for tid in tokens_needed:
        token_to_price[tid] = fetch_midpoint_for_token(client, tid)

    # Write per-token snapshots
    conn.executemany(
        "INSERT OR REPLACE INTO price_snapshots (ts_utc, token_id, p_actual) VALUES (?, ?, ?)",
        [(ts_utc, tid, float(token_to_price.get(tid, 0.0))) for tid in tokens_needed]
    )

    # Write per-market yes/no snapshots (best-effort mapping)
    rows: List[Tuple[str, str, str, float, float, Optional[float], Optional[float]]] = []
    for _, r in markets_df.iterrows():
        t1 = str(r.get("token1", ""))
        t2 = str(r.get("token2", ""))
        p1 = float(token_to_price.get(t1, 0.0)) if t1 else 0.0
        p2 = float(token_to_price.get(t2, 0.0)) if t2 else 0.0
        out1 = token_outcomes.get(t1, "").lower()
        out2 = token_outcomes.get(t2, "").lower()
        p_yes: Optional[float] = None
        p_no: Optional[float] = None
        if "yes" in out1 or ("no" not in out1 and "yes" in out2):
            # Prefer explicit label; if only one has 'yes', map accordingly
            p_yes = p1 if "yes" in out1 else p2
            p_no = p2 if "yes" in out1 else p1
        elif "no" in out1 or "no" in out2:
            # If only 'no' identifiable
            p_no = p1 if "no" in out1 else p2
            p_yes = p2 if "no" in out1 else p1
        rows.append((ts_utc, t1, t2, p1, p2, p_yes, p_no))

    conn.executemany(
        "INSERT OR REPLACE INTO market_prices_hourly (ts_utc, token1, token2, p_token1, p_token2, p_yes, p_no) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()


def run(
    csv_path: str = os.path.join(ROOT_DIR, "data", "odds_vs_market.csv"),
    db_path: str = os.path.join(ROOT_DIR, "data", "odds.db"),
    interval_seconds: int = 3600,
    once: bool = False,
) -> None:
    load_dotenv()
    configure_logging()

    conn = ensure_db(db_path)
    df = load_markets_csv(csv_path)
    if df is None or df.empty:
        logger.warning("No rows loaded from %s; continuing with empty markets table", csv_path)
    else:
        # Only keep core columns we need for linking and reference
        keep_cols = [
            "title", "category", "p_model", "confidence", "question", "token1", "token2",
            "reward_min_size", "reward_paid", "significant"
        ]
        present = [c for c in keep_cols if c in df.columns]
        df = df[present].copy()
        for c in ("token1", "token2"):
            if c in df.columns:
                df[c] = df[c].astype(str)
        upsert_markets(conn, df)

    # Refresh token outcome map initially
    token_to_outcome = build_token_outcome_map()
    upsert_token_outcomes(conn, token_to_outcome)

    snapshots_taken = 0
    while True:
        # Sleep until the next top-of-hour boundary (UTC)
        now = dt.datetime.utcnow()
        next_hour = (now + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = max(0, int((next_hour - now).total_seconds()))
        logger.info("Sleeping %ds until next top-of-hour %s", sleep_seconds, next_hour.strftime("%Y-%m-%dT%H:00:00Z"))
        time.sleep(sleep_seconds)

        # Snapshot timestamp exactly at the hour (UTC)
        ts_dt = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        ts = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            # Always re-pull markets set from DB to cover newly added rows
            markets_df = pd.read_sql_query("SELECT question, token1, token2 FROM markets", conn)
            # Refresh token outcomes every 6 hours
            if snapshots_taken % 6 == 0:
                token_to_outcome = build_token_outcome_map()
                upsert_token_outcomes(conn, token_to_outcome)
            snapshot_prices(conn, ts, markets_df, token_to_outcome)
            logger.info("Recorded hourly snapshot at %s for %d markets", ts, len(markets_df.index))
        except Exception:
            logger.exception("Failed to record snapshot at %s", ts)

        snapshots_taken += 1
        if once:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load odds_vs_market.csv into SQLite and record hourly p_actual for both sides")
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "odds_vs_market.csv"), help="Path to odds_vs_market.csv or JSON")
    parser.add_argument("--db", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "odds.db"), help="Path to SQLite DB file to create/update")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between snapshots (default 3600)")
    parser.add_argument("--once", action="store_true", help="Take one snapshot and exit (after initial load)")
    args = parser.parse_args()

    run(csv_path=args.csv, db_path=args.db, interval_seconds=args.interval, once=bool(args.once))