import asyncio
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from integrations.merger_adapter import call_merger, MergeResult
from poly_utils.google_utils import get_spreadsheet
from store_selected_markets import read_sheet


logger = logging.getLogger("mm.merge")


@dataclass
class MergeConfig:
    merge_scan_interval_sec: int
    min_merge_usdc: float
    merge_chunk_usdc: float
    merge_max_retries: int
    merge_retry_backoff_ms: int
    dry_run: bool


def _to_bool(value) -> bool:
    try:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        return s in ("true", "1", "y", "yes", "t")
    except Exception:
        return False


def _fetch_positions_by_token(address: str) -> Dict[str, float]:
    import requests
    out: Dict[str, float] = {}
    if not address:
        return out
    try:
        url = f"https://data-api.polymarket.com/positions?user={address}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = resp.json() or []
        for row in arr:
            token = str(row.get("token_id") or row.get("asset_id") or row.get("id") or "").strip()
            if not token:
                continue
            shares = row.get("shares")
            if shares is None:
                shares = row.get("balance") or row.get("qty") or 0.0
            try:
                out[token] = float(shares)
            except Exception:
                out[token] = 0.0
    except Exception:
        logger.exception("Failed to fetch positions for %s", address)
    return out


class MergeLedger:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "mm_state.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS merge_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    condition_id TEXT,
                    amount_merged_6dp INTEGER,
                    neg_risk INTEGER,
                    tx_hash TEXT,
                    exit_code INTEGER
                );
                """
            )

    def record(self, condition_id: str, amount_6dp: int, neg_risk: bool, tx_hash: Optional[str], exit_code: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO merge_ledger(ts, condition_id, amount_merged_6dp, neg_risk, tx_hash, exit_code) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), condition_id, int(amount_6dp), 1 if neg_risk else 0, tx_hash or "", int(exit_code)),
            )


class MergeManager:
    def __init__(self, cfg: MergeConfig) -> None:
        self.cfg = cfg
        self.ledger = MergeLedger()
        # Cooldown per market to avoid hammering
        self._cooldown_until: Dict[str, float] = {}

        # Metrics
        from prometheus_client import Counter, Gauge  # type: ignore
        self.g_overlap = Gauge("merge_overlap_usdc", "Computed overlap (USDC)", ["condition_id"])  # type: ignore
        self.c_attempts = Counter("merge_attempts_total", "Merge attempts total")  # type: ignore
        self.c_success = Counter("merge_success_total", "Merge success total")  # type: ignore
        self.c_failures = Counter("merge_failures_total", "Merge failures total")  # type: ignore
        self.c_amount_total = Counter("merge_amount_merged_6dp_total", "Total merged amount (6dp)")  # type: ignore

    def _read_selected_markets(self) -> pd.DataFrame:
        ss = get_spreadsheet(read_only=False)
        df = read_sheet(ss, "Selected Markets")
        if df.empty:
            return df
        # Normalize types
        for col in ("token1", "token2", "condition_id"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        if "neg_risk" in df.columns:
            df["neg_risk"] = df["neg_risk"].apply(_to_bool)
        else:
            df["neg_risk"] = False
        return df

    async def scan_and_merge(self, wallet_address: str) -> None:
        try:
            sel = self._read_selected_markets()
            if sel.empty:
                logger.info("Merge scan: Selected Markets empty")
                return
            pos_by_token = await asyncio.to_thread(_fetch_positions_by_token, wallet_address)
            min_merge_6dp = int(self.cfg.min_merge_usdc * 1_000_000)
            chunk_6dp = int(self.cfg.merge_chunk_usdc * 1_000_000)

            for _, row in sel.iterrows():
                t1 = str(row.get("token1") or "").strip()
                t2 = str(row.get("token2") or "").strip()
                cid = str(row.get("condition_id") or "").strip()
                is_neg = _to_bool(row.get("neg_risk", False))
                if not t1 or not t2 or not cid:
                    continue
                # Cooldown per market
                now_ts = time.time()
                until = self._cooldown_until.get(cid)
                if until is not None and now_ts < until:
                    continue

                qty1 = float(pos_by_token.get(t1, 0.0))
                qty2 = float(pos_by_token.get(t2, 0.0))
                overlap_shares = max(0.0, min(qty1, qty2))
                amount_6dp = int(overlap_shares * 1_000_000)
                self.g_overlap.labels(condition_id=cid).set(amount_6dp / 1_000_000.0)
                if amount_6dp < min_merge_6dp:
                    continue

                # Chunked merge with retries and refresh after each chunk
                merged_so_far = 0
                retries = 0
                while merged_so_far < amount_6dp:
                    to_merge = min(chunk_6dp, amount_6dp - merged_so_far)
                    attempt_amount = to_merge
                    self.c_attempts.inc()
                    if self.cfg.dry_run:
                        logger.info("[DRY_RUN] merge cid=%s neg_risk=%s amount_6dp=%d", cid, is_neg, attempt_amount)
                        self.c_success.inc()
                        self.c_amount_total.inc(attempt_amount)
                        merged_so_far += to_merge
                        await asyncio.sleep(0.2)
                        continue
                    # Run merger helper
                    res: MergeResult = await asyncio.to_thread(call_merger, attempt_amount, cid, is_neg)
                    if res.success:
                        logger.info("Merge ok cid=%s neg_risk=%s amount_6dp=%d tx=%s", cid, is_neg, attempt_amount, (res.tx_hash or ""))
                        self.ledger.record(cid, attempt_amount, is_neg, res.tx_hash, res.exit_code)
                        self.c_success.inc()
                        self.c_amount_total.inc(attempt_amount)
                        merged_so_far += to_merge
                        # refresh positions and recompute new max
                        pos_by_token = await asyncio.to_thread(_fetch_positions_by_token, wallet_address)
                        qty1 = float(pos_by_token.get(t1, 0.0))
                        qty2 = float(pos_by_token.get(t2, 0.0))
                        overlap_shares = max(0.0, min(qty1, qty2))
                        amount_6dp = int(overlap_shares * 1_000_000)
                        self.g_overlap.labels(condition_id=cid).set(amount_6dp / 1_000_000.0)
                        retries = 0
                    else:
                        logger.warning("Merge failed cid=%s code=%s stderr=%s", cid, res.exit_code, res.stderr[:200])
                        self.ledger.record(cid, attempt_amount, is_neg, res.tx_hash, res.exit_code)
                        self.c_failures.inc()
                        retries += 1
                        if retries >= max(1, self.cfg.merge_max_retries):
                            # cooldown this market
                            backoff = (self.cfg.merge_retry_backoff_ms / 1000.0) * (1.0 + random.random())
                            self._cooldown_until[cid] = time.time() + max(60.0, backoff)
                            break
                        backoff = (self.cfg.merge_retry_backoff_ms / 1000.0) * (1.0 + random.random())
                        await asyncio.sleep(backoff)
                    # small jitter between chunks
                    await asyncio.sleep(0.2 + random.random() * 0.3)
        except Exception:
            logger.exception("scan_and_merge error")

    async def run_loop(self, wallet_address: str) -> None:
        # Periodic scan loop
        interval = max(10, int(self.cfg.merge_scan_interval_sec))
        while True:
            await self.scan_and_merge(wallet_address)
            await asyncio.sleep(interval)
