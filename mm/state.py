import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import time

from prometheus_client import Counter, Gauge, Histogram, start_http_server


@dataclass
class OrderRecord:
    order_id: str
    token_id: str
    side: str  # BUY/SELL
    price: float
    size: float
    timestamp: float


class StateStore:
    """Thread-safe sqlite-backed store for orders, fills, and positions."""

    def __init__(self, db_path: str = "mm_state.db", metrics_port: int = 9108) -> None:
        self._lock = threading.RLock()
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        # Metrics
        self.metrics_started = False
        self._start_metrics(metrics_port)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    token_id TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    ts REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id TEXT PRIMARY KEY,
                    ts REAL,
                    token_id TEXT,
                    side TEXT,
                    px REAL,
                    qty REAL,
                    fee REAL DEFAULT 0.0
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    token_id TEXT PRIMARY KEY,
                    yes_qty REAL DEFAULT 0.0,
                    no_qty REAL DEFAULT 0.0,
                    delta_usd REAL DEFAULT 0.0,
                    avg_px REAL DEFAULT 0.0
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders_active (
                    order_id TEXT PRIMARY KEY,
                    token_id TEXT,
                    side TEXT,
                    px REAL,
                    qty REAL,
                    ts_placed REAL,
                    ts_update REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pnl_daily (
                    date TEXT PRIMARY KEY,
                    realized REAL DEFAULT 0.0,
                    unrealized REAL DEFAULT 0.0,
                    fees REAL DEFAULT 0.0,
                    rebates REAL DEFAULT 0.0
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markouts (
                    fill_id TEXT PRIMARY KEY,
                    m3s REAL,
                    m10s REAL,
                    m30s REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS selection_snapshots (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    tokens TEXT
                );
                """
            )

    # Prometheus
    def _start_metrics(self, port: int) -> None:
        if not self.metrics_started:
            start_http_server(port)
            self.metrics_started = True

    orders_total = Counter("mm_orders_total", "Total orders placed", ["side"])  # type: ignore
    fills_total = Counter("mm_fills_total", "Total fills", ["side"])  # type: ignore
    inventory_gauge = Gauge("mm_inventory_shares", "Inventory shares", ["token_id"])  # type: ignore
    markout_hist = Histogram("mm_markout", "Markout PnL over window", buckets=(
        -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05
    ))  # type: ignore

    # Persistence API
    def record_order(self, rec: OrderRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO orders(order_id, token_id, side, price, size, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (rec.order_id, rec.token_id, rec.side, rec.price, rec.size, rec.timestamp),
            )
        self.orders_total.labels(side=rec.side).inc()

    def record_fill(self, fill_id: str, token_id: str, side: str, px: float, qty: float, fee: float, ts: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO fills(id, ts, token_id, side, px, qty, fee) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fill_id, ts, token_id, side, px, qty, fee),
            )
        try:
            self.fills_total.labels(side=side).inc()
        except Exception:
            pass

    def upsert_position(self, token_id: str, yes_qty: float, no_qty: float, delta_usd: float, avg_px: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO positions(token_id, yes_qty, no_qty, delta_usd, avg_px) VALUES (?, ?, ?, ?, ?)",
                (token_id, yes_qty, no_qty, delta_usd, avg_px),
            )
        try:
            self.inventory_gauge.labels(token_id=token_id).set(yes_qty - no_qty)
        except Exception:
            pass

    def get_positions(self) -> Dict[str, Tuple[float, float]]:
        with self._lock:
            cur = self._conn.execute("SELECT token_id, yes_qty, avg_px FROM positions")
            return {row[0]: (float(row[1]), float(row[2])) for row in cur.fetchall()}

    # Active orders DAO
    def upsert_active_order(self, order_id: str, token_id: str, side: str, px: float, qty: float, ts_placed: float, ts_update: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO orders_active(order_id, token_id, side, px, qty, ts_placed, ts_update) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, token_id, side, px, qty, ts_placed, ts_update),
            )

    def clear_active_orders_for_token(self, token_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM orders_active WHERE token_id=?", (token_id,))

    # PnL computation
    def rebuild_positions_from_fills(self) -> None:
        with self._lock:
            cur = self._conn.execute("SELECT token_id, side, px, qty FROM fills ORDER BY ts ASC")
            # Simple average price and net qty per token
            agg: Dict[str, Tuple[float, float]] = {}  # token -> (net_qty, cost)
            for token_id, side, px, qty in cur.fetchall():
                qty_signed = float(qty) if str(side).upper() == "BUY" else -float(qty)
                net, cost = agg.get(token_id, (0.0, 0.0))
                net_new = net + qty_signed
                cost_new = cost + qty_signed * float(px)
                agg[token_id] = (net_new, cost_new)
            # Write positions
            for token_id, (net, cost) in agg.items():
                avg_px = (cost / net) if abs(net) > 1e-9 else 0.0
                self.upsert_position(token_id, yes_qty=net, no_qty=0.0, delta_usd=0.0, avg_px=avg_px)

    def export_pnl_for_date(self, date_str: str) -> Tuple[float, float, float, float]:
        # Very simplified: realized = sum of SELL proceeds - BUY cost for that date; unrealized left 0
        with self._lock:
            cur = self._conn.execute(
                "SELECT side, px, qty, fee FROM fills WHERE date(datetime(ts, 'unixepoch')) = ?",
                (date_str,),
            )
            realized = 0.0
            fees = 0.0
            for side, px, qty, fee in cur.fetchall():
                amt = float(px) * float(qty)
                realized += amt if str(side).upper() == "SELL" else -amt
                fees += float(fee or 0.0)
            unrealized = 0.0
            rebates = 0.0
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO pnl_daily(date, realized, unrealized, fees, rebates) VALUES (?, ?, ?, ?, ?)",
                    (date_str, realized, unrealized, fees, rebates),
                )
            return realized, unrealized, fees, rebates

    # Selection snapshot
    def write_selection_snapshot(self, tokens: List[str]) -> Tuple[int, float]:
        import json
        ts = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO selection_snapshots(ts, tokens) VALUES(?, ?)",
                (ts, json.dumps(tokens)),
            )
            cur = self._conn.execute("SELECT last_insert_rowid()")
            version = int(cur.fetchone()[0])
        return version, ts

