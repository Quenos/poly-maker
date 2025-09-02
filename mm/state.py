import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT,
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
                CREATE TABLE IF NOT EXISTS positions (
                    token_id TEXT PRIMARY KEY,
                    shares REAL,
                    avg_price REAL,
                    pnl REAL,
                    ts REAL
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

    def record_fill(self, fill_id: str, order_id: str, token_id: str, side: str, price: float, size: float, ts: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO fills(fill_id, order_id, token_id, side, price, size, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fill_id, order_id, token_id, side, price, size, ts),
            )
        self.fills_total.labels(side=side).inc()

    def upsert_position(self, token_id: str, shares: float, avg_price: float, pnl: float, ts: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO positions(token_id, shares, avg_price, pnl, ts) VALUES (?, ?, ?, ?, ?)",
                (token_id, shares, avg_price, pnl, ts),
            )
        self.inventory_gauge.labels(token_id=token_id).set(shares)

    def get_positions(self) -> Dict[str, Tuple[float, float]]:
        with self._lock:
            cur = self._conn.execute("SELECT token_id, shares, avg_price FROM positions")
            return {row[0]: (float(row[1]), float(row[2])) for row in cur.fetchall()}

