import asyncio
import collections
import json
import logging
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import requests
import websockets
from prometheus_client import Counter, Gauge


logger = logging.getLogger(__name__)


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class Trade:
    ts: float
    price: float
    size: float
    side: str  # "BUY" or "SELL"


class OrderBook:
    def __init__(self) -> None:
        self.bids: List[BookLevel] = []
        self.asks: List[BookLevel] = []
        self.seq: Optional[int] = None
        self.last_update_ts: float = 0.0

    def set_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], seq: Optional[int]) -> None:
        self.bids = [BookLevel(price=float(p), size=float(s)) for p, s in bids]
        self.asks = [BookLevel(price=float(p), size=float(s)) for p, s in asks]
        self.seq = seq
        self.last_update_ts = time.time()

    def apply_levels(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], seq: Optional[int]) -> None:
        # For simplicity treat diffs as replace-full-depth provided, which is common for Polymarket WS
        self.set_snapshot(bids, asks, seq)

    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    def mid(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    def microprice(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        b0, a0 = self.bids[0], self.asks[0]
        denom = (b0.size + a0.size)
        if denom <= 0:
            return self.mid()
        return (a0.size * b0.price + b0.size * a0.price) / denom


class MarketData:
    ws_connected_gauge = Gauge("mm_ws_connected", "WS connection status (1=connected)")  # type: ignore
    ws_reconnects = Counter("mm_ws_reconnects_total", "WS reconnects total")  # type: ignore
    seq_gaps = Counter("mm_seq_gaps_total", "Sequence gaps detected", ["token_id"])  # type: ignore
    snapshot_reloads = Counter("mm_snapshot_reload_total", "Snapshot reloads", ["token_id"])  # type: ignore
    trades_minute_gauge = Gauge("mm_trades_per_min", "Trades per minute", ["token_id"])  # type: ignore

    def __init__(self, clob_ws_url: str, clob_rest_base: str) -> None:
        self.ws_url = clob_ws_url.rstrip("/") + "/market"
        self.rest_prices = clob_rest_base.rstrip("/") + "/prices"
        self.books: Dict[str, OrderBook] = {}
        self.trades: Dict[str, Deque[Trade]] = {}
        self.expected_seq: Dict[str, Optional[int]] = {}
        self.buffered_diffs: Dict[str, Dict[int, Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]] = {}
        self.desired_tokens: List[str] = []
        self._stop = False
        self._first_snapshot_logged: set[str] = set()

    def subscribe(self, token_ids: List[str]) -> None:
        self.desired_tokens = [str(t) for t in token_ids]
        for t in self.desired_tokens:
            if t not in self.books:
                self.books[t] = OrderBook()
                self.trades[t] = collections.deque(maxlen=1000)
                self.expected_seq[t] = None
                self.buffered_diffs[t] = {}
        logger.info("Subscribing to %d tokens", len(self.desired_tokens))

    def get_signals(self, token_id: str) -> Dict[str, Optional[float]]:
        ob = self.books.get(token_id)
        if not ob:
            return {"best_bid": None, "best_ask": None, "mid": None, "microprice": None, "book_seq": None}
        return {
            "best_bid": ob.best_bid(),
            "best_ask": ob.best_ask(),
            "mid": ob.mid(),
            "microprice": ob.microprice(),
            "book_seq": float(ob.seq) if ob.seq is not None else None,
        }

    def get_trade_stats(self, token_id: str) -> Dict[str, Optional[float]]:
        buf = self.trades.get(token_id)
        if not buf:
            return {"last_price": None, "trades_per_min": 0.0, "imbalance": 0.0}
        now = time.time()
        last_price = buf[-1].price if buf else None
        recent = [t for t in buf if now - t.ts <= 60.0]
        buys = sum(t.size for t in recent if t.side.upper() == "BUY")
        sells = sum(t.size for t in recent if t.side.upper() == "SELL")
        total = buys + sells
        imbalance = (buys - sells) / total if total > 0 else 0.0
        self.trades_minute_gauge.labels(token_id=token_id).set(float(len(recent)))
        return {"last_price": last_price, "trades_per_min": float(len(recent)), "imbalance": float(imbalance)}

    async def run_ws(self, token_ids: List[str]) -> None:
        self.subscribe(token_ids)
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(self.ws_url, ping_interval=5, ping_timeout=None) as ws:
                    self.ws_connected_gauge.set(1)
                    self.ws_reconnects.inc()
                    await ws.send(json.dumps({"assets_ids": self.desired_tokens}))
                    last = time.time()
                    while not self._stop:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            self._handle_ws_message(data)
                            last = time.time()
                        except asyncio.TimeoutError:
                            # heartbeat
                            _ = last
                    self.ws_connected_gauge.set(0)
            except Exception:
                logger.exception("WS error, reconnecting")
                self.ws_connected_gauge.set(0)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _handle_ws_message(self, data) -> None:
        try:
            # Case 1: dict with 'assets' (snapshot/diff style)
            if isinstance(data, dict) and "assets" in data:
                for asset in data.get("assets", []):
                    token = str(asset.get("id"))
                    seq = asset.get("seq")
                    bids = [(float(x[0]), float(x[1])) for x in asset.get("bids", [])]
                    asks = [(float(x[0]), float(x[1])) for x in asset.get("asks", [])]
                    is_snapshot = bool(asset.get("snapshot", False))
                    self._apply_update(token, bids, asks, seq, is_snapshot)
                for tr in data.get("trades", []):
                    token = str(tr.get("id") or tr.get("token_id") or tr.get("asset_id"))
                    side = str(tr.get("side", "")).upper() or ("BUY" if float(tr.get("taker_side", 1)) > 0 else "SELL")
                    price = float(tr.get("price"))
                    size = float(tr.get("size", tr.get("amount", 0)))
                    self._record_trade(token, price, size, side)
                return

            # Case 2: list of events with 'event_type' (book/price_change)
            if isinstance(data, list):
                for ev in data:
                    et = ev.get("event_type") if isinstance(ev, dict) else None
                    if et == "book":
                        token = str(ev.get("market"))
                        bids = [(float(e.get("price")), float(e.get("size"))) for e in ev.get("bids", [])]
                        asks = [(float(e.get("price")), float(e.get("size"))) for e in ev.get("asks", [])]
                        self._apply_update(token, bids, asks, seq=None, is_snapshot=True)
                    elif et == "price_change":
                        # Optional: apply granular changes; for now, ignore or trigger future refresh
                        continue
                return

        except Exception:
            logger.exception("Error handling WS message")

    def _record_trade(self, token: str, price: float, size: float, side: str) -> None:
        if token not in self.trades:
            self.trades[token] = collections.deque(maxlen=1000)
        self.trades[token].append(Trade(ts=time.time(), price=price, size=size, side=side))

    def _apply_update(self, token: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], seq: Optional[int], is_snapshot: bool) -> None:
        if token not in self.books:
            self.books[token] = OrderBook()
            self.expected_seq[token] = None
            self.buffered_diffs[token] = {}

        ob = self.books[token]
        exp = self.expected_seq.get(token)

        if is_snapshot or exp is None:
            ob.set_snapshot(bids, asks, seq)
            self.expected_seq[token] = seq
            self.snapshot_reloads.labels(token_id=token).inc()
            if token not in self._first_snapshot_logged:
                logger.info("Snapshot applied for token %s: bids=%d asks=%d", token, len(bids), len(asks))
                self._first_snapshot_logged.add(token)
            # replay any buffered diffs in order
            buf = self.buffered_diffs.get(token, {})
            for next_seq in sorted(buf.keys()):
                if self.expected_seq[token] is not None and next_seq == self.expected_seq[token] + 1:
                    b2, a2 = buf[next_seq]
                    ob.apply_levels(b2, a2, next_seq)
                    self.expected_seq[token] = next_seq
            return

        if seq is None:
            # treat as replace without seq
            ob.apply_levels(bids, asks, seq)
            return

        # gap detection
        if exp is not None and seq != exp + 1:
            self.seq_gaps.labels(token_id=token).inc()
            # buffer and reload snapshot
            self.buffered_diffs[token][seq] = (bids, asks)
            self._reload_snapshot_rest(token)
            return

        # normal in-order update
        ob.apply_levels(bids, asks, seq)
        self.expected_seq[token] = seq

    def _reload_snapshot_rest(self, token: str) -> None:
        try:
            payload = {"requests": [{"token_id": str(token)}]}
            resp = requests.post(self.rest_prices, json=payload, timeout=10)
            resp.raise_for_status()
            out = resp.json() or {}
            arr = out.get("responses", [])
            if not arr:
                return
            entry = arr[0]
            bids = [(float(x["price"]), float(x["size"])) for x in entry.get("bids", [])]
            asks = [(float(x["price"]), float(x["size"])) for x in entry.get("asks", [])]
            # No seq in REST snapshot; reset sequence expectation
            self.books[token].set_snapshot(bids, asks, None)
            self.expected_seq[token] = None
            self.snapshot_reloads.labels(token_id=token).inc()
        except Exception:
            logger.exception("Snapshot reload failed for token %s", token)

    def backfill_prices(self, token_ids: List[str]) -> None:
        """Fetch initial top-of-book snapshots for multiple tokens via REST.

        Populates self.books with OrderBook snapshots and resets expected sequences to None.
        """
        if not token_ids:
            return
        # Try assets_ids schema first (most common); fall back to requests schema
        tokens = [str(t) for t in token_ids]
        logger.info("Starting REST backfill for %d tokens", len(tokens))
        def _apply_entries(entries):
            applied = 0
            for entry in entries:
                token = str(entry.get("token_id") or entry.get("asset_id") or entry.get("id") or "")
                if not token:
                    continue
                bids_src = entry.get("bids", [])
                asks_src = entry.get("asks", [])
                # bids/asks may be array of objects {price,size}
                try:
                    bids = [(float(x["price"]), float(x["size"])) for x in bids_src]
                    asks = [(float(x["price"]), float(x["size"])) for x in asks_src]
                except Exception:
                    # or array of [price, size]
                    bids = [(float(x[0]), float(x[1])) for x in bids_src]
                    asks = [(float(x[0]), float(x[1])) for x in asks_src]
                if token not in self.books:
                    self.books[token] = OrderBook()
                self.books[token].set_snapshot(bids, asks, None)
                self.expected_seq[token] = None
                applied += 1
                # Per-token readiness log
                ob = self.books[token]
                bb = ob.best_bid()
                ba = ob.best_ask()
                if bb is not None and ba is not None:
                    logger.info("Backfill ready for token %s: best_bid=%.4f best_ask=%.4f", token, bb, ba)
                else:
                    logger.warning("Backfill incomplete for token %s: bids=%d asks=%d", token, len(bids), len(asks))
            if applied:
                logger.info("REST snapshot applied for %d tokens", applied)
            return applied
        try:
            # First attempt: assets_ids list
            logger.info("REST backfill attempt: assets_ids (%d tokens)", len(tokens))
            resp = requests.post(self.rest_prices, json={"assets_ids": tokens}, timeout=10)
            if 200 <= resp.status_code < 300:
                body = resp.json()
                if isinstance(body, list):
                    _apply_entries(body)
                    return
                if isinstance(body, dict) and "responses" in body:
                    _apply_entries(body.get("responses", []))
                    return
            # Fallback attempt: requests: [{token_id: ...}]
            logger.info("REST backfill retry: requests schema (%d tokens)", len(tokens))
            resp2 = requests.post(self.rest_prices, json={"requests": [{"token_id": t} for t in tokens]}, timeout=10)
            if 200 <= resp2.status_code < 300:
                body2 = resp2.json()
                if isinstance(body2, list):
                    _apply_entries(body2)
                    return
                if isinstance(body2, dict) and "responses" in body2:
                    _apply_entries(body2.get("responses", []))
                    return
        except Exception:
            logger.exception("Failed REST backfill for %d tokens", len(token_ids))

