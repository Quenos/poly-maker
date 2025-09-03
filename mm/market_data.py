import asyncio
import collections
import json
import logging
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Iterable

import requests
import websockets
from prometheus_client import Counter, Gauge


logger = logging.getLogger(__name__)


def to_decimal_token_id(token_id: str) -> str:
    """
    Return canonical DECIMAL string token_id (ERC-1155).
    Accepts either decimal (returns as-is) or 0x-hex (32-byte) and converts to decimal.
    """
    t = str(token_id).strip()
    if t.startswith("0x") or t.startswith("0X"):
        return str(int(t, 16))
    # assume decimal
    int(t, 10)
    return t

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


# Token1-based fair price metrics
mid_token1_gauge = Gauge("mm_mid_token1", "Token1 midpoint price")  # type: ignore
micro_token1_gauge = Gauge("mm_micro_token1", "Token1 microprice")  # type: ignore
spread_ticks_gauge = Gauge("mm_spread_ticks", "Top-of-book spread measured in ticks")  # type: ignore
mid_recalc_total = Counter("mm_mid_recalc_total", "Count of token1 mid recalculations")  # type: ignore


def fair_prices(book_token1: OrderBook, tick: float = 0.01) -> dict:
    """Compute token1/token2 fair prices from token1 order book only.

    Returns a dict with mid/micro for token1 and its complement token2.
    """
    try:
        b0 = book_token1.bids[0] if book_token1.bids else None
        a0 = book_token1.asks[0] if book_token1.asks else None
        if b0 is None and a0 is None:
            return {"mid_t1": None, "micro_t1": None, "mid_t2": None, "micro_t2": None}

        # Base levels
        best_bid = float(b0.price) if b0 is not None else None
        best_ask = float(a0.price) if a0 is not None else None
        bid_sz = float(b0.size) if b0 is not None else 0.0
        ask_sz = float(a0.size) if a0 is not None else 0.0

        # Compute mid_t1 (handle edge cases)
        if best_bid is not None and best_ask is not None:
            mid_t1 = (best_bid + best_ask) / 2.0
            # Crossed book clamp
            if best_bid > best_ask:
                lower, upper = best_ask, best_bid
                mid_t1 = min(max(mid_t1, lower), upper)
        elif best_bid is not None:
            mid_t1 = best_bid
        else:
            mid_t1 = best_ask  # type: ignore[assignment]

        # Compute micro_t1
        denom = bid_sz + ask_sz
        if (best_bid is None) and (best_ask is None):
            micro_t1 = None
        elif denom > 0:
            bb = best_bid if best_bid is not None else 0.0
            ba = best_ask if best_ask is not None else 0.0
            micro_t1 = (ba * bid_sz + bb * ask_sz) / denom
        else:
            micro_t1 = mid_t1

        # Clip outputs into [0.01, 0.99]
        def _clip(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            return float(min(0.99, max(0.01, x)))

        mid_t1 = _clip(mid_t1)
        micro_t1 = _clip(micro_t1)

        # Mirror for token2
        mid_t2 = (1.0 - mid_t1) if mid_t1 is not None else None  # type: ignore[operator]
        micro_t2 = (1.0 - micro_t1) if micro_t1 is not None else None  # type: ignore[operator]

        # Metrics
        if best_bid is not None and best_ask is not None:
            spread_ticks_gauge.set(float((best_ask - best_bid) / max(tick, 1e-6)))
        if mid_t1 is not None:
            mid_token1_gauge.set(float(mid_t1))
        if micro_t1 is not None:
            micro_token1_gauge.set(float(micro_t1))
        mid_recalc_total.inc()

        # Parity check alert
        if mid_t1 is not None and mid_t2 is not None:
            if abs((mid_t1 + mid_t2) - 1.0) > 1e-4:
                logger.error("Parity check failed: mid_t1=%.6f mid_t2=%.6f", mid_t1, mid_t2)

        return {"mid_t1": mid_t1, "micro_t1": micro_t1, "mid_t2": mid_t2, "micro_t2": micro_t2}
    except Exception:
        logger.exception("fair_prices computation error")
        return {"mid_t1": None, "micro_t1": None, "mid_t2": None, "micro_t2": None}

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

    def _http_post_prices(
        self,
        token_ids: Iterable[str],
        timeout: float = 5.0,
        retries: int = 3,
        backoff: float = 0.5,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Calls REST /prices with a batch of DECIMAL token_ids.
        Request: [ {"token_id":"<decimal>", "side":"BUY"}, ... ]
        Returns: { "<decimal>": {"BUY": float|None, "SELL": float|None} }
        """
        ids = [to_decimal_token_id(t) for t in token_ids]
        if not ids:
            return {}
        payload: List[dict] = []
        for tid in ids:
            payload.append({"token_id": tid, "side": "BUY"})
            payload.append({"token_id": tid, "side": "SELL"})
        out: Dict[str, Dict[str, Optional[float]]] = {}
        err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                r = requests.post(self.rest_prices, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                # Normalize to dict with uppercase BUY/SELL keys, keyed by DECIMAL token_id
                if isinstance(data, dict):
                    for tid, d in data.items():
                        tdec = to_decimal_token_id(tid)
                        buy = d.get("BUY", d.get("buy"))
                        sell = d.get("SELL", d.get("sell"))
                        out[tdec] = {
                            "BUY": float(buy) if buy is not None else None,
                            "SELL": float(sell) if sell is not None else None,
                        }
                elif isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        tdec = to_decimal_token_id(item.get("token_id"))
                        side = str(item.get("side") or "").upper()
                        price = item.get("price")
                        if tdec not in out:
                            out[tdec] = {"BUY": None, "SELL": None}
                        if side in ("BUY", "SELL"):
                            out[tdec][side] = float(price) if price is not None else None
                return out
            except Exception as e:
                err = e
                time.sleep(backoff * (2 ** attempt))
        if err:
            logger.warning("prices REST failed: %s", err)
        return out

    def backfill_top_of_book(self, token_ids: Iterable[str]) -> None:
        """
        Build/refresh minimal snapshots for the given tokens using REST /prices.
        Creates empty OrderBook objects if missing; sets best bid/ask levels if present.
        """
        ids = [to_decimal_token_id(str(t)) for t in token_ids]
        if not ids:
            return
        quotes = self._http_post_prices(ids)

        for tid in ids:
            d = quotes.get(tid, {}) if quotes else {}
            best_bid = d.get("BUY")
            best_ask = d.get("SELL")
            if tid not in self.books:
                self.books[tid] = OrderBook()
            bids: List[Tuple[float, float]] = []
            asks: List[Tuple[float, float]] = []
            if best_bid is not None:
                bids.append((float(best_bid), 1.0))
            if best_ask is not None:
                asks.append((float(best_ask), 1.0))
            ob = self.books[tid]
            next_seq = (ob.seq or 0) + 1
            ob.set_snapshot(bids=bids, asks=asks, seq=next_seq)
            logger.info(
                "REST snapshot backfill: token=%s bid=%s ask=%s bids=%d asks=%d",
                tid,
                f"{best_bid:.4f}" if best_bid is not None else "None",
                f"{best_ask:.4f}" if best_ask is not None else "None",
                len(bids),
                len(asks),
            )

    def subscribe(self, token_ids: List[str]) -> None:
        self.desired_tokens = [to_decimal_token_id(str(t)) for t in token_ids]
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
                    # Immediately hydrate via REST top-of-book to avoid cold start
                    try:
                        self.backfill_top_of_book(self.desired_tokens)
                    except Exception as e:
                        logger.warning("backfill_top_of_book failed on connect: %s", e)
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
                    token = to_decimal_token_id(str(asset.get("id")))
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
                        token = to_decimal_token_id(str(ev.get("market")))
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
            # Quick hydrate top-of-book while full snapshot reloads
            try:
                self.backfill_top_of_book([token])
            except Exception:
                pass
            self._reload_snapshot_rest(token)
            return

        # normal in-order update
        ob.apply_levels(bids, asks, seq)
        self.expected_seq[token] = seq

    def _reload_snapshot_rest(self, token: str) -> None:
        try:
            token_dec = to_decimal_token_id(str(token))
            payload = {"requests": [{"token_id": token_dec}]}
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
