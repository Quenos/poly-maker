import random
import time
from dataclasses import dataclass
from typing import List, Optional

from py_clob_client.client import ClobClient
from py_clob_client.exceptions import PolyApiException
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from py_clob_client.constants import POLYGON
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from mm.state import StateStore, OrderRecord


class NonRetryableOrderError(Exception):
    """Raised when an order error should not be retried (e.g., insufficient allowance/balance)."""


class RetryableOrderError(Exception):
    """Raised to trigger a retry via tenacity for transient errors."""


class OrdersClient:
    """Wrapper around py-clob-client with retries and persistence hooks."""

    def __init__(self, host: str, key: str, funder: str, state: StateStore) -> None:
        self.client = ClobClient(host=host, key=key, chain_id=POLYGON, funder=funder)
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds=creds)
        self.state = state

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(0.1, 1.0),
        retry=retry_if_exception_type(RetryableOrderError),
    )
    def place_order(self, token_id: str, side: str, price: float, size: float, neg_risk: bool = False) -> dict:
        try:
            args = OrderArgs(token_id=str(token_id), price=price, size=size, side=side)
            options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
            signed = self.client.create_order(args, options=options) if options else self.client.create_order(args)
            resp = self.client.post_order(signed)
            order_id = str(resp.get("order_id") or resp.get("id") or f"p_{int(time.time()*1000)}_{random.randint(1,9999)}")
            if getattr(self, "state", None) is not None:
                self.state.record_order(
                    OrderRecord(
                        order_id=order_id,
                        token_id=token_id,
                        side=side,
                        price=price,
                        size=size,
                        timestamp=time.time(),
                    )
                )
            return resp
        except PolyApiException as exc:
            # Detect insufficient allowance/balance and avoid retrying
            message = str(getattr(exc, "error_message", "") or str(exc)).lower()
            if "not enough balance" in message or "not enough allowance" in message or "balance / allowance" in message:
                raise NonRetryableOrderError(message)
            # Treat other PolyApiException as retryable
            raise RetryableOrderError(str(exc))
        except Exception as exc:
            # Unknown/transient errors: retry
            raise RetryableOrderError(str(exc))

    def cancel_market_orders(self, market: Optional[str] = None, asset_id: Optional[str] = None) -> None:
        if market:
            self.client.cancel_market_orders(market=market)
        elif asset_id:
            self.client.cancel_market_orders(asset_id=str(asset_id))

    def get_orders(self) -> List[dict]:
        return list(self.client.get_orders())


@dataclass
class DesiredQuote:
    token_id: str
    side: str  # BUY/SELL
    price: float
    size: float
    level: int  # 0..N-1


class OrdersEngine:
    """Diffing and lifecycle management for layered quotes.

    Tracks order age, partial fills, and replaces/cancels/places to match desired quotes.
    """

    def __init__(
        self,
        client: OrdersClient,
        tick: float,
        partial_fill_pct: float,
        order_max_age_sec: int,
        requote_mid_ticks: int,
        requote_queue_levels: int,
    ) -> None:
        self.client = client
        self.tick = tick
        self.partial_fill_pct = partial_fill_pct
        self.order_max_age_sec = order_max_age_sec
        self.requote_mid_ticks = requote_mid_ticks
        self.requote_queue_levels = requote_queue_levels
        # local state
        self._created_ts: dict[str, float] = {}
        self._price_level_idx: dict[str, int] = {}  # order_id -> level
        self._last_mid: dict[str, float] = {}  # token -> last seen mid used for placement

    def _now(self) -> float:
        return time.time()

    def _needs_replace_due_to_age(self, order_id: str) -> bool:
        ts = self._created_ts.get(order_id)
        if ts is None:
            return False
        return (self._now() - ts) >= self.order_max_age_sec

    def _mid_shift_ticks(self, token: str, current_mid: float) -> int:
        prev = self._last_mid.get(token)
        if prev is None or current_mid is None:
            return 0
        return int(abs(current_mid - prev) / max(self.tick, 1e-6))

    def _select_mapping(self, desired: list[DesiredQuote], live_orders: list[dict]) -> dict[str, DesiredQuote]:
        """Greedy map live orders to closest desired by price per token/side."""
        mapping: dict[str, DesiredQuote] = {}
        remaining = desired[:]
        for o in live_orders:
            try:
                token = str(o.get("asset_id") or o.get("token_id") or o.get("market") or "")
                side = str(o.get("side") or o.get("action") or o.get("order_side") or "").upper()
                price = float(o.get("price"))
            except Exception:
                continue
            # find best desired match
            best_idx = -1
            best_diff = 1e9
            for idx, dq in enumerate(remaining):
                if dq.token_id != token or dq.side.upper() != side:
                    continue
                diff = abs(dq.price - price)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_idx >= 0:
                mapping[str(o.get("id") or o.get("order_id"))] = remaining.pop(best_idx)
        return mapping

    def sync(
        self,
        desired: list[DesiredQuote],
        mid_by_token: dict[str, float],
    ) -> dict:
        actions = {"placed": [], "cancelled": [], "replaced": [], "errors": []}
        live = list(self.client.get_orders())
        id_to_order = {str(o.get("id") or o.get("order_id")): o for o in live}
        order_to_desired = self._select_mapping(desired, live)

        # Global requote trigger per token if mid shift large
        requote_tokens: set[str] = set()
        for token, mid in mid_by_token.items():
            if self._mid_shift_ticks(token, mid) >= self.requote_mid_ticks:
                requote_tokens.add(token)
                self._last_mid[token] = mid

        desired_keys = {(dq.token_id, dq.side, dq.level): dq for dq in desired}
        # Identify existing mapped orders and decide keep/replace
        used_keys: set[tuple[str, str, int]] = set()
        for oid, dq in order_to_desired.items():
            o = id_to_order.get(oid, {})
            token = dq.token_id
            mid = mid_by_token.get(token, 0.0)
            # Partial fill pct
            filled = float(o.get("size_matched") or 0.0)
            orig = float(o.get("original_size") or o.get("size") or 0.0)
            pct = (filled / orig * 100.0) if orig > 0 else 0.0
            queue_loss = 0  # placeholder proxy; can be enhanced with book depth
            must_replace = (self._needs_replace_due_to_age(oid) or (pct >= self.partial_fill_pct) or (queue_loss >= self.requote_queue_levels) or (token in requote_tokens))
            if must_replace or abs(float(o.get("price", 0.0)) - dq.price) >= self.tick:
                # Replace: cancel then place new
                try:
                    self.client.cancel_market_orders(asset_id=token)
                except Exception:
                    pass
                try:
                    self.client.place_order(token_id=token, side=dq.side, price=dq.price, size=dq.size)
                    self._created_ts[oid] = self._now()
                    self._price_level_idx[oid] = dq.level
                    actions["replaced"].append({"id": oid, "token": token, "side": dq.side, "price": dq.price, "size": dq.size})
                except NonRetryableOrderError as exc:
                    actions["errors"].append({"token": token, "side": dq.side, "price": dq.price, "size": dq.size, "type": "nonretryable", "error": str(exc)})
                    # continue to try other orders
                except RetryableOrderError as exc:
                    actions["errors"].append({
                        "token": token,
                        "side": dq.side,
                        "price": dq.price,
                        "size": dq.size,
                        "type": "retryable",
                        "error": str(exc),
                    })
                    # continue to try other orders
                except Exception as exc:
                    actions["errors"].append({"token": token, "side": dq.side, "price": dq.price, "size": dq.size, "type": "unknown", "error": str(exc)})
            else:
                used_keys.add((dq.token_id, dq.side, dq.level))

        # Place missing desired
        for key, dq in desired_keys.items():
            if key in used_keys:
                continue
            try:
                res = self.client.place_order(token_id=dq.token_id, side=dq.side, price=dq.price, size=dq.size)
                new_id = str(res.get("order_id") or res.get("id") or f"local_{int(self._now()*1000)}")
                self._created_ts[new_id] = self._now()
                self._price_level_idx[new_id] = dq.level
                actions["placed"].append({"id": new_id, "token": dq.token_id, "side": dq.side, "price": dq.price, "size": dq.size})
            except NonRetryableOrderError as exc:
                actions["errors"].append({"token": dq.token_id, "side": dq.side, "price": dq.price, "size": dq.size, "type": "nonretryable", "error": str(exc)})
                continue
            except RetryableOrderError as exc:
                actions["errors"].append({
                    "token": dq.token_id,
                    "side": dq.side,
                    "price": dq.price,
                    "size": dq.size,
                    "type": "retryable",
                    "error": str(exc),
                })
                continue
            except Exception as exc:
                actions["errors"].append({"token": dq.token_id, "side": dq.side, "price": dq.price, "size": dq.size, "type": "unknown", "error": str(exc)})

        # Cancel stray live orders not desired
        desired_prices_by_ts = {(dq.token_id, dq.side): [d.price for d in desired if d.token_id == dq.token_id and d.side == dq.side] for dq in desired}
        for oid, o in id_to_order.items():
            token = str(o.get("asset_id") or o.get("token_id") or "")
            side = str(o.get("side") or "").upper()
            price = float(o.get("price") or 0.0)
            want_prices = desired_prices_by_ts.get((token, side), [])
            # If price not within 1 tick of any desired, cancel
            if not any(abs(price - p) < self.tick for p in want_prices):
                try:
                    self.client.cancel_market_orders(asset_id=token)
                    actions["cancelled"].append({"id": oid, "token": token, "side": side})
                except Exception:
                    pass

        return actions
