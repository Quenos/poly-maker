import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

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
    def place_order(self, token_id: str, side: Union["Side", str], price: float, size: float, neg_risk: bool = False) -> dict:
        try:
            side_str = side.value if isinstance(side, Side) else str(side).upper()
            args = OrderArgs(token_id=str(token_id), price=price, size=size, side=side_str)
            options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
            signed = self.client.create_order(args, options=options) if options else self.client.create_order(args)
            resp = self.client.post_order(signed)
            order_id = str(resp.get("order_id") or resp.get("id") or f"p_{int(time.time()*1000)}_{random.randint(1,9999)}")
            if getattr(self, "state", None) is not None:
                self.state.record_order(
                    OrderRecord(
                        order_id=order_id,
                        token_id=token_id,
                        side=side_str,
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


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class DesiredQuote:
    token_id: str
    side: Side
    price: float
    size: float
    level: int  # 0..N-1


@dataclass
class OrderActionPlaced:
    id: str
    token: str
    side: Side
    price: float
    size: float


@dataclass
class OrderActionReplaced:
    id: str
    token: str
    side: Side
    price: float
    size: float


@dataclass
class OrderActionCancelled:
    id: str


@dataclass
class OrderActionError:
    token: str
    side: Side
    price: float
    size: float
    type: str
    error: str


@dataclass
class SyncActions:
    placed: List[OrderActionPlaced]
    cancelled: List[OrderActionCancelled]
    replaced: List[OrderActionReplaced]
    errors: List[OrderActionError]

    def __getitem__(self, key: str):
        k = str(key)
        if k == "placed":
            return self.placed
        if k == "cancelled":
            return self.cancelled
        if k == "replaced":
            return self.replaced
        if k == "errors":
            return self.errors
        raise KeyError(k)


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
                dq_side = dq.side.value if isinstance(dq.side, Side) else str(dq.side).upper()
                if dq.token_id != token or dq_side != side:
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
    ) -> SyncActions:
        actions = SyncActions(placed=[], cancelled=[], replaced=[], errors=[])
        live = list(self.client.get_orders())
        id_to_order = {str(o.get("id") or o.get("order_id")): o for o in live}
        order_to_desired = self._select_mapping(desired, live)

        # Organize desired and live by (token, side) and compute depth from mid
        def _depth(side: Union[Side, str], price: float, mid: float) -> float:
            side_u = side.value if isinstance(side, Side) else str(side).upper()
            if side_u == "BUY":
                return max(0.0, float(mid) - float(price))
            return max(0.0, float(price) - float(mid))

        desired_by_ts: dict[tuple[str, str], list[DesiredQuote]] = {}
        for dq in desired:
            side_key = dq.side.value if isinstance(dq.side, Side) else str(dq.side).upper()
            key = (dq.token_id, side_key)
            desired_by_ts.setdefault(key, []).append(dq)
        for key, lst in desired_by_ts.items():
            tok, side = key
            mid = float(mid_by_token.get(tok, 0.5))
            lst.sort(key=lambda d: _depth(side, d.price, mid))

        live_by_ts: dict[tuple[str, str], list[dict]] = {}
        for o in live:
            try:
                token = str(o.get("asset_id") or o.get("token_id") or o.get("market") or "")
                side = str(o.get("side") or o.get("action") or o.get("order_side") or "").upper()
                if not token or side not in ("BUY", "SELL"):
                    continue
                live_by_ts.setdefault((token, side), []).append(o)
            except Exception:
                continue
        for key, lst in live_by_ts.items():
            tok, side = key
            mid = float(mid_by_token.get(tok, 0.5))

            def sort_key(o: dict) -> float:
                try:
                    price = float(o.get("price"))
                except Exception:
                    price = 0.0
                return _depth(side, price, mid)
            lst.sort(key=sort_key)
        
        # Determine which existing live orders to keep: top-N (layers) by closeness to mid per side
        keep_ids: set[str] = set()
        depths_by_id: dict[str, float] = {}
        for key, live_list in live_by_ts.items():
            tok, side = key
            desired_list = desired_by_ts.get(key, [])
            max_levels = max(1, len(desired_list)) if desired_list else 0
            mid = float(mid_by_token.get(tok, 0.5))
            for idx, o in enumerate(live_list):
                oid = str(o.get("id") or o.get("order_id"))
                try:
                    price = float(o.get("price"))
                except Exception:
                    price = 0.0
                d = _depth(side, price, mid)
                depths_by_id[oid] = d
                if idx < max_levels:
                    keep_ids.add(oid)

        # Global requote trigger per token if mid shift large (still maintained for age/shift gating)
        requote_tokens: set[str] = set()
        for token, mid in mid_by_token.items():
            if self._mid_shift_ticks(token, mid) >= self.requote_mid_ticks:
                requote_tokens.add(token)
                self._last_mid[token] = mid

        # Build a set of desired keys for quick membership checks
        # (kept for potential future use; not required in capacity-based placement)
        # desired_keys = {(dq.token_id, dq.side, dq.level): dq for dq in desired}
        # Identify existing mapped orders and decide replace only if it improves closeness and capacity allows
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
            # Compute closeness improvement if replaced and detect price change
            try:
                current_price = float(o.get("price", 0.0))
            except Exception:
                current_price = 0.0
            current_depth = _depth(dq.side, current_price, float(mid))
            desired_depth = _depth(dq.side, dq.price, float(mid))
            price_changed = abs(current_price - dq.price) >= max(self.tick, 1e-6)
            # Capacity for this side
            side_key2 = dq.side.value if isinstance(dq.side, Side) else str(dq.side).upper()
            max_levels = max(1, len(desired_by_ts.get((token, side_key2), []))) if desired_by_ts.get((token, side_key2), []) else 0
            live_for_side = live_by_ts.get((token, side_key2), [])
            capacity_left = max(0, max_levels - len(live_for_side))

            if (price_changed or must_replace):
                # Replace only if it moves closer to mid AND capacity exists; avoid mass cancel
                try:
                    self.client.place_order(token_id=token, side=dq.side, price=dq.price, size=dq.size)
                    self._created_ts[oid] = self._now()
                    self._price_level_idx[oid] = dq.level
                    actions.replaced.append(OrderActionReplaced(id=oid, token=token, side=dq.side, price=dq.price, size=dq.size))
                except NonRetryableOrderError as exc:
                    actions.errors.append(OrderActionError(token=token, side=dq.side, price=dq.price, size=dq.size, type="nonretryable", error=str(exc)))
                except RetryableOrderError as exc:
                    actions.errors.append(OrderActionError(token=token, side=dq.side, price=dq.price, size=dq.size, type="retryable", error=str(exc)))
                except Exception as exc:
                    actions.errors.append(OrderActionError(token=token, side=dq.side, price=dq.price, size=dq.size, type="unknown", error=str(exc)))
            else:
                # Keep existing; mark desired level as satisfied
                used_keys.add((dq.token_id, side_key2, dq.level))

        # Place missing desired, respecting capacity and preferring closest levels
        for (token, side), desired_list in desired_by_ts.items():
            # Determine capacity left
            live_for_side = live_by_ts.get((token, side), [])
            max_levels = max(1, len(desired_list)) if desired_list else 0
            capacity_left = max(0, max_levels - len(live_for_side))
            if capacity_left <= 0:
                continue
            # Prices covered by existing top-N live within 1 tick
            mid = float(mid_by_token.get(token, 0.5))
            covered = []
            for o in live_for_side[:max_levels]:
                try:
                    covered.append(float(o.get("price")))
                except Exception:
                    continue
            
            def is_covered(p: float) -> bool:
                return any(abs(p - cp) < self.tick for cp in covered)
            for dq in desired_list:
                if capacity_left <= 0:
                    break
                if is_covered(dq.price):
                    continue
                try:
                    res = self.client.place_order(token_id=dq.token_id, side=dq.side, price=dq.price, size=dq.size)
                    new_id = str(res.get("order_id") or res.get("id") or f"local_{int(self._now()*1000)}")
                    self._created_ts[new_id] = self._now()
                    self._price_level_idx[new_id] = dq.level
                    actions.placed.append(OrderActionPlaced(id=new_id, token=dq.token_id, side=dq.side, price=dq.price, size=dq.size))
                    capacity_left -= 1
                except NonRetryableOrderError as exc:
                    actions.errors.append(OrderActionError(token=dq.token_id, side=dq.side, price=dq.price, size=dq.size, type="nonretryable", error=str(exc)))
                except RetryableOrderError as exc:
                    actions.errors.append(OrderActionError(token=dq.token_id, side=dq.side, price=dq.price, size=dq.size, type="retryable", error=str(exc)))
                except Exception as exc:
                    actions.errors.append(OrderActionError(token=dq.token_id, side=dq.side, price=dq.price, size=dq.size, type="unknown", error=str(exc)))

        # Do NOT cancel existing closer orders when deeper desired appear; skip mass cancellations

        return actions
