import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from mm.market_data import OrderBook


@dataclass
class Quote:
    bid: float
    ask: float


class EWMA:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class AvellanedaLite:
    def __init__(self, alpha_fair: float, k_vol: float, k_fee_ticks: float, inv_gamma: float) -> None:
        self.fair_ewma = EWMA(alpha=alpha_fair)
        self.k_vol = k_vol
        self.k_fee_ticks = k_fee_ticks
        self.inv_gamma = inv_gamma
        self.prev_mid: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.vol_ewma: Optional[float] = None

    def _update_vol(self, mid: float) -> float:
        if self.prev_price is None:
            self.prev_price = mid
            self.vol_ewma = 0.0
            return 0.0
        ret = math.log(max(1e-6, mid)) - math.log(max(1e-6, self.prev_price))
        self.prev_price = mid
        inst = abs(ret)
        if self.vol_ewma is None:
            self.vol_ewma = inst
        else:
            self.vol_ewma = 0.2 * inst + 0.8 * self.vol_ewma
        return float(self.vol_ewma)

    def compute_quote(self, book: OrderBook, inventory_norm: float) -> Optional[Quote]:
        micro = book.microprice()
        if micro is None:
            return None
        fair = self.fair_ewma.update(micro)
        mid = book.mid() or fair
        sigma = self._update_vol(mid)
        h = self.k_vol * sigma + self.k_fee_ticks * 0.01
        delta_r = -self.inv_gamma * inventory_norm
        bid = max(0.01, min(0.99, fair + delta_r - h))
        ask = max(0.01, min(0.99, fair + delta_r + h))
        return Quote(bid=bid, ask=ask)

    @staticmethod
    def mirror_no_side(q: Quote) -> Quote:
        return Quote(bid=1.0 - q.ask, ask=1.0 - q.bid)


@dataclass
class LayeredQuotes:
    bid_prices: List[float]
    ask_prices: List[float]
    sizes: List[float]
    timestamp: float


def build_layered_quotes(
    base_quote: Quote,
    layers: int,
    base_size: float,
    max_size: float,
    tick: float = 0.01,
    step_ticks: int = 1,
    jitter_ticks: float = 0.25,
) -> LayeredQuotes:
    bid_prices: List[float] = []
    ask_prices: List[float] = []
    sizes: List[float] = []
    for i in range(max(1, layers)):
        size = min(max_size, base_size * (1.5 ** i))
        jitter = (random.random() - 0.5) * 2.0 * jitter_ticks * tick
        bid = max(0.01, min(0.99, base_quote.bid - tick * step_ticks * i + jitter))
        jitter2 = (random.random() - 0.5) * 2.0 * jitter_ticks * tick
        ask = max(0.01, min(0.99, base_quote.ask + tick * step_ticks * i + jitter2))
        bid_prices.append(bid)
        ask_prices.append(ask)
        sizes.append(size)
    return LayeredQuotes(bid_prices=bid_prices, ask_prices=ask_prices, sizes=sizes, timestamp=time.time())


def apply_inventory_risk(
    quotes: LayeredQuotes,
    inventory_norm: float,
    soft_cap: float,
    hard_cap: float,
    widen_ticks: int = 1,
    tick: float = 0.01,
) -> LayeredQuotes:
    bp = quotes.bid_prices[:]
    ap = quotes.ask_prices[:]
    sizes = quotes.sizes[:]
    if abs(inventory_norm) >= soft_cap:
        # widen both sides
        bp = [max(0.01, b - widen_ticks * tick) for b in bp]
        ap = [min(0.99, a + widen_ticks * tick) for a in ap]
    if abs(inventory_norm) >= hard_cap:
        # stop quoting the inventory-heavy side
        if inventory_norm > 0:
            # long YES: stop bids
            sizes = [0.0] * len(sizes)
        else:
            # short YES: stop asks
            sizes = [0.0] * len(sizes)
    return LayeredQuotes(bid_prices=bp, ask_prices=ap, sizes=sizes, timestamp=quotes.timestamp)


def should_requote(
    last_mid: Optional[float],
    current_mid: Optional[float],
    last_timestamp: Optional[float],
    order_max_age_sec: int,
    requote_mid_ticks: int,
    tick: float = 0.01,
) -> bool:
    now = time.time()
    if last_timestamp is None or (now - last_timestamp) >= order_max_age_sec:
        return True
    if last_mid is None or current_mid is None:
        return True
    shift_ticks = abs(current_mid - last_mid) / tick
    if shift_ticks >= requote_mid_ticks:
        return True
    return False

