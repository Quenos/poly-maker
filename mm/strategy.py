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

    def compute_quote(self, book: OrderBook, inventory_norm: float, fair_hint: Optional[float] = None) -> Optional[Quote]:
        # Use provided fair_hint (token1-based or mirrored) when available; else fall back to book microprice
        if fair_hint is not None:
            fair = self.fair_ewma.update(fair_hint)
            mid_for_vol = fair
        else:
            micro = book.microprice()
            if micro is None:
                return None
            fair = self.fair_ewma.update(micro)
            mid_for_vol = fair
        sigma = self._update_vol(mid_for_vol)
        # Note: tick should be threaded from config; for now fee is expressed in ticks at 0.01
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
    rng: Optional[random.Random] = None,
) -> LayeredQuotes:
    def _round_to_tick(x: float, t: float) -> float:
        return max(t, min(1.0 - t, round(x / t) * t))
    bid_prices: List[float] = []
    ask_prices: List[float] = []
    sizes: List[float] = []
    for i in range(max(1, layers)):
        size = min(max_size, base_size * (1.5 ** i))
        r = rng.random() if rng is not None else random.random()
        jitter = (r - 0.5) * 2.0 * jitter_ticks * tick
        bid = _round_to_tick(base_quote.bid - tick * step_ticks * i + jitter, tick)
        r2 = rng.random() if rng is not None else random.random()
        jitter2 = (r2 - 0.5) * 2.0 * jitter_ticks * tick
        ask = _round_to_tick(base_quote.ask + tick * step_ticks * i + jitter2, tick)
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


@dataclass
class StrategyState:
    fair: float
    sigma: float
    inventory_usd: float
    bankroll_usd: float
    time_to_resolution_sec: float
    tick: float = 0.01


@dataclass
class QuotesOut:
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    size_multiplier: float


class AdvancedAvellanedaStrategy:
    def __init__(self, k_vol: float, k_fee_ticks: float, inv_gamma: float, min_spread_ticks: int = 1) -> None:
        self.k_vol = k_vol
        self.k_fee_ticks = k_fee_ticks
        self.base_gamma = inv_gamma
        self.min_spread_ticks = min_spread_ticks

    @staticmethod
    def _event_scalers(time_to_resolution_sec: float) -> tuple[float, float, float]:
        # Returns (h_multiplier, gamma_multiplier, size_multiplier)
        hrs = max(0.0, time_to_resolution_sec) / 3600.0
        if hrs <= 1:
            return (2.0, 2.0, 0.25)
        if hrs <= 4:
            return (1.7, 1.7, 0.4)
        if hrs <= 12:
            return (1.5, 1.5, 0.6)
        if hrs <= 48:
            return (1.2, 1.2, 0.8)
        return (1.0, 1.0, 1.0)

    @staticmethod
    def _apply_caps(gamma: float, h: float, q_norm: float, soft_cap: float, hard_cap: float) -> tuple[float, float, float]:
        size_mult = 1.0
        if abs(q_norm) >= soft_cap:
            gamma *= 1.2
            h *= 1.2
            size_mult *= 0.7
        if abs(q_norm) >= hard_cap:
            gamma *= 1.5
            h *= 1.5
            size_mult *= 0.1
        return gamma, h, size_mult

    def compute_quotes(
        self,
        state: StrategyState,
        soft_cap: float,
        hard_cap: float,
    ) -> QuotesOut:
        q_norm = 0.0
        if state.bankroll_usd > 0:
            q_norm = state.inventory_usd / state.bankroll_usd

        # Base spread and gamma
        h = self.k_vol * state.sigma + self.k_fee_ticks * state.tick
        # Enforce minimum spread in ticks
        min_h = self.min_spread_ticks * state.tick
        if h < min_h:
            h = min_h
        gamma = self.base_gamma

        # Event-time scalers
        h_mult, g_mult, size_mult_ev = self._event_scalers(state.time_to_resolution_sec)
        h *= h_mult
        gamma *= g_mult

        # Inventory caps
        gamma, h, size_mult_caps = self._apply_caps(gamma, h, q_norm, soft_cap, hard_cap)
        size_mult = size_mult_ev * size_mult_caps

        # Reservation price shift
        delta_r = -gamma * q_norm

        yes_bid = max(0.01, min(0.99, state.fair + delta_r - h))
        yes_ask = max(0.01, min(0.99, state.fair + delta_r + h))

        # Jitter one tick both sides (symmetric; caller can add time jitter externally)
        j = (random.random() - 0.5) * 2.0 * state.tick
        yes_bid = yes_bid + j
        yes_ask = yes_ask - j
        # Round to tick after jitter
        yes_bid = max(0.01, min(0.99, round(yes_bid / state.tick) * state.tick))
        yes_ask = max(0.01, min(0.99, round(yes_ask / state.tick) * state.tick))

        # Parity for NO, avoid self-cross after fees
        no_bid = max(0.01, min(0.99, 1.0 - yes_ask))
        no_ask = max(0.01, min(0.99, 1.0 - yes_bid))
        if no_bid >= no_ask:
            mid_no = (no_bid + no_ask) / 2.0
            no_bid = max(0.01, min(0.99, round((mid_no - state.tick) / state.tick) * state.tick))
            no_ask = max(0.01, min(0.99, round((mid_no + state.tick) / state.tick) * state.tick))

        return QuotesOut(yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask, size_multiplier=size_mult)