import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from mm.market_data import OrderBook
from prometheus_client import Counter  # type: ignore


logger = logging.getLogger("mm.strategy")

# Telemetry counters
sigma_clip_hits_total = Counter("mm_sigma_clip_hits_total", "Sigma clip hits total", ["token_id"])  # type: ignore
delta_clip_hits_total = Counter("mm_delta_clip_hits_total", "Delta clip hits total", ["token_id"])  # type: ignore
empty_book_nudges_total = Counter("mm_empty_book_nudges_total", "Empty book nudge events total", ["token_id"])  # type: ignore

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
    def __init__(
        self,
        alpha_fair: float,
        k_vol: float,
        k_fee_ticks: float,
        inv_gamma: float,
        *,
        w_mid: float = 0.25,
        w_micro: float = 0.35,
        w_hint: float = 0.40,
        vol_lambda: float = 0.94,
        max_sigma: float = 0.25,
        k_base: float = 1.0,
        k_depth_coeff: float = 0.02,
        tick: float = 0.01,
        k_min: float = 1e-3,
    ) -> None:
        self.fair_ewma = EWMA(alpha=alpha_fair)
        self.k_vol = k_vol
        self.k_fee_ticks = k_fee_ticks
        self.inv_gamma = inv_gamma
        # Fair blend weights
        self.w_mid = w_mid
        self.w_micro = w_micro
        self.w_hint = w_hint
        # Volatility (EWMA of squared returns)
        self.vol_lambda = vol_lambda
        self._last_px: Optional[float] = None
        self._sigma2: float = 0.0
        self._sigma: float = 0.0
        self.max_sigma = max_sigma
        # Liquidity scaling
        self.k_base = k_base
        self.k_depth_coeff = k_depth_coeff
        self.k_min = k_min
        # Tick
        self.tick = tick

    def _update_vol(self, anchor_price: float) -> float:
        lam = float(self.vol_lambda)
        if self._last_px is None:
            self._last_px = anchor_price
            return float(self._sigma)
        r = float(anchor_price) - float(self._last_px)
        self._last_px = anchor_price
        self._sigma2 = lam * self._sigma2 + (1.0 - lam) * (r * r)
        self._sigma = self._sigma2 ** 0.5
        return float(self._sigma)

    def compute_quote(self, book: OrderBook, inventory_norm: float, fair_hint: Optional[float] = None, token_id: Optional[str] = None) -> Optional[Quote]:
        # Fair blend: mid, microprice, and previous EWMA (or hint as anchor)
        m = book.mid()
        if m is None:
            return None
        micro = book.microprice() or m
        prev_ewma = self.fair_ewma.value if self.fair_ewma.value is not None else m
        anchor = fair_hint if fair_hint is not None else prev_ewma
        fair_raw = float(self.w_mid) * float(m) + float(self.w_micro) * float(micro) + float(self.w_hint) * float(anchor)
        fair = self.fair_ewma.update(fair_raw)
        try:
            logger.debug(
                "fair_components: m=%.6f micro=%.6f anchor=%.6f w=(%.2f,%.2f,%.2f) fair_raw=%.6f fair_ewma=%.6f",
                float(m), float(micro), float(anchor), float(self.w_mid), float(self.w_micro), float(self.w_hint), float(fair_raw), float(fair)
            )
        except Exception:
            pass
        # Hard bounds only (do not clamp to book to avoid masking edge)
        fair = float(min(0.99, max(0.01, fair)))

        # Volatility (EWMA of squared returns) on the fair anchor
        sigma = self._update_vol(fair)
        # Clip sigma to avoid explosive deltas on transient spikes
        clipped_sigma = False
        if float(sigma) >= float(self.max_sigma):
            clipped_sigma = True
            sigma = float(self.max_sigma)
            try:
                if token_id:
                    sigma_clip_hits_total.labels(token_id=token_id).inc()
            except Exception:
                pass
        try:
            logger.debug("volatility_state: sigma=%.6f sigma2=%.8f lam=%.3f", float(self._sigma), float(self._sigma2), float(self.vol_lambda))
        except Exception:
            pass

        # Liquidity scale k from visible top depth
        qb = float(book.bids[0].size) if getattr(book, "bids", None) and book.bids else 0.0
        qa = float(book.asks[0].size) if getattr(book, "asks", None) and book.asks else 0.0
        k_liq = float(self.k_base) + float(self.k_depth_coeff) * (qb + qa)
        k_liq = max(float(self.k_min), k_liq)

        gamma = float(self.inv_gamma)
        fee_ticks = float(self.k_fee_ticks)
        t = float(self.tick)
        half_fee = fee_ticks * t

        # Avellaneda-lite reservation price and half-spread
        r = float(fair) - (gamma / (2.0 * k_liq)) * float(inventory_norm)
        delta_core = (gamma * (sigma ** 2)) / (2.0 * k_liq)
        # Enforce tick minimum and add fee compensation
        delta_preclip = float(delta_core)
        delta = max(t, float(delta_core)) + half_fee
        if delta != (delta_preclip + half_fee) and not clipped_sigma:
            try:
                if token_id:
                    delta_clip_hits_total.labels(token_id=token_id).inc()
            except Exception:
                pass
        try:
            logger.debug(
                "r_delta_calc: fair=%.6f inv_norm=%.6f gamma=%.4f k_liq=%.6f fee_ticks=%.2f tick=%.4f r=%.6f delta=%.6f qb=%.2f qa=%.2f",
                float(fair), float(inventory_norm), float(gamma), float(k_liq), float(fee_ticks), float(t), float(r), float(delta), float(qb), float(qa)
            )
        except Exception:
            pass

        bid = max(0.01, min(0.99, r - delta))
        ask = max(0.01, min(0.99, r + delta))
        # Book-aware safety: keep a tick inside bests when present
        b0 = book.best_bid()
        a0 = book.best_ask()
        if a0 is not None and b0 is not None:
            ask = max(ask, float(b0) + t)
            bid = min(bid, float(a0) - t)
        # Round to tick and ensure non-crossing
        bid = max(0.01, min(0.99, round(bid / t) * t))
        ask = max(0.01, min(0.99, round(ask / t) * t))
        # Enforce spacing relative to current bests if available
        if a0 is not None:
            bid = min(bid, float(a0) - t)
        if b0 is not None:
            ask = max(ask, float(b0) + t)
        # Empty-book nudge: if no bests, avoid hard edges by 1 tick
        if a0 is None and b0 is None:
            pre_bid, pre_ask = bid, ask
            bid = max(0.01 + t, bid)
            ask = min(0.99 - t, ask)
            if (bid != pre_bid or ask != pre_ask) and token_id:
                try:
                    empty_book_nudges_total.labels(token_id=token_id).inc()
                except Exception:
                    pass
        if bid >= ask:
            bid = max(0.01, round(min(bid, ask - t) / t) * t)
        try:
            logger.debug(
                "final_quotes: bid=%.6f ask=%.6f clamped_in=[%.6f,%.6f] tick=%.4f",
                float(bid), float(ask), float(b0 if b0 is not None else -1.0), float(a0 if a0 is not None else -1.0), float(t)
            )
        except Exception:
            pass
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
    # Ensure positive tick/step for correct ladder direction
    t = abs(tick) if tick != 0 else 0.01
    st = max(1, abs(step_ticks))

    def _round_to_tick(x: float) -> float:
        return max(t, min(1.0 - t, round(x / t) * t))
    bid_prices: List[float] = []
    ask_prices: List[float] = []
    sizes: List[float] = []
    for i in range(max(1, layers)):
        size = min(max_size, base_size * (1.5 ** i))
        r = rng.random() if rng is not None else random.random()
        jitter = (r - 0.5) * 2.0 * jitter_ticks * t
        # Base ladder direction: bids descend, asks ascend
        bid = _round_to_tick(base_quote.bid - t * st * i + jitter)
        r2 = rng.random() if rng is not None else random.random()
        jitter2 = (r2 - 0.5) * 2.0 * jitter_ticks * t
        ask = _round_to_tick(base_quote.ask + t * st * i + jitter2)
        # Enforce monotonicity despite jitter/rounding
        if i > 0:
            if bid >= bid_prices[-1]:
                bid = max(t, bid_prices[-1] - t)
            if ask <= ask_prices[-1]:
                ask = min(1.0 - t, ask_prices[-1] + t)
        bid_prices.append(bid)
        ask_prices.append(ask)
        sizes.append(size)
    try:
        logger.debug(
            "LayeredQuotes: layers=%d tick=%.4f step=%d base_size=%.2f max_size=%.2f first_bid=%.4f first_ask=%.4f last_bid=%.4f last_ask=%.4f sizes=%s",
            int(max(1, layers)), float(tick), int(step_ticks), float(base_size), float(max_size),
            float(bid_prices[0]), float(ask_prices[0]), float(bid_prices[-1]), float(ask_prices[-1]), [round(s, 2) for s in sizes]
        )
    except Exception:
        pass
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