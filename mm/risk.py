import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import collections


@dataclass
class StrategyStateLike:
    fair: float
    sigma: float
    inventory_usd: float
    bankroll_usd: float
    time_to_resolution_sec: float
    tick: float


@dataclass
class AdjustedParams:
    h_multiplier: float
    gamma_multiplier: float
    size_multiplier: float
    freeze_tightening: bool
    hard_stop: bool
    ioc_reduce: bool


class RiskManager:
    def __init__(
        self,
        soft_cap_delta_pct: float,
        hard_cap_delta_pct: float,
        daily_loss_limit_pct: float,
        daily_hard_stop_pct: float = 2.0,
        markout_window_secs: Tuple[int, int, int] = (3, 10, 30),
        markout_widen_factor: float = 1.2,
        markout_decay_halflife_sec: float = 60.0,
    ) -> None:
        self.soft_cap = soft_cap_delta_pct
        self.hard_cap = hard_cap_delta_pct
        self.loss_limit = daily_loss_limit_pct
        self.hard_stop_limit = daily_hard_stop_pct
        self.markout_windows = markout_window_secs
        self.markout_widen_factor = markout_widen_factor
        self.markout_decay_halflife_sec = markout_decay_halflife_sec

        self.day_start_nav: Optional[float] = None
        self.last_nav: Optional[float] = None
        # Markout tracking per token: list of (timestamp, pnl_delta)
        self.markouts: Dict[str, Deque[Tuple[float, float]]] = collections.defaultdict(lambda: collections.deque(maxlen=1000))
        self.markout_penalty: float = 1.0
        self._last_decay_ts: float = time.time()

    def reset_day(self, starting_nav: float) -> None:
        self.day_start_nav = starting_nav
        self.last_nav = starting_nav
        self.markouts.clear()
        self.markout_penalty = 1.0
        self._last_decay_ts = time.time()

    def update_nav(self, nav: float) -> None:
        self.last_nav = nav

    def record_fill(self, token_id: str, side: str, price: float, size_usd: float, current_fair: float) -> None:
        # Positive markout means favorable; negative adverse
        pnl_delta = (current_fair - price) * size_usd if side.upper() == "BUY" else (price - current_fair) * size_usd
        self.markouts[token_id].append((time.time(), pnl_delta))

    def _compute_markout_penalty(self, now: float) -> float:
        # Decay previous penalty
        dt = max(0.0, now - self._last_decay_ts)
        decay = 0.5 ** (dt / max(1e-6, self.markout_decay_halflife_sec))
        # Move penalty towards 1.0 by decay; clamp minimum at 1.0
        self.markout_penalty = 1.0 + max(0.0, (self.markout_penalty - 1.0) * decay)
        self._last_decay_ts = now
        return self.markout_penalty

    def _check_markout(self, token_id: str, fair_now: float) -> float:
        now = time.time()
        decayed = self._compute_markout_penalty(now)
        buf = self.markouts.get(token_id)
        if not buf:
            return decayed
        # rolling sums for windows
        adverse = False
        for window in self.markout_windows:
            recent = [p for (t, p) in buf if now - t <= window]
            if recent and (sum(recent) / len(recent)) < 0.0:
                adverse = True
                break
        # Only apply widening once per adverse episode; thereafter decay back toward 1.0
        if adverse:
            if decayed <= 1.000001:
                widened = 1.0 * self.markout_widen_factor
                self.markout_penalty = widened
                return widened
        self.markout_penalty = decayed
        return decayed

    def apply(self, state: StrategyStateLike, token_id: str, nav_usd: float) -> AdjustedParams:
        if self.day_start_nav is None:
            self.reset_day(starting_nav=nav_usd)
        self.update_nav(nav_usd)

        # Inventory caps
        q_norm = 0.0
        if state.bankroll_usd > 0:
            q_norm = state.inventory_usd / state.bankroll_usd
        h_mult = 1.0
        g_mult = 1.0
        size_mult = 1.0
        ioc_reduce = False
        if abs(q_norm) >= self.soft_cap:
            h_mult *= 1.2
            g_mult *= 1.2
            size_mult *= 0.7
        if abs(q_norm) >= self.hard_cap:
            ioc_reduce = True
            size_mult *= 0.1

        # Daily loss limit gates
        freeze_tightening = False
        hard_stop = False
        if self.day_start_nav is not None and self.last_nav is not None and self.day_start_nav > 0:
            dd_pct = max(0.0, (self.day_start_nav - self.last_nav) / self.day_start_nav * 100.0)
            if dd_pct >= self.loss_limit:
                freeze_tightening = True
                h_mult *= 1.2
                g_mult *= 1.2
            if dd_pct >= self.hard_stop_limit:
                hard_stop = True
                size_mult = 0.0

        # Markout widening with decay
        h_mult *= self._check_markout(token_id, state.fair)

        return AdjustedParams(
            h_multiplier=h_mult,
            gamma_multiplier=g_mult,
            size_multiplier=size_mult,
            freeze_tightening=freeze_tightening,
            hard_stop=hard_stop,
            ioc_reduce=ioc_reduce,
        )


