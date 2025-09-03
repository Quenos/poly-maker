import time
from mm.risk import RiskManager
from mm.strategy import StrategyState


def test_markout_bad_widens():
    rm = RiskManager(soft_cap_delta_pct=1.0, hard_cap_delta_pct=2.0, daily_loss_limit_pct=50.0, markout_widen_factor=1.5, markout_decay_halflife_sec=0.1)
    rm.reset_day(100000.0)
    st = StrategyState(fair=0.5, sigma=0.02, inventory_usd=0.0, bankroll_usd=100000.0, time_to_resolution_sec=3600.0, tick=0.01)
    # Adverse fills
    for _ in range(3):
        rm.record_fill("tok", side="BUY", price=0.51, size_usd=1000.0, current_fair=0.50)
    a1 = rm.apply(st, token_id="tok", nav_usd=100000.0)
    assert a1.h_multiplier > 1.0
    time.sleep(0.3)
    a2 = rm.apply(st, token_id="tok", nav_usd=100000.0)
    assert a2.h_multiplier <= a1.h_multiplier



