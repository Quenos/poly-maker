import time

from mm.risk import RiskManager, StrategyStateLike


def _st(inv=0.0, br=100000.0):
    return StrategyStateLike(
        fair=0.5,
        sigma=0.02,
        inventory_usd=inv,
        bankroll_usd=br,
        time_to_resolution_sec=3600.0,
        tick=0.01,
    )


def test_inventory_caps_transitions():
    rm = RiskManager(soft_cap_delta_pct=0.1, hard_cap_delta_pct=0.2, daily_loss_limit_pct=10.0)
    rm.reset_day(100000.0)
    p1 = rm.apply(_st(inv=5000.0), token_id="t1", nav_usd=100000.0)
    p2 = rm.apply(_st(inv=15000.0), token_id="t1", nav_usd=100000.0)
    p3 = rm.apply(_st(inv=25000.0), token_id="t1", nav_usd=100000.0)
    assert p2.h_multiplier >= p1.h_multiplier and p2.gamma_multiplier >= p1.gamma_multiplier
    assert p3.ioc_reduce is True and p3.size_multiplier < p2.size_multiplier


def test_loss_limit_trigger():
    rm = RiskManager(soft_cap_delta_pct=0.5, hard_cap_delta_pct=1.0, daily_loss_limit_pct=1.0, daily_hard_stop_pct=2.0)
    rm.reset_day(100000.0)
    p_ok = rm.apply(_st(inv=0.0), token_id="t", nav_usd=99500.0)  # 0.5% drawdown
    p_warn = rm.apply(_st(inv=0.0), token_id="t", nav_usd=98500.0)  # 1.5% drawdown
    p_stop = rm.apply(_st(inv=0.0), token_id="t", nav_usd=98000.0)  # 2% drawdown
    assert p_ok.freeze_tightening is False and p_ok.hard_stop is False
    assert p_warn.freeze_tightening is True and p_warn.hard_stop is False
    assert p_stop.hard_stop is True and p_stop.size_multiplier == 0.0


def test_markout_widening_and_decay():
    rm = RiskManager(soft_cap_delta_pct=1.0, hard_cap_delta_pct=2.0, daily_loss_limit_pct=50.0, markout_widen_factor=1.5, markout_decay_halflife_sec=0.5)
    rm.reset_day(100000.0)
    # Record adverse fills
    now_state = _st(inv=0.0)
    for _ in range(3):
        rm.record_fill("tok", side="BUY", price=0.51, size_usd=1000.0, current_fair=0.50)
    p1 = rm.apply(now_state, token_id="tok", nav_usd=100000.0)
    h1 = p1.h_multiplier
    # Wait to decay
    time.sleep(1.0)
    p2 = rm.apply(now_state, token_id="tok", nav_usd=100000.0)
    assert p2.h_multiplier <= h1


