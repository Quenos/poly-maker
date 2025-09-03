from mm.risk import RiskManager
from mm.strategy import StrategyState


def test_inventory_cap_soft_and_hard():
    rm = RiskManager(soft_cap_delta_pct=0.1, hard_cap_delta_pct=0.2, daily_loss_limit_pct=50.0)
    st = StrategyState(fair=0.5, sigma=0.02, inventory_usd=15000.0, bankroll_usd=100000.0, time_to_resolution_sec=3600.0, tick=0.01)
    adj_soft = rm.apply(st, token_id="t", nav_usd=100000.0)
    assert adj_soft.h_multiplier > 1.0 and adj_soft.gamma_multiplier > 1.0
    st2 = StrategyState(fair=0.5, sigma=0.02, inventory_usd=25000.0, bankroll_usd=100000.0, time_to_resolution_sec=3600.0, tick=0.01)
    adj_hard = rm.apply(st2, token_id="t", nav_usd=100000.0)
    assert adj_hard.ioc_reduce is True



