import random

from mm.strategy import AdvancedAvellanedaStrategy, StrategyState


def _mk_state(fair=0.5, sigma=0.02, inv_usd=0.0, bankroll=100000.0, ttr_sec=3600.0, tick=0.01):
    return StrategyState(
        fair=fair,
        sigma=sigma,
        inventory_usd=inv_usd,
        bankroll_usd=bankroll,
        time_to_resolution_sec=ttr_sec,
        tick=tick,
    )


def test_monotone_inventory_skew():
    strat = AdvancedAvellanedaStrategy(k_vol=2.0, k_fee_ticks=1.0, inv_gamma=1.0, min_spread_ticks=1)
    # Use deterministic jitter
    random.seed(123)
    q0 = strat.compute_quotes(_mk_state(inv_usd=0.0), soft_cap=0.2, hard_cap=0.4)
    random.seed(123)
    q1 = strat.compute_quotes(_mk_state(inv_usd=10000.0), soft_cap=0.2, hard_cap=0.4)
    random.seed(123)
    q2 = strat.compute_quotes(_mk_state(inv_usd=20000.0), soft_cap=0.2, hard_cap=0.4)

    # More long inventory should shift YES prices downward (more willingness to sell YES)
    assert q1.yes_bid <= q0.yes_bid and q2.yes_bid <= q1.yes_bid
    assert q1.yes_ask <= q0.yes_ask and q2.yes_ask <= q1.yes_ask


def test_event_time_scalers_affect_spread_and_size():
    strat = AdvancedAvellanedaStrategy(k_vol=2.0, k_fee_ticks=1.0, inv_gamma=1.0, min_spread_ticks=1)
    random.seed(7)
    far = strat.compute_quotes(_mk_state(ttr_sec=72 * 3600, sigma=0.02), soft_cap=0.5, hard_cap=1.0)
    random.seed(7)
    near = strat.compute_quotes(_mk_state(ttr_sec=1 * 3600, sigma=0.02), soft_cap=0.5, hard_cap=1.0)
    spread_far = near.yes_ask - near.yes_bid
    spread_near = far.yes_ask - far.yes_bid
    # Near resolution should have wider spread and reduced size multiplier
    assert spread_far > spread_near
    assert near.size_multiplier < far.size_multiplier


def test_min_spread_enforced():
    # Zero volatility, zero fee ticks, but min_spread_ticks forces spread
    strat = AdvancedAvellanedaStrategy(k_vol=0.0, k_fee_ticks=0.0, inv_gamma=1.0, min_spread_ticks=2)
    random.seed(99)
    st = _mk_state(sigma=0.0, tick=0.01)
    q = strat.compute_quotes(st, soft_cap=1.0, hard_cap=2.0)
    spread = q.yes_ask - q.yes_bid
    assert spread >= 2 * st.tick * 2  # 2 * min_h where min_h = 2 * tick -> spread >= 4 * tick


