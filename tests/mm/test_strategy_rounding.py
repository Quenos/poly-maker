import random

from mm.strategy import AdvancedAvellanedaStrategy, StrategyState, build_layered_quotes, Quote


def test_quotes_rounded_to_tick_after_jitter():
    strat = AdvancedAvellanedaStrategy(k_vol=0.0, k_fee_ticks=0.0, inv_gamma=1.0, min_spread_ticks=1)
    random.seed(42)
    st = StrategyState(fair=0.5, sigma=0.0, inventory_usd=0.0, bankroll_usd=1000.0, time_to_resolution_sec=3600.0, tick=0.01)
    q = strat.compute_quotes(st, soft_cap=1.0, hard_cap=2.0)
    # Prices should lie on tick grid
    assert abs((q.yes_bid / st.tick) - round(q.yes_bid / st.tick)) < 1e-9
    assert abs((q.yes_ask / st.tick) - round(q.yes_ask / st.tick)) < 1e-9


def test_layered_quotes_deterministic_with_rng_and_tick_rounded():
    from random import Random

    base = Quote(bid=0.45, ask=0.55)
    rng = Random(123)
    lq = build_layered_quotes(base_quote=base, layers=3, base_size=10.0, max_size=100.0, tick=0.01, step_ticks=1, jitter_ticks=0.25, rng=rng)
    # Deterministic across runs for given seed
    assert len(lq.bid_prices) == 3 and len(lq.ask_prices) == 3
    # Tick-rounded
    for p in lq.bid_prices + lq.ask_prices:
        assert abs((p / 0.01) - round(p / 0.01)) < 1e-9
