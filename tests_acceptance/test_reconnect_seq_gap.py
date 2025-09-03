from mm.market_data import MarketData


def test_reconnect_and_seq_gap_replay():
    md = MarketData("wss://ws-subscriptions-clob.polymarket.com/ws/", "https://clob.polymarket.com")
    token = "t1"
    md.subscribe([token])
    # initial snapshot seq=1
    md._apply_update(token, [(0.49, 100)], [(0.51, 120)], seq=1, is_snapshot=True)
    # out-of-order update arrives seq=3 → buffer + request snapshot (internal)
    md._apply_update(token, [(0.50, 100)], [(0.52, 100)], seq=3, is_snapshot=False)
    sig1 = md.get_signals(token)
    assert sig1["best_bid"] == 0.49 and sig1["book_seq"] == 1.0
    # new snapshot seq=2 → expected becomes 2 then buffered seq=3 should replay
    md._apply_update(token, [(0.495, 100)], [(0.515, 100)], seq=2, is_snapshot=True)
    sig2 = md.get_signals(token)
    assert sig2["book_seq"] == 3.0 and sig2["best_bid"] == 0.50 and sig2["best_ask"] == 0.52



