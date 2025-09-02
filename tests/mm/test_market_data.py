import time
from typing import List, Tuple

import pytest

from mm.market_data import MarketData


def _b(levels: List[Tuple[float, float]]):
    return [(float(p), float(s)) for p, s in levels]


def test_subscribe_and_signals_defaults():
    md = MarketData("wss://ws-subscriptions-clob.polymarket.com/ws/", "https://clob.polymarket.com")
    md.subscribe(["1", "2"])
    sig = md.get_signals("1")
    assert set(sig.keys()) == {"best_bid", "best_ask", "mid", "microprice", "book_seq"}
    assert sig["best_bid"] is None and sig["best_ask"] is None


def test_in_order_seq_updates_apply_and_signals():
    md = MarketData("wss://ws-subscriptions-clob.polymarket.com/ws/", "https://clob.polymarket.com")
    token = "100"
    md.subscribe([token])
    # initial snapshot seq=1
    md._apply_update(token, _b([(0.49, 100)]), _b([(0.51, 120)]), seq=1, is_snapshot=True)
    sig = md.get_signals(token)
    assert sig["best_bid"] == 0.49
    assert sig["best_ask"] == 0.51
    assert sig["book_seq"] == 1.0
    # next diff seq=2
    md._apply_update(token, _b([(0.50, 150)]), _b([(0.52, 140)]), seq=2, is_snapshot=False)
    sig = md.get_signals(token)
    assert sig["best_bid"] == 0.50
    assert sig["best_ask"] == 0.52
    assert sig["book_seq"] == 2.0


def test_gap_detection_and_buffer_then_replay_after_snapshot():
    md = MarketData("wss://ws-subscriptions-clob.polymarket.com/ws/", "https://clob.polymarket.com")
    token = "200"
    md.subscribe([token])
    # snapshot seq=1
    md._apply_update(token, _b([(0.40, 100)]), _b([(0.60, 100)]), seq=1, is_snapshot=True)
    # out-of-order update seq=3 arrives, should be buffered and not applied
    md._apply_update(token, _b([(0.41, 90)]), _b([(0.59, 110)]), seq=3, is_snapshot=False)
    sig = md.get_signals(token)
    # still old snapshot
    assert sig["best_bid"] == 0.40
    assert sig["best_ask"] == 0.60
    # now a fresh snapshot seq=2 arrives; expected_seq becomes 2 and buffered seq=3 should replay
    md._apply_update(token, _b([(0.405, 95)]), _b([(0.595, 105)]), seq=2, is_snapshot=True)
    sig = md.get_signals(token)
    # after replay, we should be at seq 3 top of book
    assert sig["book_seq"] == 3.0
    assert sig["best_bid"] == 0.41
    assert sig["best_ask"] == 0.59


def test_trade_stats_rolling_window_and_imbalance():
    md = MarketData("wss://ws-subscriptions-clob.polymarket.com/ws/", "https://clob.polymarket.com")
    token = "300"
    md.subscribe([token])
    # record some trades
    md._record_trade(token, price=0.5, size=10, side="BUY")
    md._record_trade(token, price=0.51, size=5, side="SELL")
    md._record_trade(token, price=0.52, size=15, side="BUY")
    stats = md.get_trade_stats(token)
    assert stats["last_price"] == 0.52
    assert stats["trades_per_min"] >= 3.0
    # imbalance = (buys - sells) / total size = (25 - 5) / 30 = 0.6666...
    assert 0.6 <= stats["imbalance"] <= 0.8


