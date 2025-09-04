import asyncio
import pandas as pd
import pytest

from mm.merge_manager import MergeManager, MergeConfig
from integrations.merger_adapter import MergeResult


@pytest.mark.asyncio
async def test_no_merge_below_min(monkeypatch):
    # Selected Markets with small overlap
    df = pd.DataFrame([
        {"question": "Q1", "token1": "t1", "token2": "t2", "condition_id": "0xabc", "neg_risk": False}
    ])

    # Patch sheet read
    monkeypatch.setattr("mm.merge_manager.get_spreadsheet", lambda read_only=False: object())
    monkeypatch.setattr("mm.merge_manager.read_sheet", lambda ss, name: df)

    # Positions: small overlap 0.05 shares -> 0.05 USDC
    monkeypatch.setattr("mm.merge_manager._fetch_positions_by_token", lambda addr: {"t1": 0.05, "t2": 0.05})

    calls = []

    def _fake_call(amount_6dp, cid, is_neg):
        calls.append((amount_6dp, cid, is_neg))
        return MergeResult(success=True, exit_code=0, stdout="", stderr="", tx_hash="0xdead")
    monkeypatch.setattr("mm.merge_manager.call_merger", _fake_call)

    cfg = MergeConfig(
        merge_scan_interval_sec=120,
        min_merge_usdc=0.10,  # higher than overlap
        merge_chunk_usdc=0.25,
        merge_max_retries=1,
        merge_retry_backoff_ms=1,
        dry_run=False,
    )
    mgr = MergeManager(cfg)
    await mgr.scan_and_merge(wallet_address="0xWALLET")
    assert calls == []  # no merges


@pytest.mark.asyncio
async def test_overlap_and_chunking_success(monkeypatch):
    df = pd.DataFrame([
        {"question": "Q1", "token1": "t1", "token2": "t2", "condition_id": "0xabc", "neg_risk": True}
    ])
    monkeypatch.setattr("mm.merge_manager.get_spreadsheet", lambda read_only=False: object())
    monkeypatch.setattr("mm.merge_manager.read_sheet", lambda ss, name: df)

    # Start with 0.60 shares overlap; after each success, reduce by the chunk size (0.25)
    state = {"t1": 0.60, "t2": 0.60}

    def _positions(_):
        return dict(state)
    monkeypatch.setattr("mm.merge_manager._fetch_positions_by_token", _positions)

    calls = []

    def _fake_call(amount_6dp, cid, is_neg):
        # emulate success and reduce positions by chunk in shares
        chunk_shares = amount_6dp / 1_000_000.0
        state["t1"] = max(0.0, state["t1"] - chunk_shares)
        state["t2"] = max(0.0, state["t2"] - chunk_shares)
        calls.append((amount_6dp, cid, is_neg))
        return MergeResult(success=True, exit_code=0, stdout="tx:0xabc", stderr="", tx_hash="0xabc")
    monkeypatch.setattr("mm.merge_manager.call_merger", _fake_call)

    # Avoid sleeping in test
    monkeypatch.setattr("mm.merge_manager.asyncio.sleep", lambda *_args, **_kw: asyncio.sleep(0))

    cfg = MergeConfig(
        merge_scan_interval_sec=120,
        min_merge_usdc=0.10,
        merge_chunk_usdc=0.25,
        merge_max_retries=2,
        merge_retry_backoff_ms=1,
        dry_run=False,
    )
    mgr = MergeManager(cfg)
    await mgr.scan_and_merge(wallet_address="0xWALLET")

    # Expected: chunks of 0.25, 0.25, 0.10 (remaining), all with neg_risk true
    amounts = [c[0] for c in calls]
    assert amounts[:2] == [250_000, 250_000]
    assert amounts[-1] == 100_000
    assert all(c[1] == "0xabc" for c in calls)
    assert all(c[2] is True for c in calls)


@pytest.mark.asyncio
async def test_retry_and_cooldown(monkeypatch):
    df = pd.DataFrame([
        {"question": "Q1", "token1": "t1", "token2": "t2", "condition_id": "0xabc", "neg_risk": False}
    ])
    monkeypatch.setattr("mm.merge_manager.get_spreadsheet", lambda read_only=False: object())
    monkeypatch.setattr("mm.merge_manager.read_sheet", lambda ss, name: df)
    monkeypatch.setattr("mm.merge_manager._fetch_positions_by_token", lambda addr: {"t1": 1.0, "t2": 1.0})

    attempts = {"n": 0}

    def _fail_call(amount_6dp, cid, is_neg):
        attempts["n"] += 1
        return MergeResult(success=False, exit_code=1, stdout="", stderr="boom", tx_hash=None)
    monkeypatch.setattr("mm.merge_manager.call_merger", _fail_call)
    monkeypatch.setattr("mm.merge_manager.asyncio.sleep", lambda *_args, **_kw: asyncio.sleep(0))

    cfg = MergeConfig(
        merge_scan_interval_sec=120,
        min_merge_usdc=0.10,
        merge_chunk_usdc=0.25,
        merge_max_retries=1,
        merge_retry_backoff_ms=1,
        dry_run=False,
    )
    mgr = MergeManager(cfg)
    await mgr.scan_and_merge(wallet_address="0xWALLET")
    # With max_retries=1, we should see exactly 1 attempt then cooldown
    assert attempts["n"] == 1
