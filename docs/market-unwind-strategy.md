## Market Unwind Strategy for De-selected Markets

This document describes how to safely and gradually exit markets that are removed from the "Selected Markets" Google Sheet while the bot is running. The goal is to stop new risk-taking immediately, cancel resting maker orders, and unwind any remaining positions in a controlled, low-impact manner.

### Objectives
- **Stop building new inventory** in de-selected markets immediately
- **Cancel all resting orders** for de-selected markets
- **Prefer merges** when both sides are held to recover collateral without crossing the spread
- **Gradually unwind residual positions** with small slices, respecting liquidity and minimum order sizes
- **Cleanup after exit**: drop subscriptions and in-memory tracking for markets fully unwound

## Change Detection
- **Track selected set** on each reload of the spreadsheet (e.g., inside `update_markets()`):
  - Build `new_selected` as the union of `token1` and `token2` from the updated `global_state.df`
  - Compare to `old_selected` (e.g., `global_state.selected_tokens`)
  - Compute deltas:
    - `added_tokens = new_selected - old_selected`
    - `removed_tokens = old_selected - new_selected`
  - Update `global_state.selected_tokens = new_selected`

## Immediate Actions on Removal
- **Cancel orders** for each token in `removed_tokens` and its pair (via `global_state.REVERSE_TOKENS`):
  - Use `client.cancel_all_asset(token)`
- **Do not place new maker orders** for removed tokens
- **Keep streaming order books** temporarily for removed tokens so we can exit efficiently (do not remove from `global_state.all_tokens` yet)

## Preferred Exit: Merge When Possible
- If both outcome tokens for a market are held (YES and NO), prefer merge:
  - Use `PolymarketClient.merge_positions(amount_to_merge, condition_id, is_neg_risk_market)` to recover collateral
  - Determine `amount_to_merge` from the smaller side’s raw amount
  - If merge succeeds, re-check remaining positions and proceed with residual unwind as needed

## Gradual Unwind of Residual Positions
- **Unwind mode** per token (tracked in `global_state.unwind_tokens`):
  - Fields: `remaining_target`, `min_size`, `slice_size`, `price_mode`, `last_attempt_ts`
  - Example default policy:
    - `slice_size = max(min_size, trade_size)` but capped to a small multiple of `min_size` (e.g., 1–2x)
    - `price_mode = at top-of-book`: for longs, sell at best bid; for the opposite side, unwind at best ask
    - **Spread/volatility guard**: skip if spread too wide or top size too thin
    - Retry every N seconds

### Exit Loop (concept)
1. Fetch current `position['size']` and best bid/ask from `global_state.all_data` (updated via websockets)
2. If `position <= dust_threshold`:
   - Cancel any residual orders
   - Remove token from `global_state.unwind_tokens`
   - Mark token eligible for cleanup (see below)
3. If no resting exit order near top-of-book:
   - Post an exit slice at price determined by `price_mode`
   - Ensure compliance with `min_size` and rate-limit posts
4. Repeat next cycle

## Cleanup After Exit
- Once a token is fully unwound (position at or below `dust_threshold`):
  - Cancel all orders for that token
  - Remove token from:
    - `global_state.unwind_tokens`
    - `global_state.all_tokens` (so future reconnects won’t subscribe)
  - Allow websocket reconnect (natural or forced) to drop the market subscription

## Minimal Integration Points
- **`poly_data/data_utils.update_markets()`**
  - Compute selected set deltas
  - For `removed_tokens`:
    - Cancel orders
    - If position size > `dust_threshold`, add an entry in `global_state.unwind_tokens`

- **`main.update_periodically()` loop**
  - After `update_positions(avgOnly=True)` and `update_orders()`:
    - Call a new `unwind_once()` function to process `global_state.unwind_tokens`
  - Optionally, if selected token set changed, schedule a controlled reconnect to resubscribe with the pruned `global_state.all_tokens`

## Practical Defaults
- **dust_threshold**: treat positions with shares < 1 (or configurable) as zero
- **slice_size**: `max(min_size, trade_size)`, capped at small multiple of `min_size`
- **rate limit**: post at most one new exit order per token per N seconds (e.g., 5–10s)
- **safety**: skip posting if the visible liquidity at top-of-book is below `min_size`

## Notes on Existing Code
- Sheet reload already occurs periodically in `main.update_periodically()` via `update_markets()`
- Stakes and orders are already tracked in `global_state.positions` and `global_state.orders`
- Use `client.cancel_all_asset(asset_id)` for cancellations, and `merge_positions(...)` for merges
- `global_state.REVERSE_TOKENS` provides pairing for outcomes in the same market

## Example Change Detection Snippet
```python
# inside update_markets() after updating global_state.df
selected_1 = set(global_state.df['token1'].astype(str))
selected_2 = set(global_state.df['token2'].astype(str))
new_selected = selected_1 | selected_2
old_selected = getattr(global_state, 'selected_tokens', set())

removed_tokens = old_selected - new_selected
added_tokens = new_selected - old_selected

global_state.selected_tokens = new_selected
```

## Safety Considerations
- Avoid aggressive exits that worsen execution (honor spread and top-of-book depth)
- Merge first whenever possible to avoid spread crossing
- Ensure no new maker inventory is posted once a market is de-selected


