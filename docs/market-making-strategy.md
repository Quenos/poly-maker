## Polymarket Market-Making Strategy

This document explains how the market-making strategy in this repository operates end-to-end. It covers the runtime architecture, data flow, trading logic, risk management, state management, and all external endpoints used by the system.

### High-level goals
- Provide two-sided liquidity on selected Polymarket markets.
- Scale positions up to a configurable cap while quoting around the spread.
- Manage risk via stop-loss, take-profit, volatility checks, reverse-position checks, and position merging.

### Key modules and responsibilities
- `main.py`: Process orchestrator. Initializes client, loads configuration and state, spawns background updates, and maintains market and user WebSocket connections.
- `poly_data/polymarket_client.py`: Thin wrapper around Polymarket CLOB client and on-chain helpers. Creates/cancels orders, fetches order books, gets balances/positions, and triggers merges.
- `poly_data/websocket_handlers.py`: Connects to market and user WebSockets, subscribes/authenticates, receives updates, and hands them to processors.
- `poly_data/data_processing.py`: Maintains in-memory order books, reacts to book/price updates, and triggers `trading.perform_trade` per market; processes user trade/order events and updates local state.
- `trading.py`: Core trading logic. Computes quotes and sizes, applies risk rules, and decides when to place/cancel/update orders.
- `poly_data/data_utils.py`: Loads configuration and hyperparameters from Google Sheets; keeps `global_state` in sync with positions and open orders while coordinating with trade lifecycle state.
- `poly_data/global_state.py`: Shared in-memory state across modules (markets, books, parameters, positions, orders, in-flight trades, etc.).
- `poly_data/trading_utils.py`: Quote construction utilities (best bid/ask, pricing rules, sizing logic, rounding helpers).

### Startup and lifecycle (main loop)
1. Load environment via `dotenv`.
2. Initialize `PolymarketClient` (CLOB host, API creds, Polygon RPC, contracts).
3. Seed state:
   - `update_markets()`: Load selected markets and hyperparameters from Google Sheets into `global_state.df`/`params` and map `REVERSE_TOKENS`.
   - `update_positions()`: Pull wallet positions from the Data API; populate `global_state.positions`.
   - `update_orders()`: Pull open orders; populate `global_state.orders`.
4. Start a background thread that every 5s:
   - Clears stale in-flight trades (>15s) from `global_state.performing`.
   - Refreshes positions/average prices (`avgOnly=True`) and open orders.
   - Every 30s also refreshes markets and parameters from Google Sheets.
5. Enter an infinite loop that keeps two WebSocket connections alive in parallel:
   - Market stream: order-book and price updates for tracked tokens.
   - User stream: authenticated order/trade lifecycle events.
   On disconnect, both reconnect after a short delay.

### Data flow
- Market WebSocket → `process_data` updates an in-memory SortedDict book per market, then asynchronously triggers `trading.perform_trade(market)`.
- User WebSocket → `process_user_data` updates open orders and positions based on server-side order/trade events, manages the in-flight trade sets, and can retrigger trading for the relevant market.
- Background refresh → reconciles positions and orders via REST when no trades are pending (to avoid fighting with in-flight updates).
- Google Sheets → configuration (`Selected Markets`, `All Markets`) and hyperparameters (`Hyperparameters`) drive which markets to trade and strategy thresholds.

### Global state snapshot (selected)
- `all_tokens`: List of asset IDs to subscribe to.
- `REVERSE_TOKENS`: Mapping between opposite outcomes in the same market.
- `all_data[market]`: Order-book bids/asks as `SortedDict`s.
- `df`: Per-market config row (question, tick size, min size, trade size, spread caps, neg-risk flag, etc.).
- `params[param_type]`: Hyperparameters dict (stop-loss, take-profit, volatility thresholds, sleep period, etc.).
- `orders[token]`: Current open buy/sell order state `{price, size}` per side.
- `positions[token]`: Position tracking `{size, avgPrice}`.
- `performing[col]`: Set of in-flight trade IDs, keyed by `"{token}_{side}"`.
- `performing_timestamps[col][id]`: Wall-clock timestamp for stale-trade pruning.

### Trading loop details (`trading.perform_trade`)
This function is lock-guarded per market to prevent concurrent trading on the same market.

1. Load the market config row (`global_state.df`) and derive decimal precision from `tick_size`.
2. Compute quote inputs using recent book:
   - `get_best_bid_ask_deets`: Best/second-best/top price levels and size, plus liquidity within a deviation band. Handles token inversion for the reverse outcome (`token2`).
   - Round to `tick_size` precision and compute mid, spread, and a liquidity ratio.
3. Retrieve current position and average price for both outcomes; compute opposite-outcome exposure.
4. Merge positions (capital recovery): If both outcomes have positions above `CONSTANTS.MIN_MERGE_SIZE`, call `PolymarketClient.merge_positions` (Node script) for on-chain merge, then adjust local positions via `set_position`.
5. Construct target quotes:
   - `get_order_prices`: Start at best levels, adjust by `tick_size`, guard against crossing the spread, optionally anchor asks to average price when needed.
   - Round prices to tick precision.
6. Determine order sizes:
   - `get_buy_sell_amount(position, bid_price, row, other_token_position)`: Computes desired buy/sell notional using `trade_size`, optional `max_size` cap, min-size enforcement, low-price multiplier, and combined exposure across outcomes.
7. Risk controls and actions:
   - Stop-loss: if PnL < `stop_loss_threshold` with acceptable `spread_threshold`, or 3-hour volatility exceeds `volatility_threshold`, sell at best bid, cancel market orders, and write a risk-off window (`sleep_period`) to a file to delay re-entry.
   - Reverse-position guard: if holding significant opposite-outcome size, skip or cancel new buys for this token.
   - Volatility/anchor guard: if 3-hour volatility is high or price deviates > 0.05 from the sheet reference, cancel buy orders.
   - Liquidity ratio check: if bid/ask liquidity ratio is negative, suppress buys and cancel.
8. Order management rules:
   - Buy path: Only if `position < max_size`, absolute cap (250) not exceeded, desired `buy_amount ≥ min_size`, not in risk-off.
   - Sell path: Calculate a take-profit anchor (`avgPrice * (1 + take_profit_threshold%)`) and update if current sell differs sufficiently or sell size is too small relative to position.
   - Price bounds: Buys must be in [0.1, 0.9) to avoid extremes.
   - Replace vs keep: If price/size changed beyond thresholds, cancel and re-place; otherwise keep the existing order.
   - All order transitions go through `send_buy_order` and `send_sell_order` which encapsulate cancel/replace logic.

### Order and position tracking (user events)
- Trade events: On MATCHED, add trade ID to `performing`, update position immediately with executed size/price (maker vs taker aware), and retrigger trading. On MINED/CONFIRMED/FAILED, clear from `performing` and reconcile (FAILED triggers a positions refresh).
- Order events: Update `orders[token][side] = original_size - size_matched` and retrigger trading.
- REST reconciliation: `update_positions(avgOnly=True)` only overwrites sizes when there are no in-flight trades for the token and when the last trade update is not too recent, preventing oscillations.
- Stale in-flight cleanup: Any `performing` trade older than ~15s is pruned to avoid blocking reconciliations.

### Parameterization (Google Sheets)
Hyperparameters come from the `Hyperparameters` sheet and are grouped by a `type` key referenced by each market row. Important parameters include (names as they appear in the sheet):
- `stop_loss_threshold`: PnL % below which to risk-off (with spread check).
- `spread_threshold`: Max spread to allow stop-loss execution.
- `volatility_threshold`: 3-hour volatility ceiling for buying; above this triggers risk-off or cancels buys.
- `sleep_period`: Hours to wait before buying again after a risk-off.
- Per-market fields (from the merged market sheets): `tick_size`, `min_size`, `trade_size`, optional `max_size`, `multiplier` for low prices, `max_spread`, `neg_risk` (TRUE/FALSE), and reference best bid/ask used for sanity checks.

### Negative risk markets and merging
When `neg_risk` is TRUE, orders are created with the appropriate flag so the CLOB handles them as negative risk. If both outcomes are held simultaneously, the strategy calls `merge_positions(amount, condition_id, is_neg_risk)` to recover USDC on-chain via a Node script (`poly_merger/merge.js`), then decrements both positions locally.

### Environment
- `PK`: Wallet private key used by the CLOB client for API creds derivation.
- `BROWSER_ADDRESS`: Public wallet address used for balances and user identity with the CLOB.
- `POLYGON_RPC_URL` (optional): RPC endpoint for on-chain reads and the merge script (defaults to `https://polygon-rpc.com`).

### Failure handling and resilience
- WebSockets auto-reconnect after brief delays.
- Mined/confirmed/failed trade transitions clear `performing` and/or trigger a positions refresh.
- Background refresh reconciles drift, gated by in-flight sets and recent update timestamps.
- All trading per-market is guarded by an `asyncio.Lock` to avoid race conditions.

### Endpoints overview

Below is a reference of all remote endpoints/services the strategy calls, grouped by protocol. For CLOB REST calls, the exact paths are managed by the `py_clob_client` library; the host is listed.

#### WebSocket endpoints
- `wss://ws-subscriptions-clob.polymarket.com/ws/market`
  - Subscribes to order book and price updates for specified `assets_ids`; feeds the trading loop with real-time depth.
- `wss://ws-subscriptions-clob.polymarket.com/ws/user`
  - Authenticated user stream for order and trade lifecycle events; drives local order/position state and trade lifecycle (`performing`).

#### HTTP endpoints (Polymarket Data API)
- `https://data-api.polymarket.com/value?user={address}`
  - Returns the wallet’s total marked-to-market position value; used for balance aggregation.
- `https://data-api.polymarket.com/positions?user={address}`
  - Returns per-asset positions with `size` and `avgPrice`; used for positions sync.

#### HTTP host (Polymarket CLOB API via SDK)
- `https://clob.polymarket.com`
  - Base host for order create/cancel/get and order book queries via `py_clob_client`; the SDK handles request signing and routes.

#### Polygon RPC (on-chain)
- `https://polygon-rpc.com` (or `POLYGON_RPC_URL`)
  - Used for USDC ERC-20 balance reads, conditional token balance reads, and by the merge script for on-chain transactions.

#### Google Sheets
- Spreadsheet worksheets: `Selected Markets`, `All Markets`, `Hyperparameters`
  - Provide market selection, per-market fields (tokens, question, tick size, sizes), and per-type hyperparameters for strategy behavior.

### Glossary

- **On-chain merging (a.k.a. merge positions)**
  - **What it is**: When you hold both complementary outcomes of the same market (e.g., YES and NO ERC-1155 outcome tokens), you can “merge” equal amounts of those shares back into the collateral (USDC). Technically, this burns matched YES/NO shares via the Conditional Tokens contracts (and, for certain markets, through the Negative Risk Adapter), returning collateral to your wallet. There is no slippage because it is not a trade on the CLOB; it is a contract operation.
  - **Why/when used**: Free up collateral and reduce risk when you’ve accumulated opposing positions. The strategy checks for overlapping positions above a minimum threshold and merges them instead of trying to unwind through the order book.
  - **How it works here**: `trading.perform_trade` detects overlap and calls `PolymarketClient.merge_positions(amount_to_merge, condition_id, is_neg_risk)`. That method shells out to `poly_merger/merge.js`, which performs the on-chain transaction on Polygon using the addresses in `poly_data/polymarket_client.py` (`neg_risk_adapter`, `conditional_tokens`, `collateral`). After a successful merge, the strategy updates local positions for both tokens via `set_position(..., side='SELL', size=scaled_amt, price=0, source='merge')` to reflect the burned shares.
  - **Inputs**: `amount_to_merge` (raw token amount, 1e6 base units), `condition_id` (market), `is_neg_risk_market` (TRUE/FALSE). The code scales between raw and share units as needed.
  - **Effects**: Reduces both outcome positions by the merged amount; increases USDC balance by the recovered collateral; cancels exposure without trading through the spread.
  - **Costs and requirements**: Consumes gas on Polygon; may require prior approvals/allowances depending on flow; needs sufficient matched amounts on both outcomes (below threshold, merge is skipped). Environment variable `POLYGON_RPC_URL` can redirect RPC provider if needed.
  - **Alternatives**: Unwind via CLOB orders (market/limit) if merging isn’t possible. At resolution time, redeeming resolved outcome tokens returns collateral based on the result.

- **Negative risk markets**
  - **What it is (conceptually)**: A collateral-efficiency mechanism that treats complementary outcomes (e.g., YES/NO of the same market) as a risk-neutral pair. Instead of double-counting collateral when a trader holds both sides, a Negative Risk Adapter (on-chain) allows those opposing positions to be recognized as net-flat exposure and redeemed back into collateral via “merging.” In practice, it prevents “double margining” and keeps payoff bounded (sum of outcomes ≤ collateral), improving capital efficiency and market quality.
  - **Why it exists**: Without negative-risk treatment, buying both YES and NO would lock twice the collateral for a payoff that is capped by design (one side goes to 1, the other to 0). The adapter solves this by:
    - Netting opposing positions so margin reflects true risk.
    - Enabling deterministic redemption (merge) of matched YES/NO back to USDC without trading through the book.
    - Supporting better liquidity, tighter spreads, and more inventory-neutral market making.
  - **What it means for a trader**:
    - You can hold both sides without needing “double” collateral for the same risk; matched amounts can be merged to reclaim USDC.
    - If you accidentally end up long both sides (“boxed” position), you don’t need to cross the spread to unwind—you can merge on-chain and get collateral back (minus gas/fees).
    - Pricing often respects a sum-to-one style bound at the market level; you may observe tighter, more consistent pricing between YES and NO due to collateral coupling.
    - UX-wise, most of this is transparent; what matters is that opposing positions are capital-efficient and redeemable.
  - **How the strategy handles them (implementation)**:
    - Order placement: `PolymarketClient.create_order(..., neg_risk=True)` sets `PartialCreateOrderOptions(neg_risk=True)` so the CLOB treats orders under the adapter’s accounting.
    - Position merging: `merge_positions(..., is_neg_risk_market=True)` routes the on-chain merge through the adapter (`neg_risk_adapter` address) to burn matched YES/NO and return USDC.
  - **Operational implications and caveats**:
    - Requires correct contract addresses/allowances (`neg_risk_adapter`, `conditional_tokens`, `collateral`).
    - Merge is subject to minimum sizes (dust may not be mergeable); it consumes gas on Polygon.
    - Strategy logic (quoting, sizing, risk) stays the same; negative-risk mainly changes collateralization and the merge path under the hood.

### Endpoint status cross-check (as of latest docs)

Note: Polymarket maintains multiple surfaces: the CLOB API (`clob.polymarket.com`), the Data API (`data-api.polymarket.com`), and streaming via WebSockets at `ws-subscriptions-clob.polymarket.com`. Many REST paths are abstracted by `py_clob_client`.

- WebSockets
  - `wss://ws-subscriptions-clob.polymarket.com/ws/market` — Available. Official real-time market stream for book/price changes.
    - Alternative: Poll order books via CLOB REST if streaming unavailable (higher latency, not recommended).
  - `wss://ws-subscriptions-clob.polymarket.com/ws/user` — Available. Authenticated user stream for order/trade lifecycle.
    - Alternative: Periodically poll `client.get_orders()` and Data-API `/activity` for trade confirmations.

- Data-API
  - `GET https://data-api.polymarket.com/positions?user={address}` — Available. Official positions endpoint.
    - Alternative: The Graph subgraph for portfolio/positions if Data-API is degraded.
  - `GET https://data-api.polymarket.com/value?user={address}` — Available. Account value aggregation.
    - Alternative: Sum `USDC balance + Σ(position size × mid)` combining ERC20 balance via RPC and best price via CLOB.
  - `GET https://data-api.polymarket.com/activity?user={address}` — Available. On-chain/user activity including trades and rewards (repo uses it in `wallet_pnl.py`).
    - Alternative: `GET /trades` with user filter (where supported) or user WebSocket stream for real-time; subgraph for historical.
  - Markets/assets helpers used in utilities (e.g., `.../markets`, `.../assets?ids=...`) — Available as of latest checks; schema can vary (list vs keyed dict). The code already guards for both shapes.

- CLOB API (via SDK)
  - Host: `https://clob.polymarket.com` — Available. Order placement, cancellation, order book, and user orders are accessed through `py_clob_client`.
    - Alternatives when SDK/host is unreachable: none equivalent; consider exponential backoff and failover logic; for read-only book, you can use the market WebSocket.
  - `GET https://clob.polymarket.com/price?token_id=...&side=...` — Used in `wallet_pnl.py` for best price; currently available.
    - Alternatives: compute best from order book snapshots via WebSocket or the SDK’s order book call.

- Rewards (level-2 headers)
  - `GET https://polymarket.com/api/rewards/markets` — Available; requires L2 headers (repo constructs via `py_clob_client` headers utilities). Used by `poly_stats/account_stats.py` for earnings.
    - Alternatives: None public without L2 headers; fall back to zero earnings or cached results when blocked.

- Polygon RPC
  - `https://polygon-rpc.com` — Public RPC available. Alternatives: `https://rpc.ankr.com/polygon`, `https://polygon.llamarpc.com`, or provider-specific endpoints.

If any endpoint becomes unavailable, recommended fallback combinations:
- Positions and value: combine Data-API `/positions` with best prices from CLOB (WebSocket book or `GET /price`) and on-chain USDC via RPC to synthesize account value.
- User lifecycle: if user WebSocket is down, poll `client.get_orders()` plus Data-API `/activity` to detect MATCHED/MINED and reconcile state.

### Strategy flows and scenarios (worked examples)

- **Boot/start flow**
  - Initialize client and load sheets (`update_markets`), positions (`update_positions`), and open orders (`update_orders`).
  - Connect WebSockets (market + user). First book snapshot triggers `perform_trade(market)` per selected market.

- **Baseline quoting example (no positions)**
  - Inputs (example): best bid = 0.40, best ask = 0.60, `tick_size` = 0.01, `trade_size` ≥ `min_size`.
  - Quote construction (`get_order_prices`):
    - Bid → 0.41 (best bid + tick).
    - Ask → 0.59 (best ask − tick).
    - Guards ensure quotes don’t cross/collapse; if equal, fallback to top-of-book bounds.
  - Sizing (`get_buy_sell_amount`):
    - With `position = 0`, `max_size = row.get('max_size', trade_size)`, compute `buy_amount = min(trade_size, max_size)`.
    - If `buy_amount` ∈ (0.7*`min_size`, `min_size`) → bump to `min_size`.
  - Actions:
    - Place BUY at 0.41 for `buy_amount` (if volatility/ratio/reference checks pass).
    - Place SELL is skipped until a position and `avgPrice` exist.
    - Note: Initial buy quotes are placed on both YES and NO; asks are only posted once a position (and `avgPrice`) exists.

- **Scenario A — Book doesn’t move (no positions)**
  - Orders remain parked at 0.41/0.59.
  - No re-quotes unless price/size deviation thresholds trigger (e.g., >0.5¢ price diff or >10% size diff) or risk gates (volatility/ratio) change.

- **Scenario B — Both bid and ask move up (no positions)**
  - New best bid/ask e.g., 0.42/0.62 → recompute to 0.43/0.61 (tick-adjusted).
  - If current orders stale by threshold, cancel/replace via `send_buy_order`/`send_sell_order` (sell likely still skipped without position).

- **Scenario C — Both bid and ask move down (no positions)**
  - New best bid/ask e.g., 0.38/0.58 → recompute to 0.39/0.57.
  - Re-quote buys if thresholds/risk checks allow. Sells still skipped without a position.

- **Scenario D — With a dust position (e.g., 0.37 on YES, NO = 0)**
  - Sizing continues to “top up” until `max_size`, respecting `min_size` bump and price/volatility/ratio checks.
  - Reverse-position guard doesn’t block buys on NO (the other side) unless opposite side > `min_size`.
  - No sell orders until `avgPrice > 0` and `position ≥ trade_size`.

- **Scenario E — With a position (e.g., YES = 50 shares @ avgPrice 0.45)**
  - Buy side:
    - If `position < max_size` and `buy_amount ≥ min_size` and not in a risk-off window, place/maintain buys near 0.41 (or recomputed).
    - If `3_hour` volatility > threshold or price deviates > 0.05 from sheet reference, cancel buys.
    - If opposite outcome position > `min_size`, skip/cancel buys to avoid boxing unless needed.
  - Sell side (take-profit management):
    - Compute `tp_price = avgPrice × (1 + take_profit_threshold%)` and compare to `ask_price`.
    - If current sell order is >2% off target or sell size < 97% of position, refresh via `send_sell_order`.
  - Stop-loss/risk-off:
    - If PnL < `stop_loss_threshold` with acceptable spread or volatility too high, sell at best bid, cancel market orders, and write a `sleep_till` window to avoid immediate re-entry.

- **Scenario F — Opposing positions on both sides (YES = 60, NO = 45)**
  - Compute overlap `min(60, 45) = 45`.
  - If overlap > `MIN_MERGE_SIZE`, perform on-chain merge to reclaim USDC; update both positions down by 45.
  - Post-merge, continue quoting per standard rules with reduced exposure.

- **Scenario G — Liquidity ratio is adverse**
  - Definition: The liquidity ratio L is computed from the order book within a band around the mid price (the code uses a 10% band):
    - `L = (sum of bid sizes near mid) / (sum of ask sizes near mid)`.
    - An "adverse" ratio means the bid side is materially weaker than the ask side in that band (e.g., numerator near zero or far smaller than denominator), indicating poor support for immediate resale if you buy now.
  - Action: When the ratio indicates adverse conditions (implementation uses a conservative guard), the strategy suppresses new buys and cancels existing buy orders for that token to avoid accumulating inventory into weak demand. Sells and other risk logic continue as usual.

- **Scenario H — Position on one outcome, open order on the other (reverse-position guard)**
  - Example: You hold YES = 30 shares (avgPrice > 0) and have an open BUY order on NO for 20 shares.
  - Guard condition: If the reverse outcome’s position size exceeds `min_size`, the strategy will not add new buy orders on the other side. Concretely:
    - It skips creating new NO buys while the YES position > `min_size`.
    - If there are existing NO buy orders and their size is meaningful (>`MIN_MERGE_SIZE`), it may cancel them to avoid boxing the book.
  - Rationale: Avoid accumulating opposing exposure across outcomes. If both sides grow, the bot prefers on-chain merge (when overlap > `MIN_MERGE_SIZE`) to reclaim collateral rather than trading into a box via the CLOB.
  - Ongoing actions: Continue managing sells/take-profit for the side with position; keep buy-side quoting only for the side where reverse-position guard does not apply.

### File pointers (for reference)
The following code paths implement the above behavior:

```79:113:main.py
async def main():
    # Initialize, seed state, spawn refresh thread, and maintain WebSockets
```

```20:76:trading.py
def send_buy_order(order):
    # Cancel/replace logic and order placement with price bounds
```

```82:124:trading.py
def send_sell_order(order):
    # Cancel/replace logic for sells and order placement
```

```128:471:trading.py
async def perform_trade(market):
    # Core trading logic: merging, quoting, sizing, risk controls, order management
```

```9:50:poly_data/websocket_handlers.py
async def connect_market_websocket(chunk):
    # Market WebSocket subscription and message loop
```

```51:98:poly_data/websocket_handlers.py
async def connect_user_websocket():
    # User WebSocket auth and message loop
```

```32:53:poly_data/data_processing.py
def process_data(json_datas, trade=True):
    # Book maintenance and scheduling of perform_trade
```

```75:147:poly_data/data_processing.py
def process_user_data(rows):
    # User trade/order events → local positions/orders + trade lifecycle
```

```91:121:poly_data/data_utils.py
def update_orders():
    # Reconciles open orders and cancels duplicates per asset
```

```6:80:poly_data/polymarket_client.py
class PolymarketClient:
    # CLOB client init (host, creds, RPC) and helpers
```


