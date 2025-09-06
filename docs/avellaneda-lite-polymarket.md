## Avellaneda-lite for Polymarket (Implementation Notes)

This document summarizes the quoting algorithm implemented in `mm/strategy.py::AvellanedaLite` and the sizing logic in `mm/market_maker.py`.

### Fair price (blend → EWMA → clamp)

- Inputs per token (from `OrderBook`):
  - `m` = mid = (best_bid + best_ask) / 2
  - `micro` = microprice = (a_size*b_price + b_size*a_price) / (a_size + b_size)
  - `fair_hint` (optional) from token pairing, and `prev_ewma` = previous fair EWMA value
- Weights (sum to 1): `w_mid`, `w_micro`, `w_hint`
- Blend anchor:
  - `anchor = fair_hint if provided else prev_ewma`
  - `fair_raw = w_mid*m + w_micro*micro + w_hint*anchor`
- Smooth:
  - `fair = EWMA(fair_raw)` with `alpha_fair`
- Clamp:
  - Hard bounds only: `fair ∈ [0.01, 0.99]`
  - We intentionally do not clamp fair to the current book to avoid masking signal when the book is thin/stale.

### Volatility (EWMA of squared fair deltas)

- Parameters: `vol_lambda ∈ (0,1)`
- Recursion:
  - `sigma2_t = vol_lambda * sigma2_{t-1} + (1 - vol_lambda) * (Δfair)^2`
  - `sigma_t = sqrt(sigma2_t)`

### Liquidity scale k

- Top-of-book depth: `Qb = best_bid_size`, `Qa = best_ask_size`
- Parameters: `k_base`, `k_depth_coeff`, `k_min`
- Formula:
  - `k_liq = max(k_min, k_base + k_depth_coeff * (Qb + Qa))`
- Notes: consider extending to multi-level depth if desired.

### Inventory normalization (USD)

- Definition (payout exposure at resolution):
  - `inventory_usd = (yes_shares - no_shares) * 1.0`
- Use a per-market budget `risk_budget_usd`.
- Normalized inventory:
  - `inventory_norm = inventory_usd / risk_budget_usd`
  - For sizing skew we also clamp to [-1, 1]: `I_norm = clamp(inventory_norm, -1, 1)`.

### Avellaneda-lite pricing

- Parameters: `gamma` (named `inv_gamma` in settings), `fee_ticks`, `tick`.
- Reservation price:
  - `r = fair - (gamma / (2 * k_liq)) * inventory_norm`
- Half-spread (before fees):
  - `δ_core = (gamma * sigma^2) / (2 * k_liq)`
- Add tick minimum and fee compensation:
  - `δ = max(tick, δ_core) + fee_ticks * tick`

### Quote formation (tick rounding + non-crossing + book safety)

- Raw quotes:
  - `bid_raw = r - δ`, `ask_raw = r + δ`
- Round to tick and clamp to hard bounds `[0.01, 0.99]`.
- Book-aware spacing (only if bests exist):
  - Enforce non-crossing with a one-tick buffer relative to current bests:
    - `bid ≤ best_ask - tick`
    - `ask ≥ best_bid + tick`
  - If no bests exist on either side, nudge off hard edges by one tick to avoid dust orders.
- Ensure `bid < ask` after all adjustments.

### Per-side USDC sizing with taper (inventory-aware)

- Total per-reprice budget: `K_total`.
- Inventory bias (flatten toward neutral):
  - `K_yes = 0.5 * K_total * (1 - I_norm)`
  - `K_no  = 0.5 * K_total * (1 + I_norm)`
- Taper across levels: `[40%, 30%, 20%, 10%]`.
- Ladders:
  - YES (BUY): prices at `bid - i * tick`
  - NO (implemented as SELL YES): prices at `ask + i * tick`
- Engine-facing sizes are provided in USD; we log the implied YES-shares as `usd_cap / price` for verification.
- Additional safeguards:
  - Minimum buy price
  - Position cap (shares)
  - Capacity check vs open BUY orders

### Parameters (primary)

- Fair blend: `w_mid`, `w_micro`, `w_hint`, `alpha_fair`
- Volatility: `vol_lambda`
- Liquidity scale: `k_base`, `k_depth_coeff`, `k_min`
- Inventory: `risk_budget_usd`, `inv_gamma`
- Fees & ticks: `fee_ticks`, `tick`
- Sizing: `K_total` (`per_reprice_usdc`), taper `[0.40, 0.30, 0.20, 0.10]`

### Telemetry & Logging

- Fair components: `m`, `micro`, `anchor`, weights, `fair_raw`, `fair_ewma`
- Volatility: `sigma`, `sigma2`, `vol_lambda` (clip at `max_sigma`)
- r/δ calc: `fair`, `inventory_norm`, `gamma`, `k_liq`, `fee_ticks`, `tick`, `qb`, `qa`, `r`, `delta`
- Final quotes: `bid`, `ask`, book bounds
- Sizing inputs: `risk_budget_usd`, `inventory_usd`, `I_norm`, `K_total`, `K_yes`, `K_no`, YES/NO ladder prices, per-level caps and implied shares

- Counters (Prometheus):
  - `mm_sigma_clip_hits_total{token_id}` – increments when `sigma >= max_sigma`
  - `mm_delta_clip_hits_total{token_id}` – increments when `delta` widened due to tick-min clip
  - `mm_empty_book_nudges_total{token_id}` – increments when bests are missing and quotes are nudged off edges

### Operational notes

- Hysteresis: replace gating on age and mid-shift (start at ≥1 tick; use 2 if churn is high).
- Volatility gating: optional multiplier on `δ` when `sigma` exceeds a high quantile (e.g., p95) to reduce adverse selection during spikes.



