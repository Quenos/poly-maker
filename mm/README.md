## Polymarket Market-Making Daemon (mm)

This module provides a production-oriented market-making daemon for Polymarket binary markets. It reads a curated list of markets from Google Sheets, filters by liquidity and volume, enriches metadata from the Gamma API, consumes live market data from the CLOB WebSocket, computes two-sided quotes (Avellaneda-lite), and places orders via py-clob-client. It persists basic state in sqlite and exposes Prometheus metrics.

### Features
- Import Selected Markets from Google Sheets
- Filtering by thresholds and MM score (Liquidity × sqrt(Weekly Volume))
- Gamma API market enrichment
- CLOB WebSocket market data (order books + trades) with sequence tracking and gap recovery
- REST snapshot backfill via POST /prices
- Avellaneda-lite quoting (EWMA microprice, volatility, inventory skew)
- Layered quotes with jitter and requote triggers
- Basic risk controls (soft/hard inventory caps)
- sqlite persistence for orders/fills/positions (mm_state.db)
- Prometheus metrics (WS status, seq gaps, snapshot reloads, trades/min)

### Prerequisites
- Python 3.10+
- Virtual environment (recommended) at project root: `.venv/`
- Google Sheets credentials/json and `SPREADSHEET_URL` env configured (see `poly_utils/google_utils.py`)

### Installation
```bash
cd /Users/coenkuijpers/projects/poly-maker
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Required Environment Variables
- `SPREADSHEET_URL`: URL of the Google Sheet containing a tab named "Selected Markets" with columns:
  - `market_id`, `yes_token_id`, `no_token_id`, `Liquidity`, `Volume_24h`, `Volume_7d`, `Volume_30d`
- `PK`: Private key for CLOB signing (hex string) used by py-clob-client
- `BROWSER_ADDRESS`: Wallet address used as funder for CLOB client

Optional:
- `POLYGON_RPC_URL` (default: `https://polygon-rpc.com`)
- Threshold overrides (defaults shown):
  - `MIN_LIQUIDITY=10000`
  - `MIN_WEEKLY_VOLUME=50000`
  - `MIN_TREND=0.30`
  - `MM_SCORE_MIN=1000000`
  - `K_VOL=2.0`, `K_FEE_TICKS=1`, `ALPHA_FAIR=0.2`, `EWMA_VOL_WINDOW_SEC=600`, `INV_GAMMA=1.0`
  - `SOFT_CAP_DELTA_PCT=0.015`, `HARD_CAP_DELTA_PCT=0.03`
  - `ORDER_LAYERS=3`, `BASE_SIZE_USD=300`, `MAX_SIZE_USD=1500`
  - `REQUOTE_MID_TICKS=1`, `REQUOTE_QUEUE_LEVELS=2`, `ORDER_MAX_AGE_SEC=12`
  - `DAILY_LOSS_LIMIT_PCT=1.0`

### Run the Daemon
```bash
source /Users/coenkuijpers/projects/poly-maker/.venv/bin/activate
PYTHONPATH=/Users/coenkuijpers/projects/poly-maker \
python /Users/coenkuijpers/projects/poly-maker/mm/main.py
```

What it does:
- Reads `Selected Markets` from Sheets
- Applies thresholds and computes `mm_score`
- Enriches via Gamma `/markets`
- Subscribes to CLOB WS (orderbooks + trades) and backfills book snapshots via REST `/prices`
- Computes quotes and places layered orders through py-clob-client

### Market Data Interface
`mm/market_data.py` exposes:
- `subscribe(token_ids: list[str])`
- `get_signals(token_id)` → `best_bid`, `best_ask`, `mid`, `microprice`, `book_seq`
- `get_trade_stats(token_id)` → `last_price`, `trades_per_min`, `imbalance`

Internals:
- Maintains L2 books with `seq` per token
- Detects gaps; reloads snapshot via REST and replays buffered diffs
- Heartbeat, auto-reconnect with backoff, resubscribe

### Strategy
`mm/strategy.py` (Avellaneda-lite):
- Fair price: EWMA of microprice
- Volatility: EWMA of log returns
- Half-spread: `h = k_vol * sigma + k_fee`
- Reservation shift: `Δr = -γ * q_norm`
- Bid/Ask: `clip(fair + Δr ± h, 0.01, 0.99)`
- Layered quotes with jitter and size geometry
- Basic inventory soft/hard caps

### Orders
`mm/orders.py`: thin wrapper over py-clob-client
- Places signed orders with retries
- Market-wide cancel helpers are available

### Persistence & Metrics
- sqlite file: `mm_state.db` (orders, fills, positions)
- Prometheus metrics server (default port 9108) started by `StateStore`
  - `mm_ws_connected`
  - `mm_ws_reconnects_total`
  - `mm_seq_gaps_total{token_id}`
  - `mm_snapshot_reload_total{token_id}`
  - `mm_trades_per_min{token_id}`

### Tests
Run MarketData tests:
```bash
source .venv/bin/activate
PYTHONPATH=/Users/coenkuijpers/projects/poly-maker \
pytest -q tests/mm/test_market_data.py
```

### Notes & Next Steps
- Fills/positions ingestion can be extended via Data-API `/trades` or user WS
- Add advanced risk (resolution proximity, daily loss limit, markout widening)
- Persist periodic state snapshots, order lifecycle, and PnL reporting


