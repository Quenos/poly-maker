## Polymarket Market-Making Daemon (mm)

This module provides a production-oriented market-making daemon for Polymarket binary markets. It reads a curated list of markets from Google Sheets, enriches metadata from the Gamma API, consumes live market data from the CLOB WebSocket, computes two-sided quotes using Avellaneda-lite strategy, and places orders via py-clob-client. It persists state in sqlite and exposes Prometheus metrics.

### Features
- **Market Selection**: Import markets from Google Sheets with real-time monitoring
- **Token Processing**: Uses `token1` and `token2` columns from Selected Markets sheet
- **Gamma API Integration**: Enriches market data via targeted API calls
- **Live Market Data**: CLOB WebSocket for order books and trades with sequence tracking
- **Avellaneda-Lite Strategy**: EWMA microprice, volatility-based spreads, inventory skew
- **Layered Quoting**: Multi-level order placement with configurable sizes and jitter
- **Risk Management**: Soft/hard inventory caps, daily loss limits, markout penalties
- **Position Tracking**: Real-time position monitoring via Polymarket Data API
- **State Persistence**: SQLite database for orders, fills, positions, and selection snapshots
- **Prometheus Metrics**: Comprehensive monitoring and alerting
- **Enhanced Logging**: Real-time visibility into market selection changes and system updates

### Prerequisites
- Python 3.10+
- Virtual environment (recommended) at project root: `.venv/`
- Google Sheets credentials/json and `SPREADSHEET_URL` env configured
- Polymarket API access (CLOB and Data API)

### Installation
```bash
cd /Users/coenkuijpers/projects/poly-maker
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Configuration

The market making daemon is configured via environment variables. All parameters have sensible defaults but can be customized for your trading strategy.

#### Required Environment Variables
- `SPREADSHEET_URL`: URL of the Google Sheet containing a tab named "Selected Markets" with columns:
  - `token1`, `token2` (required), `condition_id` (recommended), `Liquidity`, `Volume_24h`, `Volume_7d`, `Volume_30d`
- `PK`: Private key for CLOB signing (hex string) used by py-clob-client
- `BROWSER_ADDRESS`: Wallet address used as funder for CLOB client

#### Optional Environment Variables

##### Market Selection Thresholds
- `MIN_LIQUIDITY`: Minimum liquidity required for market selection (default: 10000.0)
- `MIN_WEEKLY_VOLUME`: Minimum weekly volume required (default: 50000.0)
- `MIN_TREND`: Minimum trend score required (default: 0.30)
- `MM_SCORE_MIN`: Minimum market making score required (default: 1000000.0)

##### Avellaneda-Lite Quoting Strategy
- `K_VOL`: Volatility parameter for spread calculation (default: 2.0)
- `K_FEE_TICKS`: Fee parameter in ticks (default: 1.0)
- `ALPHA_FAIR`: Fair price adjustment factor (default: 0.2)
- `EWMA_VOL_WINDOW_SEC`: Exponential weighted moving average window for volatility (default: 600)
- `INV_GAMMA`: Inventory gamma parameter (default: 1.0)

##### Risk Management
- `SOFT_CAP_DELTA_PCT`: Soft cap for delta exposure percentage (default: 0.015)
- `HARD_CAP_DELTA_PCT`: Hard cap for delta exposure percentage (default: 0.03)
- `DAILY_LOSS_LIMIT_PCT`: Daily loss limit percentage (default: 1.0)

##### Order Management
- `ORDER_LAYERS`: Number of order layers to place (default: 3)
- `BASE_SIZE_USD`: Base order size in USD (default: 300.0)
- `MAX_SIZE_USD`: Maximum order size in USD (default: 1500.0)
- `REQUOTE_MID_TICKS`: Mid-price change threshold for requoting (default: 1)
- `REQUOTE_QUEUE_LEVELS`: Number of queue levels to consider for requoting (default: 2)
- `ORDER_MAX_AGE_SEC`: Maximum age of orders before replacement (default: 12)

##### Network Configuration
- `GAMMA_BASE_URL`: Gamma API base URL (default: "https://gamma-api.polymarket.com")
- `CLOB_BASE_URL`: CLOB API base URL (default: "https://clob.polymarket.com")
- `CLOB_WS_URL`: CLOB WebSocket URL (default: "wss://ws-subscriptions-clob.polymarket.com/ws/")
- `POLYGON_RPC_URL`: Polygon RPC endpoint (default: "https://polygon-rpc.com")

##### Sheet Configuration
- `SELECTED_SHEET_NAME`: Name of the sheet containing selected markets (default: "Selected Markets")

#### Configuration Examples

##### Basic Market Making Setup
```bash
# Required
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id
PK=0x1234567890abcdef...
BROWSER_ADDRESS=0x1234567890abcdef...

# Market Making Strategy
K_VOL=2.0
ALPHA_FAIR=0.2
ORDER_LAYERS=3
BASE_SIZE_USD=300.0
MAX_SIZE_USD=1500.0

# Risk Management
SOFT_CAP_DELTA_PCT=0.015
HARD_CAP_DELTA_PCT=0.03
DAILY_LOSS_LIMIT_PCT=1.0
```

##### Conservative Strategy
```bash
# Wider spreads, smaller sizes
K_VOL=3.0
ALPHA_FAIR=0.1
BASE_SIZE_USD=150.0
MAX_SIZE_USD=750.0

# Tighter risk controls
SOFT_CAP_DELTA_PCT=0.01
HARD_CAP_DELTA_PCT=0.02
DAILY_LOSS_LIMIT_PCT=0.5
```

##### Aggressive Strategy
```bash
# Tighter spreads, larger sizes
K_VOL=1.5
ALPHA_FAIR=0.3
BASE_SIZE_USD=500.0
MAX_SIZE_USD=2500.0

# Relaxed risk controls
SOFT_CAP_DELTA_PCT=0.02
HARD_CAP_DELTA_PCT=0.05
DAILY_LOSS_LIMIT_PCT=2.0
```

#### Parameter Tuning Guide

##### Market Selection Parameters
- **MIN_LIQUIDITY**: Higher values select more liquid markets, reducing slippage but potentially lower returns
- **MIN_WEEKLY_VOLUME**: Ensures markets have sufficient activity for reliable pricing
- **MIN_TREND**: Higher values select markets with stronger directional movement
- **MM_SCORE_MIN**: Higher values select markets with better market making opportunities

##### Quoting Strategy Parameters
- **K_VOL**: Controls spread width based on volatility (higher = wider spreads)
- **ALPHA_FAIR**: Adjusts how much to deviate from mid-price (higher = more aggressive)
- **INV_GAMMA**: Controls inventory risk adjustment (higher = more conservative)
- **EWMA_VOL_WINDOW_SEC**: How long to look back for volatility calculation

##### Risk Management Parameters
- **SOFT_CAP_DELTA_PCT**: Warning threshold for position exposure
- **HARD_CAP_DELTA_PCT**: Maximum allowed position exposure
- **DAILY_LOSS_LIMIT_PCT**: Daily loss limit as percentage of capital

##### Order Management Parameters
- **ORDER_LAYERS**: More layers provide better liquidity but increase complexity
- **BASE_SIZE_USD**: Base order size (adjust based on available capital)
- **MAX_SIZE_USD**: Maximum order size (prevents overexposure)
- **ORDER_MAX_AGE_SEC**: How frequently to refresh orders
- **REQUOTE_MID_TICKS**: How much mid-price must move to trigger requoting

### Run the Daemon

#### Production Mode
```bash
source /Users/coenkuijpers/projects/poly-maker/.venv/bin/activate
PYTHONPATH=/Users/coenkuijpers/projects/poly-maker \
python /Users/coenkuijpers/projects/poly-maker/mm/main.py
```

#### Test Mode
```bash
python -m mm.main --test
```
Test mode performs a dry run with no actual orders sent and logs to both console and `logs/mm_test_<timestamp>.log`.

### What It Does
1. **Initialization**: Reads "Selected Markets" from Google Sheets
2. **Market Enrichment**: Fetches additional data from Gamma API
3. **WebSocket Connection**: Subscribes to CLOB for live order book and trade data
4. **Quote Generation**: Computes Avellaneda-lite quotes with inventory skew
5. **Order Placement**: Places layered orders through py-clob-client
6. **Continuous Monitoring**: Checks for sheet changes every 15 minutes
7. **Risk Management**: Applies position caps and loss limits
8. **Position Tracking**: Monitors real-time positions via Polymarket Data API

### Market Selection Monitoring

The daemon provides comprehensive visibility into market selection changes:

- **Initial Load**: Shows how many markets were loaded from the sheet
- **Change Detection**: Logs when markets are added/removed from the sheet
- **Real-time Updates**: Shows progress through websocket restarts, strategy creation, etc.
- **Status Summary**: Provides complete state after each update
- **Periodic Status**: Logs current system state even when no changes occur

Example log output:
```
🎯 Initial market selection: 4 markets loaded from sheet
🔄 MARKET SELECTION CHANGE DETECTED - Updating trading configuration
Markets ADDED: ['token1', 'token2']
Markets REMOVED: ['old_token']
✅ Market selection update complete:
   - Active markets: 4
   - Market mappings: 4
   - Trading strategies: 4
```

### Core Components

#### Market Data (`mm/market_data.py`)
- **OrderBook**: Maintains L2 order book with sequence tracking
- **MarketData**: WebSocket connection management with automatic reconnection
- **Gap Detection**: Identifies sequence gaps and reloads snapshots via REST
- **Trade Tracking**: Maintains rolling trade statistics per token

#### Strategy (`mm/strategy.py`)
- **AvellanedaLite**: Core quoting algorithm with EWMA microprice
- **Volatility Calculation**: Rolling volatility estimation
- **Inventory Skew**: Position-based price adjustments
- **Layered Quotes**: Multi-level order placement with jitter

#### Risk Management (`mm/risk.py`)
- **Position Caps**: Soft and hard delta exposure limits
- **Daily Loss Limits**: Configurable daily loss thresholds
- **Markout Penalties**: Dynamic spread widening based on fill performance
- **Time Decay**: Automatic penalty reduction over time

#### State Management (`mm/state.py`)
- **SQLite Database**: Persistent storage for orders, fills, positions
- **Prometheus Metrics**: Real-time monitoring and alerting
- **Selection Snapshots**: Historical tracking of market selection changes
- **Thread Safety**: Concurrent access with proper locking

#### Orders (`mm/orders.py`)
- **OrdersClient**: Thin wrapper over py-clob-client
- **OrdersEngine**: Advanced order lifecycle management (planned)
- **Market Cancellation**: Bulk order cancellation utilities

#### Selection (`mm/selection.py`)
- **SelectionManager**: Google Sheets integration and change detection
- **Market Filtering**: Applies liquidity and volume thresholds
- **Real-time Updates**: Monitors sheet changes every 15 minutes
- **State Integration**: Shares StateStore instance with main daemon

### CLI Tools (`mm/cli.py`)

#### Export PnL
```bash
python -m mm.cli export_pnl --date 2025-09-03
```

#### Rebuild Positions
```bash
python -m mm.cli rebuild_positions
```

### Persistence & Metrics

#### Database Schema
- **orders**: Order placement history
- **fills**: Trade execution records
- **positions**: Current position state
- **orders_active**: Active order tracking
- **pnl_daily**: Daily PnL aggregation
- **markouts**: Fill performance tracking
- **selection_snapshots**: Market selection history

#### Prometheus Metrics
- **mm_ws_connected**: WebSocket connection status
- **mm_ws_reconnects_total**: Connection reconnection count
- **mm_seq_gaps_total**: Sequence gap detection per token
- **mm_snapshot_reload_total**: Snapshot reloads per token
- **mm_trades_per_min**: Trade frequency per token
- **mm_orders_total**: Order placement count by side
- **mm_fills_total**: Fill count by side
- **mm_inventory_shares**: Current inventory per token

### Tests

#### Market Data Tests
```bash
source .venv/bin/activate
PYTHONPATH=/Users/coenkuijpers/projects/poly-maker \
pytest -q tests/mm/test_market_data.py
```

#### Strategy Tests
```bash
source .venv/bin/activate
PYTHONPATH=/Users/coenkuijpers/projects/poly-maker \
pytest -q tests/mm/test_strategy.py
```

### Operational Notes

#### Market Selection
- The daemon only trades markets listed in "Selected Markets" sheet
- No runtime filtering - all selection happens at sheet level
- Token IDs are derived strictly from `token1` and `token2` columns
- Automatic monitoring every 15 minutes for sheet changes

#### Data Flow
1. **Startup**: REST backfill for initial order book snapshots
2. **Runtime**: WebSocket maintains live order book updates
3. **Gap Recovery**: Automatic snapshot reload on sequence gaps
4. **Position Sync**: Real-time position updates via Data API

#### Risk Controls
- **Soft Cap**: Warning threshold for position exposure
- **Hard Cap**: Maximum allowed position exposure
- **Daily Limits**: Configurable daily loss thresholds
- **Markout Penalties**: Dynamic spread adjustment based on performance

#### Logging
- **Console Output**: Real-time trading activity and system status
- **File Logging**: Comprehensive logs in `logs/` directory
- **Market Changes**: Clear visibility into selection updates
- **Error Handling**: Detailed exception logging with context

### Future Enhancements

#### Planned Features
- **Advanced Order Engine**: Sophisticated order lifecycle management
- **Portfolio Optimization**: Multi-market position balancing
- **Advanced Risk Models**: VaR and stress testing capabilities
- **Performance Analytics**: Detailed PnL attribution and analysis

#### Current Limitations
- **Single Strategy**: Only Avellaneda-lite currently implemented
- **Basic Order Management**: Limited order lifecycle features
- **Manual Configuration**: No automated parameter optimization
- **Single Exchange**: Polymarket CLOB only

### Troubleshooting

#### Common Issues
- **Port Conflicts**: Ensure port 9108 is available for Prometheus metrics
- **Sheet Access**: Verify Google Sheets API credentials and permissions
- **WebSocket Issues**: Check network connectivity to CLOB endpoints
- **Database Errors**: Verify write permissions for `mm_state.db`

#### Monitoring
- **Logs**: Check console output and log files for errors
- **Metrics**: Monitor Prometheus metrics at `http://localhost:9108`
- **Database**: Use SQLite tools to inspect `mm_state.db`
- **Sheet Changes**: Monitor logs for market selection updates

### License

MIT


