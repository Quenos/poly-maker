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
 - **On-chain Position Merging**: Automatically merges overlapping YES/NO positions based on configurable thresholds

### Prerequisites
- Python 3.10+
- Virtual environment (recommended) at project root: `.venv/`
- Google Sheets credentials/json and `SPREADSHEET_URL` env configured
- Polymarket API access (CLOB and Data API)

**Note**: The new Google Sheets-based configuration system requires the same Google Sheets setup as the existing system. If you're already using Google Sheets for market selection, no additional setup is needed.

### Installation
```bash
cd /Users/coenkuijpers/projects/poly-maker
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Configuration

The market making daemon now supports **two configuration methods**:

1. **Google Sheets-based Configuration** (Recommended) - All settings managed in a "Settings" worksheet
2. **Environment Variables** (Legacy) - Traditional .env file configuration

#### 🆕 Google Sheets Configuration (Recommended)

The new system reads most configuration parameters from a **Settings** worksheet in Google Sheets, providing:

- **Centralized Configuration**: All settings in one place
- **Easy Updates**: Modify settings without restarting the daemon
- **Team Collaboration**: Share configuration across team members
- **Better Documentation**: Each setting includes a description

**Setup**:
```bash
# Create the Settings worksheet
python -m mm.setup_settings_sheet

# Test configuration loading
python -c "from mm.sheet_config import load_config; config = load_config(); print('Success!')"
```

**Required .env variables** (only secrets):
```bash
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id
PK=0x1234567890abcdef...  # Private key (secret)
BROWSER_ADDRESS=0x1234567890abcdef...  # Wallet address (secret)
```

**All other settings** are managed in the Google Sheets "Settings" worksheet.

📖 **See `SETTINGS_MIGRATION.md` for detailed migration instructions.**

#### 🔧 Environment Variables (Legacy)

The traditional method using environment variables. All parameters have sensible defaults but can be customized for your trading strategy.

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
Test mode performs a dry run with no actual orders sent and logs to both console and `logs/mm_main_<timestamp>.log`.

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
- **OrdersEngine**: Advanced order lifecycle management with diffing and layered quotes
- **Market Cancellation**: Bulk order cancellation utilities

#### Selection (`mm/selection.py`)
- **SelectionManager**: Google Sheets integration and change detection
- **Market Filtering**: Applies liquidity and volume thresholds
- **Real-time Updates**: Monitors sheet changes every 15 minutes
- **State Integration**: Shares StateStore instance with main daemon

#### Configuration (`mm/sheet_config.py`)
- **Google Sheets Integration**: Reads configuration from Settings worksheet
- **Fallback Support**: Environment variables and hard-coded defaults
- **Type Safety**: Automatic type conversion and validation
- **Backward Compatibility**: Maintains existing config.py interface

#### Setup Tools (`mm/setup_settings_sheet.py`)
- **Automated Setup**: Creates Settings worksheet with all parameters
- **Comprehensive Documentation**: Includes descriptions for each setting
- **Default Values**: Pre-populates with recommended configurations
- **Error Handling**: Robust connection and permission checking

### CLI Tools (`mm/cli.py`)

The market making daemon includes several command-line utilities for data analysis and maintenance operations.

### Utility Functions (`poly_utils`)

The project includes shared utility functions in the `poly_utils` module that can be used by various parts of the system.

### Close Positions (`poly_utils.close_positions`)

A utility to flatten positions on Polymarket via **LIMIT** orders using a single `{token_id: price}` map.

- **Explicit price** (`0.0 < price < 1.0`): places a limit order at that price.  
- **`None` as price**: submits an aggressive *marketable* limit  
  - `SELL @ 0.01` to close **longs**  
  - `BUY  @ 0.99` to close **shorts**  
- **No dict at all** (`None`): closes **all** non-zero positions with aggressive defaults.

**Features**
- Close specific positions at chosen prices
- Close all open positions with aggressive defaults
- Unified dict input: `{token_id: price}` where `price=None` means marketable
- Programmatic API and CLI interface

**Usage**
```python
from poly_utils import close_positions

# Close all positions aggressively (no dict)
close_positions()

# Close specific tokens at explicit limits
close_positions({"0xabc...": 0.37, "0xdef...": 0.62})

# Mix explicit price and marketable (None)
close_positions({"0xabc...": None, "0xdef...": 0.58})

# Close all positions aggressively
python poly_utils/close_positions.py

# Close specific tokens at explicit limits
python poly_utils/close_positions.py 0xabc...=0.37 0xdef...=0.62

# Mix explicit price and marketable (None)
python poly_utils/close_positions.py 0xabc...=None 0xdef...=0.58
```

#### Export PnL

python -m mm.cli export_pnl --date 2025-09-03
```
**What it does**: Exports daily PnL data from the SQLite database to CSV format. This tool aggregates realized and unrealized PnL, fees, and rebates for a specific date, providing detailed performance analysis for reporting and analysis purposes.

**Output**: CSV file with columns for date, realized PnL, unrealized PnL, fees, and rebates.

#### Rebuild Positions
```bash
python -m mm.cli rebuild_positions
```
**What it does**: Reconstructs the current position state by analyzing the complete fill history in the database. This is useful when the position tracking gets out of sync or when you need to verify the current state against historical data.

**Output**: Updates the positions table in the database and logs the reconstructed position counts.

#### Setup Settings Sheet
```bash
python -m mm.setup_settings_sheet
```
**What it does**: Creates or updates the Settings worksheet in Google Sheets with all configuration parameters. This is the first step in migrating from environment variables to Google Sheets-based configuration.

**Output**: Creates a "Settings" worksheet with 25+ configuration parameters organized by category, including descriptions and default values.

### Market Sheet Creation Tools

The market making system relies on Google Sheets to define which markets to trade. Several tools help create and maintain these sheets with filtered and ranked markets.

#### 1. Market Size Sheet Creation (`update_market_size.py`)

This tool fetches market data from the Gamma API and creates two essential sheets with intelligent market protection:

**Market Size Sheet**: Contains comprehensive market data including:
- Market questions and descriptions
- Token IDs (token1, token2, condition_id)
- Liquidity metrics and volume data
- Trend scores and market making scores
- Best bid/ask prices

**Filtered Markets Sheet**: Uses AI-powered filtering with **Grandfather Clause Protection**:
- Applies liquidity and volume thresholds
- Uses OpenAI to evaluate market suitability
- Removes markets labeled as "AVOID" by AI analysis
- **🛡️ Protects currently traded markets** that still meet criteria
- **🗑️ Removes currently traded markets** that no longer meet criteria
- Preserves only markets suitable for market making

**🛡️ Grandfather Clause Protection**:
The tool automatically protects markets currently in "Selected Markets" from being removed by AI filtering, **EXCEPT** when they no longer meet the minimum market size criteria:

**Protection Criteria** (markets must meet ALL):
- **Liquidity**: Minimum $1,000,000
- **Weekly Volume**: Minimum $50,000  
- **Market Making Score**: Minimum 1,000,000 (liquidity × √weekly_volume)
- **Bid Price Range**: Between 0.15 and 0.85

**Protection Logic**:
```
Currently Traded Market Filtered Out by AI?
├─ Yes → Check if still meets market size criteria
│   ├─ Meets criteria → 🛡️ PROTECT (add back to filtered results)
│   └─ Fails criteria → 🗑️ ALLOW REMOVAL (don't protect)
└─ No → Already passed AI filtering (no action needed)
```

**Usage**:
```bash
# Create both Market Size and Filtered Markets sheets with protection
python update_market_size.py

# Create only Market Size sheet (no AI filtering, no protection needed)
python update_market_size.py --all
```

**What it does**:
1. Fetches all markets from Gamma API
2. Calculates liquidity, volume, and trend metrics
3. Applies configurable filtering criteria
4. Uses AI to evaluate market suitability
5. **Applies Grandfather Clause protection** for currently traded markets
6. **Removes markets that fail criteria** even if currently traded
7. Writes results to Google Sheets with comprehensive logging

**Benefits**:
- **Zero Disruption**: Quality markets are never disrupted by AI filtering
- **Quality Control**: Poor markets are automatically removed
- **Transparency**: Complete visibility into protection decisions
- **Automatic**: No manual intervention required
- **Consistent**: Uses same criteria as original market filtering

#### 2. Market Filtering (`ai/filter_markets.py`)

Advanced AI-powered tool that evaluates markets for trading suitability:

**Features**:
- **AI Evaluation**: Uses OpenAI GPT models to analyze market questions
- **Content Filtering**: Removes markets with inappropriate content
- **Risk Assessment**: Identifies markets that may be unsuitable for trading
- **Batch Processing**: Handles large numbers of markets efficiently
- **Configurable Models**: Supports different OpenAI models for filtering

**Usage**:
```bash
python ai/filter_markets.py --stage filter --model gpt-4o-mini
```

**What it does**:
1. Fetches all available markets from Polymarket
2. Sends market questions to OpenAI for evaluation
3. Uses structured prompts to classify markets as ELIGIBLE or AVOID
4. Returns filtered DataFrame with only suitable markets
5. Supports parallel processing for large datasets

#### 3. Market Data Updates (`update_markets.py`)

Comprehensive tool for updating market data and calculating trading metrics:

**Features**:
- **Real-time Data**: Fetches current order book and market data
- **Volatility Calculation**: Computes short and long-term volatility metrics
- **Reward Analysis**: Calculates maker rewards and trading opportunities
- **Composite Scoring**: Ranks markets by multiple factors
- **Sheet Synchronization**: Updates multiple Google Sheets automatically

**Usage**:
```bash
python update_markets.py
```

**What it does**:
1. Fetches current market data from Polymarket CLOB
2. Calculates volatility across multiple timeframes
3. Computes maker rewards and trading scores
4. Sorts markets by composite scoring
5. Updates All Markets, Volatility Markets, and Full Markets sheets

#### 4. Selected Markets Management (`store_selected_markets.py`)

Tool for managing the final "Selected Markets" sheet that the daemon actually trades:

**Features**:
- **Sheet Synchronization**: Syncs selected markets across multiple sheets
- **Data Validation**: Ensures token IDs and condition IDs are properly formatted
- **Column Preservation**: Maintains consistent data structure across sheets

**Usage**:
```bash
python store_selected_markets.py
```

**What it does**:
1. Reads the "Selected Markets" sheet
2. Finds corresponding data in "All Markets" sheet
3. Creates "Stored Sel Markets" with complete market information
4. Ensures all required columns are present and properly formatted

### Creating Your Market Selection Workflow

To set up a complete market selection system:

#### Step 1: Initial Setup
```bash
# 1. Create Market Size sheet with all available markets
python update_market_size.py --all

# 2. Review the Market Size sheet and identify potential markets
# Look for markets with good liquidity, volume, and clear questions
```

#### Step 2: AI Filtering
```bash
# 3. Apply AI filtering to remove unsuitable markets
python update_market_size.py
# This creates the Filtered Markets sheet with AI-approved markets
```

#### Step 3: Manual Selection
```bash
# 4. Manually review Filtered Markets and copy desired markets to Selected Markets
# Use Google Sheets to copy rows from "Filtered Markets" to "Selected Markets"
# Ensure columns: question, token1, token2, condition_id are present
```

#### Step 4: Daemon Integration
```bash
# 5. The market making daemon will automatically read from "Selected Markets"
# It monitors this sheet every 15 minutes for changes
```

#### Step 5: Ongoing Maintenance
```bash
# 6. Regularly update market data and re-run filtering
python update_markets.py          # Update market metrics
python update_market_size.py      # Refresh AI filtering
```

### Sheet Structure Requirements

#### Selected Markets Sheet (Required)
The daemon reads from this sheet and requires these columns:
- **token1**: First token ID (required)
- **token2**: Second token ID (required)  
- **condition_id**: Market condition ID (recommended)
- **question/market**: Market question text (optional, for reference)

#### Market Size Sheet (Generated)
Contains comprehensive market data for analysis:
- Market metadata and descriptions
- Liquidity and volume metrics
- Trend scores and market making scores
- Current bid/ask prices

#### Filtered Markets Sheet (Generated)
AI-filtered subset of markets:
- Only markets passing AI suitability checks
- Preserves all Market Size columns
- Ready for manual review and selection

### Configuration for Market Tools

#### Environment Variables
```bash
# Required for Google Sheets access
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id

# Required for AI filtering
OPENAI_API_KEY=your-openai-api-key

# Required for market data access
POLYGON_RPC_URL=https://polygon-rpc.com
```

#### Google Sheets Setup
1. Create a Google Sheet with the URL from `SPREADSHEET_URL`
2. Ensure the service account has edit access
3. The tools will automatically create required worksheets
4. Recommended sheet names: "Market Size", "Filtered Markets", "Selected Markets"

### Best Practices

#### Market Selection Criteria
- **Liquidity**: Minimum $10,000+ for reliable pricing
- **Volume**: Weekly volume > $50,000 for active trading
- **Question Quality**: Clear, unambiguous questions
- **Content**: Appropriate content that passes AI filtering
- **Token Availability**: Both YES/NO tokens must be available

#### Maintenance Schedule
- **Daily**: Run `update_markets.py` for current data
- **Weekly**: Run `update_market_size.py` for AI re-filtering
- **As Needed**: Manually review and update Selected Markets
- **Monitoring**: Check daemon logs for market selection changes

#### Troubleshooting
- **Missing Markets**: Ensure condition_id columns are populated
- **AI Filtering Failures**: Check OpenAI API key and rate limits
- **Sheet Sync Issues**: Verify Google Sheets permissions
- **Data Quality**: Ensure token IDs are properly formatted as strings

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

#### Configuration Reloads
- Sending `SIGHUP` reloads configuration for in-process components but does not rebuild the merger loop. If you change merger settings, restart the daemon to apply them.

### Position Merging (On-chain)

The daemon runs a background merger that scans for overlapping YES/NO positions and merges them on-chain to free USDC collateral.

**How it works**
- Reads markets from the "Selected Markets" sheet and uses `token1`, `token2`, and `condition_id`.
- Fetches wallet positions using the configured `BROWSER_ADDRESS`.
- Computes overlap as `min(shares(token1), shares(token2))`.
- If overlap ≥ `MIN_MERGE_USDC`, initiates merging in chunks of `MERGE_CHUNK_USDC` until the overlap is exhausted or a retry/cooldown stops the cycle.
- For markets flagged `neg_risk` in the sheet, uses the Negative Risk Adapter; otherwise, uses the Conditional Tokens `mergePositions`.

**Merger settings** (in the Settings sheet)
- `MERGE_SCAN_INTERVAL_SEC` (int): Interval between scans (default 120)
- `MIN_MERGE_USDC` (float): Minimum overlap in USDC (6dp shares) to trigger merging (default 0.10)
- `MERGE_CHUNK_USDC` (float): Per-transaction merge chunk size (default 0.25)
- `MERGE_MAX_RETRIES` (int): Max retries per market per scan (default 3)
- `MERGE_RETRY_BACKOFF_MS` (int): Backoff between retries in milliseconds (default 500)
- `MERGE_DRY_RUN` (bool): If true, log planned merges without sending on-chain transactions (default false)

Notes
- Overlap and thresholds are compared in "share" terms where 1.00 share equals 1 USDC (6 decimals on-chain). For example, 0.10 YES and 0.10 NO will merge.
- The merger uses the configured `BROWSER_ADDRESS` (from .env via `sheet_config`) for position lookups and on-chain actions.
- After changing merger settings, restart the daemon to apply changes to the merger loop.

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
- **Database Errors**: Verify write permissions for `data/mm_state.db`

#### Monitoring
- **Logs**: Check console output and log files for errors
- **Metrics**: Monitor Prometheus metrics at `http://localhost:9108`
- **Database**: Use SQLite tools to inspect `data/mm_state.db`
- **Sheet Changes**: Monitor logs for market selection updates

### License

MIT


