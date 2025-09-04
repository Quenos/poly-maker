## Poly Maker Dashboard

A lightweight FastAPI app that serves a simple dashboard for open orders, positions, recent trades, and PnL. The dashboard now includes **position closing functionality** allowing traders to close positions at market or at specific limit prices directly from the web interface.

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
cd /Users/coenkuijpers/projects/poly-maker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file in the project root (`/Users/coenkuijpers/projects/poly-maker/.env`) with at least:

**Required:**
- `BROWSER_ADDRESS` (or `BROWSER_WALLET`) – your wallet address (0x…)
- `PK` – private key used for API auth when fetching open orders (required for Orders)

**Optional:**
- `POLYGON_RPC_URL` – Polygon RPC endpoint (default: `https://polygon-rpc.com`)
- `LOG_LEVEL` – e.g. `INFO` or `DEBUG`
- `SESSION_SECRET` – Secret key for session management (default: "CHANGE_ME")
- `GITHUB_CLIENT_ID` – GitHub OAuth client ID for authentication
- `GITHUB_CLIENT_SECRET` – GitHub OAuth client secret
- `OAUTH_REDIRECT_URI` – OAuth redirect URI
- `ALLOWED_GH_IDS` – Comma-separated list of allowed GitHub user IDs
- `ODDS_DB_PATH` – Path to odds SQLite database (default: `data/odds.db`)

**Notes:**
- The PnL and Trades views require `BROWSER_ADDRESS`/`BROWSER_WALLET`
- The Orders endpoint requires `PK` to initialize the CLOB client
- GitHub OAuth is required for accessing the dashboard
- Market metadata is fetched from public APIs (Gamma API, Polymarket APIs)

### Start the server (dev)
Run from the project root so imports resolve correctly:
```bash
cd /Users/coenkuijpers/projects/poly-maker
source .venv/bin/activate
uvicorn web.dashboard:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### Dashboard Features

#### Core Dashboard (`/`)
- **Open Orders**: View and monitor open orders
- **Positions**: View current positions with **position closing controls**
- **Recent Trades**: Browse recent trading activity with pagination
- **PnL Summary**: Real-time profit/loss overview

#### PnL Detail Page (`/pnl`)
- Detailed PnL breakdown by market
- Realized vs unrealized PnL
- Earnings from rewards
- Per-trade cash flow analysis

#### Odds Analysis (`/odds`)
- Model predictions vs actual market prices
- Hourly price snapshots
- Market categorization and confidence metrics

#### Position Closing Functionality
The dashboard includes comprehensive position closing capabilities:

- **Individual Position Controls**: Each position row includes:
  - Price input field (0.01-0.99 range)
  - "Close at Limit" button (submits limit order at specified price)
  - "Close at Market" button (submits aggressive marketable limit order)

- **Bulk Position Closing**: 
  - "Close All Positions" button at the top of positions section
  - Confirmation dialog for safety
  - Uses aggressive market defaults (0.01 for longs, 0.99 for shorts)

- **Smart Price Handling**:
  - **Explicit price** (0.0 < price < 1.0): Places limit order at that price
  - **Market (None)**: Submits aggressive marketable limit orders
  - **Validation**: Frontend and backend price validation for security

### API Endpoints

#### Core Dashboard Endpoints
- `/` – Main dashboard UI (orders, positions, trades, PnL pill)
- `/pnl` – PnL detail page
- `/odds` – Odds analysis page
- `/login` – GitHub OAuth login
- `/logout` – User logout

#### Data API Endpoints
- `GET /api/orders` – Open orders (requires `PK` and `BROWSER_ADDRESS`)
- `GET /api/positions` – Current positions with asset IDs for closing
- `GET /api/positions/debug` – Debug endpoint for position data and metadata
- `GET /api/trades` – Recent trades with pagination
- `GET /api/pnl` – Current PnL totals and breakdown by market
- `GET /api/pnl/trades` – Per-trade PnL cash flows
- `GET /api/odds` – Model predictions and market odds data

#### Position Management Endpoints
- `POST /api/positions/close` – Close positions using `poly_utils.close_positions`

**Request Body:**
```json
{
  "token_id": "0xabc...",  // Optional: specific token to close
  "price": 0.45            // Optional: limit price (0.01-0.99) or null for market
}
```

**Examples:**
- Close specific position at limit: `{"token_id": "0xabc...", "price": 0.45}`
- Close specific position at market: `{"token_id": "0xabc...", "price": null}`
- Close all positions: `{}` (empty body)

**Response:**
```json
{
  "success": true,
  "closed_count": 1,
  "message": "Successfully submitted close orders for 1 positions"
}
```

### Data Sources & Metadata

#### Market Data Enrichment
The dashboard automatically enriches position and trade data with market metadata:

- **Gamma API**: Primary source for token metadata and market information
- **Polymarket APIs**: Fallback for market data and condition information
- **Real-time Updates**: Dashboard refreshes every 5 seconds
- **Smart Fallbacks**: Multiple API sources ensure reliable data

#### Position Metadata
- Market names and questions
- Outcome descriptions
- Condition IDs and market IDs
- Automatic mapping from token IDs to readable information

### Position Closing Logic

The dashboard integrates with the existing `poly_utils.close_positions` utility:

- **Long positions** (positive size): Closed with SELL orders
- **Short positions** (negative size): Closed with BUY orders
- **Market orders**: Use aggressive defaults (0.01 for SELL, 0.99 for BUY)
- **Limit orders**: Use specified price within valid range
- **Real-time updates**: Dashboard refreshes after closing operations
- **Error handling**: Comprehensive error messages and validation

### Authentication & Security

#### GitHub OAuth
- Required for all dashboard access
- Configurable allowed user IDs
- Session-based authentication
- Secure cookie handling

#### Security Features
- Position closing requires user authentication
- All close operations are logged for audit purposes
- Price validation prevents invalid limit orders
- Confirmation dialogs prevent accidental bulk operations
- CSRF protection via session middleware

### Troubleshooting

#### Common Issues
- **400 "BROWSER_WALLET/BROWSER_ADDRESS not configured"**: Set `BROWSER_ADDRESS` in `.env`
- **500 when fetching Orders**: Ensure `PK` and `BROWSER_ADDRESS` are set; check network access to `clob.polymarket.com`
- **Missing market names**: Market metadata is fetched automatically from APIs
- **Position closing errors**: Verify `poly_utils.close_positions` is available and wallet has sufficient permissions
- **Authentication issues**: Check GitHub OAuth configuration and allowed user IDs

#### Debug Endpoints
- Use `/api/positions/debug` to inspect raw position data and metadata fetching
- Check browser console for JavaScript errors
- Monitor server logs for backend issues

### Development

#### Local Development
```bash
# Start with auto-reload
uvicorn web.dashboard:app --reload --host 127.0.0.1 --port 8000

# Start with specific port
uvicorn web.dashboard:app --reload --host 127.0.0.1 --port 8080

# Start with debug logging
LOG_LEVEL=DEBUG uvicorn web.dashboard:app --reload
```

#### Code Structure
- `web/dashboard.py` – Main FastAPI application
- `web/static/` – Frontend assets (HTML, CSS, JavaScript)
- `web/static/index.html` – Main dashboard interface
- `web/static/styles.css` – Dashboard styling

### Production Deployment

#### Basic Production
```bash
uvicorn web.dashboard:app --host 0.0.0.0 --port 8000
```

#### Production Considerations
- Set `SESSION_SECRET` to a secure random string
- Configure proper `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`
- Use reverse proxy (nginx) for SSL termination
- Set appropriate `LOG_LEVEL` for production
- Monitor server resources and logs
- Consider using Gunicorn with multiple workers

### API Documentation
- **Swagger UI**: Available at `/docs` when running
- **ReDoc**: Alternative documentation at `/redoc`
- **OpenAPI Schema**: Raw schema at `/openapi.json`


