## Poly Maker Dashboard

A lightweight FastAPI app that serves a simple dashboard for open orders, positions, recent trades, and PnL.

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

- `BROWSER_ADDRESS` (or `BROWSER_WALLET`) – your wallet address (0x…)
- `PK` – private key used for API auth when fetching open orders (required for Orders)
- `SPREADSHEET_URL` – Google Sheet URL used for market metadata enrichment

Optional:
- `POLYGON_RPC_URL` – Polygon RPC endpoint (default: `https://polygon-rpc.com`)
- `LOG_LEVEL` – e.g. `INFO` or `DEBUG`

Notes:
- If `credentials.json` is present in the repo root (or one directory up), Google Sheets access will be authenticated. Without it, the app will try read‑only access via public CSV export.
- The PnL and Trades views require `BROWSER_ADDRESS`/`BROWSER_WALLET`. The Orders endpoint also requires `PK` to initialize the CLOB client.

### Start the server (dev)
Run from the project root so imports resolve correctly:
```bash
cd /Users/coenkuijpers/projects/poly-maker
source .venv/bin/activate
uvicorn web.dashboard:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### Endpoints
- `/` – Dashboard UI (orders, positions, trades, PnL pill)
- `/pnl` – PnL detail UI
- `/api/orders` – JSON of open orders (requires `PK` and `BROWSER_ADDRESS`)
- `/api/positions` – JSON of positions
- `/api/trades` – JSON of recent trades
- `/api/pnl` – JSON of current PnL totals and rows

### Troubleshooting
- 400 "BROWSER_WALLET/BROWSER_ADDRESS not configured": set `BROWSER_ADDRESS` (or `BROWSER_WALLET`) in `.env`.
- 500 when fetching Orders: ensure `PK` and `BROWSER_ADDRESS` are set; network access to `clob.polymarket.com` required.
- Missing market names/outcomes: ensure `SPREADSHEET_URL` is set; optionally add `credentials.json` for authenticated access.

### Production
```bash
uvicorn web.dashboard:app --host 0.0.0.0 --port 8000
```


