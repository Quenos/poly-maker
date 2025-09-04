#!/usr/bin/env python3
"""
Setup script to create the Settings worksheet in Google Sheets.
Run this script to migrate from environment variables to Google Sheets-based configuration.
"""

import logging
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe
from poly_utils.google_utils import get_spreadsheet

# Allow running as a script: add project root to sys.path for sibling package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_settings_sheet():
    """Create a Settings sheet in Google Sheets with all configuration parameters."""
    
    # Get the spreadsheet
    try:
        spreadsheet = get_spreadsheet(read_only=False)
        logger.info("Successfully connected to Google Sheets")
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {e}")
        logger.error("Please ensure SPREADSHEET_URL is set in your .env file")
        logger.error("and that you have proper Google Sheets credentials")
        return False
    
    # Define all settings with their descriptions and defaults
    settings_data = [
        # Market Selection Thresholds
        {
            'setting_name': 'MIN_LIQUIDITY',
            'value': '10000.0',
            'type': 'float',
            'category': 'Market Selection',
            'description': 'Minimum liquidity required for market selection (in USD)',
            'default': '10000.0'
        },
        {
            'setting_name': 'MIN_WEEKLY_VOLUME',
            'value': '50000.0',
            'type': 'float',
            'category': 'Market Selection',
            'description': 'Minimum weekly volume required for market selection (in USD)',
            'default': '50000.0'
        },
        {
            'setting_name': 'MIN_TREND',
            'value': '0.30',
            'type': 'float',
            'category': 'Market Selection',
            'description': 'Minimum trend score required for market selection',
            'default': '0.30'
        },
        {
            'setting_name': 'MM_SCORE_MIN',
            'value': '1000000.0',
            'type': 'float',
            'category': 'Market Selection',
            'description': 'Minimum market making score required (liquidity × √weekly_volume)',
            'default': '1000000.0'
        },
        
        # Avellaneda-Lite Quoting Strategy
        {
            'setting_name': 'K_VOL',
            'value': '2.0',
            'type': 'float',
            'category': 'Quoting Strategy',
            'description': 'Volatility parameter for spread calculation (higher = wider spreads)',
            'default': '2.0'
        },
        {
            'setting_name': 'K_FEE_TICKS',
            'value': '1.0',
            'type': 'float',
            'category': 'Quoting Strategy',
            'description': 'Fee parameter in ticks for spread calculation',
            'default': '1.0'
        },
        {
            'setting_name': 'ALPHA_FAIR',
            'value': '0.2',
            'type': 'float',
            'category': 'Quoting Strategy',
            'description': 'Fair price adjustment factor (higher = more aggressive pricing)',
            'default': '0.2'
        },
        {
            'setting_name': 'EWMA_VOL_WINDOW_SEC',
            'value': '600',
            'type': 'int',
            'category': 'Quoting Strategy',
            'description': 'Exponential weighted moving average window for volatility calculation (in seconds)',
            'default': '600'
        },
        {
            'setting_name': 'INV_GAMMA',
            'value': '1.0',
            'type': 'float',
            'category': 'Quoting Strategy',
            'description': 'Inventory gamma parameter (higher = more conservative inventory adjustment)',
            'default': '1.0'
        },
        
        # Risk Management
        {
            'setting_name': 'SOFT_CAP_DELTA_PCT',
            'value': '0.015',
            'type': 'float',
            'category': 'Risk Management',
            'description': 'Soft cap for delta exposure percentage (warning threshold)',
            'default': '0.015'
        },
        {
            'setting_name': 'HARD_CAP_DELTA_PCT',
            'value': '0.03',
            'type': 'float',
            'category': 'Risk Management',
            'description': 'Hard cap for delta exposure percentage (maximum allowed)',
            'default': '0.03'
        },
        {
            'setting_name': 'DAILY_LOSS_LIMIT_PCT',
            'value': '1.0',
            'type': 'float',
            'category': 'Risk Management',
            'description': 'Daily loss limit as percentage of capital',
            'default': '1.0'
        },
        
        # Order Management
        {
            'setting_name': 'ORDER_LAYERS',
            'value': '3',
            'type': 'int',
            'category': 'Order Management',
            'description': 'Number of order layers to place (more layers = better liquidity)',
            'default': '3'
        },
        {
            'setting_name': 'PRICE_TICK',
            'value': '0.01',
            'type': 'float',
            'category': 'Order Management',
            'description': 'Minimum price tick size for quoting and rounding',
            'default': '0.01'
        },
        {
            'setting_name': 'BASE_SIZE_USD',
            'value': '300.0',
            'type': 'float',
            'category': 'Order Management',
            'description': 'Base order size in USD',
            'default': '300.0'
        },
        {
            'setting_name': 'MAX_SIZE_USD',
            'value': '1500.0',
            'type': 'float',
            'category': 'Order Management',
            'description': 'Maximum order size in USD',
            'default': '1500.0'
        },
        {
            'setting_name': 'REQUOTE_MID_TICKS',
            'value': '1',
            'type': 'int',
            'category': 'Order Management',
            'description': 'Mid-price change threshold for requoting (in ticks)',
            'default': '1'
        },
        {
            'setting_name': 'REQUOTE_QUEUE_LEVELS',
            'value': '2',
            'type': 'int',
            'category': 'Order Management',
            'description': 'Number of queue levels to consider for requoting',
            'default': '2'
        },
        {
            'setting_name': 'ORDER_MAX_AGE_SEC',
            'value': '12',
            'type': 'int',
            'category': 'Order Management',
            'description': 'Maximum age of orders before replacement (in seconds)',
            'default': '12'
        },
        
        # Network Configuration
        {
            'setting_name': 'GAMMA_BASE_URL',
            'value': 'https://gamma-api.polymarket.com',
            'type': 'string',
            'category': 'Network',
            'description': 'Gamma API base URL for market data',
            'default': 'https://gamma-api.polymarket.com'
        },
        {
            'setting_name': 'CLOB_BASE_URL',
            'value': 'https://clob.polymarket.com',
            'type': 'string',
            'category': 'Network',
            'description': 'CLOB API base URL for order placement',
            'default': 'https://clob.polymarket.com'
        },
        {
            'setting_name': 'CLOB_WS_URL',
            'value': 'wss://ws-subscriptions-clob.polymarket.com/ws/',
            'type': 'string',
            'category': 'Network',
            'description': 'CLOB WebSocket URL for live market data',
            'default': 'wss://ws-subscriptions-clob.polymarket.com/ws/'
        },
        {
            'setting_name': 'POLYGON_RPC_URL',
            'value': 'https://polygon-rpc.com',
            'type': 'string',
            'category': 'Network',
            'description': 'Polygon RPC endpoint for blockchain interactions',
            'default': 'https://polygon-rpc.com'
        },
        
        # Sheet Configuration
        {
            'setting_name': 'SELECTED_SHEET_NAME',
            'value': 'Selected Markets',
            'type': 'string',
            'category': 'Sheet Configuration',
            'description': 'Name of the sheet containing selected markets to trade',
            'default': 'Selected Markets'
        },
        # Merger settings (inline with main structure)
        {
            'setting_name': 'MERGE_SCAN_INTERVAL_SEC',
            'value': '120',
            'type': 'int',
            'category': 'Merger',
            'description': 'Interval between merger scans (seconds)',
            'default': '120'
        },
        {
            'setting_name': 'MIN_MERGE_USDC',
            'value': '0.10',
            'type': 'float',
            'category': 'Merger',
            'description': 'Minimum overlap (USDC) required to trigger merging',
            'default': '0.10'
        },
        {
            'setting_name': 'MERGE_CHUNK_USDC',
            'value': '0.25',
            'type': 'float',
            'category': 'Merger',
            'description': 'Chunk size (USDC) per merger transaction',
            'default': '0.25'
        },
        {
            'setting_name': 'MERGE_MAX_RETRIES',
            'value': '3',
            'type': 'int',
            'category': 'Merger',
            'description': 'Maximum retries per market per scan cycle',
            'default': '3'
        },
        {
            'setting_name': 'MERGE_RETRY_BACKOFF_MS',
            'value': '500',
            'type': 'int',
            'category': 'Merger',
            'description': 'Backoff between retries (milliseconds)',
            'default': '500'
        },
        {
            'setting_name': 'MERGE_DRY_RUN',
            'value': 'false',
            'type': 'bool',
            'category': 'Merger',
            'description': 'If true, log planned merges without calling the helper',
            'default': 'false'
        },
        # Logging
        {
            'setting_name': 'LOG_LEVEL',
            'value': 'INFO',
            'type': 'string',
            'category': 'Logging',
            'description': 'Root logging level (DEBUG/INFO/WARNING/ERROR)',
            'default': 'INFO'
        },
        {
            'setting_name': 'LOG_FILE',
            'value': 'logs/mm_main.log',
            'type': 'string',
            'category': 'Logging',
            'description': 'Log file path',
            'default': 'logs/mm_main.log'
        },
        {
            'setting_name': 'LOG_ROTATION_BACKUPS',
            'value': '5',
            'type': 'int',
            'category': 'Logging',
            'description': 'Number of rotated log backups to keep',
            'default': '5'
        },
        # Loop intervals
        {
            'setting_name': 'SELECTION_LOOP_SEC',
            'value': '900',
            'type': 'int',
            'category': 'Loop Intervals',
            'description': 'Seconds between selection loop ticks',
            'default': '900'
        },
        {
            'setting_name': 'HEARTBEAT_SEC',
            'value': '5',
            'type': 'int',
            'category': 'Loop Intervals',
            'description': 'Seconds between heartbeat logs',
            'default': '5'
        },
        {
            'setting_name': 'BACKFILL_THROTTLE_SEC',
            'value': '10',
            'type': 'int',
            'category': 'Loop Intervals',
            'description': 'Seconds throttle for REST backfill per token',
            'default': '10'
        },
    ]
    
    # Convert to DataFrame
    df_all = pd.DataFrame(settings_data)
    # Reorder columns for better readability and sort by category then name
    df_all = df_all[['category', 'setting_name', 'value', 'type', 'default', 'description']]
    try:
        df_all = df_all.sort_values(by=['category', 'setting_name']).reset_index(drop=True)
    except Exception:
        pass
    
    try:
        # Check if Settings sheet already exists
        try:
            existing_sheet = spreadsheet.worksheet('Settings')
            logger.info("Settings sheet exists; will append only missing settings (no overwrite)")
            # Read existing settings
            records = existing_sheet.get_all_records()
            existing_names = set()
            if records:
                try:
                    existing_df = pd.DataFrame(records)
                    if 'setting_name' in existing_df.columns:
                        existing_names = set(existing_df['setting_name'].astype(str).tolist())
                except Exception:
                    existing_names = set()
            # Determine missing settings across the full defined list
            missing = [row for row in settings_data if row['setting_name'] not in existing_names]
            if not missing:
                logger.info("No new settings to add.")
            else:
                # Sort missing settings by category then name for consistency
                try:
                    missing = sorted(missing, key=lambda m: (m['category'], m['setting_name']))
                except Exception:
                    pass
                # Append rows in expected column order
                rows = [[m['category'], m['setting_name'], m['value'], m['type'], m['default'], m['description']] for m in missing]
                # Ensure header exists; append after last row
                existing_sheet.append_rows(rows, value_input_option='RAW')
                logger.info("Appended %d settings without overwriting existing values", len(rows))
        except Exception:
            # Create new sheet with all settings
            logger.info("Creating new Settings sheet with full defaults...")
            new_sheet = spreadsheet.add_worksheet(title='Settings', rows=len(df_all) + 1, cols=len(df_all.columns))
            set_with_dataframe(new_sheet, df_all, include_index=False)
        
        logger.info("Settings sheet ensured. Merger defaults added if missing.")
        
        # Print summary
        print("\n📋 Settings Sheet Created Successfully!")
        print(f"📊 Total settings (defined): {len(df_all)}")
        print("📁 Categories:")
        for category in df_all['category'].unique():
            count = len(df_all[df_all['category'] == category])
            print(f"   • {category}: {count} settings")
        print("\n🔧 Next steps:")
        print("   1. Review the Settings sheet in Google Sheets")
        print("   2. Modify any values as needed")
        print("   3. Update your .env file to remove old config variables")
        print("   4. Keep only SPREADSHEET_URL, PK, and BROWSER_ADDRESS in .env")
        print("   5. Restart the market making daemon")
        print("\n📖 See SETTINGS_MIGRATION.md for detailed migration instructions")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create/update Settings sheet: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Google Sheets-based configuration...")
    print("This will create a Settings worksheet with all configuration parameters.")
    print("Make sure you have SPREADSHEET_URL set in your .env file.\n")
    
    success = create_settings_sheet()
    
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup failed. Check the error messages above.")
        print("Common issues:")
        print("  - SPREADSHEET_URL not set in .env file")
        print("  - Missing or invalid Google Sheets credentials")
        print("  - Insufficient permissions on the Google Sheet")
