# Settings Migration Guide

This guide explains how to migrate from environment variable-based configuration to the new Google Sheets-based configuration system.

## Overview

The market making daemon now reads most configuration parameters from a **Settings** worksheet in Google Sheets instead of environment variables. This provides several benefits:

- **Centralized Configuration**: All settings in one place
- **Easy Updates**: Modify settings without restarting the daemon
- **Team Collaboration**: Share configuration across team members
- **Version Control**: Track configuration changes over time
- **Better Documentation**: Each setting includes a description

## What Changed

### Before (Environment Variables)
```bash
# .env file
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id
MIN_LIQUIDITY=10000.0
K_VOL=2.0
ALPHA_FAIR=0.2
ORDER_LAYERS=3
# ... many more environment variables
```

### After (Google Sheets + Minimal .env)
```bash
# .env file (only secrets)
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id
PK=0x1234567890abcdef...  # Private key (secret)
BROWSER_ADDRESS=0x1234567890abcdef...  # Wallet address (secret)
```

All other settings are now managed in the **Settings** worksheet in Google Sheets.

## Migration Steps

### Step 1: Create the Settings Sheet

Run the provided script to create the Settings worksheet:

```bash
python -m mm.setup_settings_sheet
```

This will:
- Connect to your Google Sheets
- Create a new "Settings" worksheet
- Populate it with all configuration parameters
- Include descriptions and default values

### Step 2: Review and Customize

1. Open your Google Sheet
2. Navigate to the "Settings" worksheet
3. Review all settings and their descriptions
4. Modify any values as needed for your strategy
5. The system will automatically use these new values

### Step 3: Test the New Configuration

Verify that the new system works:

```bash
python -c "from mm.sheet_config import load_config; config = load_config(); print('Configuration loaded successfully')"
```

This will test the connection and configuration loading.

### Step 4: Update Your .env File

Remove all the old configuration variables, keeping only:

```bash
# Required
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id

# Secrets (keep these)
PK=0x1234567890abcdef...
BROWSER_ADDRESS=0x1234567890abcdef...

# Optional (if you want to override sheet values)
# MIN_LIQUIDITY=5000.0  # This will override the sheet value
```

### Step 5: Restart the Daemon

The market making daemon will now automatically read from the Settings sheet:

```bash
python -m mm.main
```

## Settings Categories

The Settings worksheet organizes parameters into logical categories:

### Market Selection
- **MIN_LIQUIDITY**: Minimum liquidity required for market selection
- **MIN_WEEKLY_VOLUME**: Minimum weekly volume required
- **MIN_TREND**: Minimum trend score required
- **MM_SCORE_MIN**: Minimum market making score required

### Quoting Strategy
- **K_VOL**: Volatility parameter for spread calculation
- **K_FEE_TICKS**: Fee parameter in ticks
- **ALPHA_FAIR**: Fair price adjustment factor
- **EWMA_VOL_WINDOW_SEC**: Volatility calculation window
- **INV_GAMMA**: Inventory gamma parameter

### Risk Management
- **SOFT_CAP_DELTA_PCT**: Soft cap for delta exposure
- **HARD_CAP_DELTA_PCT**: Hard cap for delta exposure
- **DAILY_LOSS_LIMIT_PCT**: Daily loss limit percentage

### Order Management
- **ORDER_LAYERS**: Number of order layers to place
- **BASE_SIZE_USD**: Base order size in USD
- **MAX_SIZE_USD**: Maximum order size in USD
- **REQUOTE_MID_TICKS**: Mid-price change threshold for requoting
- **REQUOTE_QUEUE_LEVELS**: Queue levels to consider for requoting
- **ORDER_MAX_AGE_SEC**: Maximum age of orders before replacement

### Network Configuration
- **GAMMA_BASE_URL**: Gamma API base URL
- **CLOB_BASE_URL**: CLOB API base URL
- **CLOB_WS_URL**: CLOB WebSocket URL
- **POLYGON_RPC_URL**: Polygon RPC endpoint

### Sheet Configuration
- **SELECTED_SHEET_NAME**: Name of the sheet containing selected markets

## Fallback Behavior

The system provides robust fallback behavior:

1. **Google Sheets Settings** (primary source)
2. **Environment Variables** (fallback)
3. **Hard-coded Defaults** (final fallback)

This means:
- If a setting exists in the sheet, it's used
- If not in the sheet but in .env, the .env value is used
- If neither exists, the hard-coded default is used

## Updating Settings

### Method 1: Google Sheets (Recommended)
1. Open the Settings worksheet
2. Modify the value in the "value" column
3. Save the sheet
4. The daemon will automatically pick up changes on the next configuration reload

### Method 2: Environment Variables (Override)
You can still override sheet values using environment variables:

```bash
# This will override the sheet value for K_VOL
export K_VOL=3.0
python -m mm.main
```

## Troubleshooting

### Common Issues

#### 1. "Settings sheet not found"
- Run `python -m mm.setup_settings_sheet` first
- Verify the sheet name is exactly "Settings"

#### 2. "Failed to connect to Google Sheets"
- Check your `SPREADSHEET_URL` in .env
- Verify `credentials.json` exists and has proper permissions
- Ensure the service account has access to the sheet

#### 3. "Configuration values not updating"
- Check that the Settings worksheet has the correct column names
- Verify the "setting_name" column matches the expected names exactly
- Check the "type" column for proper data types (float, int, string)

#### 4. "Type conversion errors"
- Ensure numeric settings have numeric values in the sheet
- Check that the "type" column is correctly set
- Verify no extra spaces or characters in values

### Debug Mode

Enable debug logging to see detailed configuration loading:

```bash
export LOG_LEVEL=DEBUG
python -c "from mm.sheet_config import load_config; config = load_config()"
```

## Benefits of the New System

### For Developers
- **Centralized Configuration**: All settings in one place
- **Type Safety**: Automatic type conversion and validation
- **Fallback Support**: Robust error handling and defaults
- **Easy Testing**: Modify settings without code changes

### For Operators
- **Real-time Updates**: Change settings without restarting
- **Visual Interface**: Easy to see and modify all settings
- **Documentation**: Each setting includes a description
- **Team Access**: Share configuration across team members

### For Maintenance
- **Version Control**: Track configuration changes over time
- **Audit Trail**: See who changed what and when
- **Backup**: Google Sheets provides automatic backup
- **Rollback**: Easy to revert to previous configurations

## Migration Checklist

- [ ] Run `python -m mm.setup_settings_sheet`
- [ ] Review and customize settings in Google Sheets
- [ ] Test configuration loading
- [ ] Update .env file (remove old config variables)
- [ ] Restart the market making daemon
- [ ] Verify all settings are working correctly
- [ ] Remove old environment variables from deployment scripts

## Support

If you encounter issues during migration:

1. Check the troubleshooting section above
2. Review the logs for detailed error messages
3. Verify Google Sheets permissions and credentials
4. Ensure the Settings worksheet structure is correct

The new system maintains full backward compatibility, so you can always fall back to environment variables if needed.
