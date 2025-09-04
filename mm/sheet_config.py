"""
Configuration module that reads settings from Google Sheets Settings sheet.
This replaces the need for most environment variables.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv

from poly_utils.google_utils import get_spreadsheet

# Load environment variables for secrets
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MMConfig:
    """Configuration values for the market making daemon, loaded from Google Sheets with defaults."""

    # Sheet and project (still from env as these are secrets)
    spreadsheet_url: str
    selected_sheet_name: str = "Selected Markets"

    # Thresholds
    min_liquidity: float = 10000.0
    min_weekly_volume: float = 50000.0
    min_trend: float = 0.30
    mm_score_min: float = 1_000_000.0

    # Avellaneda-lite and quoting
    k_vol: float = 2.0
    k_fee_ticks: float = 1.0
    alpha_fair: float = 0.2
    ewma_vol_window_sec: int = 600
    inv_gamma: float = 1.0
    soft_cap_delta_pct: float = 0.015
    hard_cap_delta_pct: float = 0.03
    order_layers: int = 3
    base_size_usd: float = 300.0
    max_size_usd: float = 1500.0
    price_tick: float = 0.01
    requote_mid_ticks: int = 1
    requote_queue_levels: int = 2
    order_max_age_sec: int = 12
    daily_loss_limit_pct: float = 1.0
    # Non-retryable error backoff
    nonretryable_cooldown_sec: int = 60
    # Merger settings
    merge_scan_interval_sec: int = 120
    min_merge_usdc: float = 0.10
    merge_chunk_usdc: float = 0.25
    merge_max_retries: int = 3
    merge_retry_backoff_ms: int = 500
    merge_dry_run: bool = False

    # Networking
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    # Auth (still from env as these are secrets)
    pk: Optional[str] = None
    browser_address: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/mm_main.log"
    log_rotation_backups: int = 5

    # Loop intervals
    selection_loop_sec: int = 900
    heartbeat_sec: int = 5
    backfill_throttle_sec: int = 10


def _parse_setting_value(value: str, setting_type: str) -> Union[float, int, str]:
    """Parse a setting value based on its type."""
    try:
        if setting_type == 'float':
            return float(value)
        elif setting_type == 'int':
            return int(value)
        else:  # string
            return value
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse setting value '{value}' as {setting_type}: {e}")
        return value


def _get_sheet_settings() -> Dict[str, Any]:
    """Read settings from the Google Sheets Settings worksheet."""
    settings = {}
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(read_only=False)
        
        # Read the Settings sheet
        try:
            settings_sheet = spreadsheet.worksheet('Settings')
            records = settings_sheet.get_all_records()
            
            for record in records:
                setting_name = record.get('setting_name')
                value = record.get('value')
                setting_type = record.get('type', 'string')
                
                if setting_name and value is not None:
                    parsed_value = _parse_setting_value(value, setting_type)
                    settings[setting_name] = parsed_value
                    logger.debug(f"Loaded setting: {setting_name} = {parsed_value} ({setting_type})")
            
            logger.info(f"Successfully loaded {len(settings)} settings from Google Sheets")
            
        except Exception as e:
            logger.warning(f"Failed to read Settings sheet: {e}. Using defaults.")
            return {}
            
    except Exception as e:
        logger.warning(f"Failed to connect to Google Sheets: {e}. Using defaults.")
        return {}
    
    return settings


def _get_float(name: str, default: float, sheet_settings: Dict[str, Any]) -> float:
    """Get a float setting value, first from sheet, then from env, then default."""
    # Try sheet first
    if name in sheet_settings:
        value = sheet_settings[name]
        if isinstance(value, (int, float)):
            return float(value)
        logger.warning(f"Sheet setting {name} is not numeric: {value}, using default")
    
    # Fallback to environment variable
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def _get_int(name: str, default: int, sheet_settings: Dict[str, Any]) -> int:
    """Get an int setting value, first from sheet, then from env, then default."""
    # Try sheet first
    if name in sheet_settings:
        value = sheet_settings[name]
        if isinstance(value, (int, float)):
            return int(value)
        logger.warning(f"Sheet setting {name} is not numeric: {value}, using default")
    
    # Fallback to environment variable
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


def _get_string(name: str, default: str, sheet_settings: Dict[str, Any]) -> str:
    """Get a string setting value, first from sheet, then from env, then default."""
    # Try sheet first
    if name in sheet_settings:
        value = sheet_settings[name]
        if isinstance(value, str):
            return value
        logger.warning(f"Sheet setting {name} is not a string: {value}, using default")
    
    # Fallback to environment variable
    return os.getenv(name, default)


def _get_bool(name: str, default: bool, sheet_settings: Dict[str, Any]) -> bool:
    # Try sheet first
    if name in sheet_settings:
        v = sheet_settings[name]
        if isinstance(v, bool):
            return v
        try:
            sv = str(v).strip().lower()
            return sv in ("1", "true", "yes", "y", "t")
        except Exception:
            pass
    # Fallback to environment variable
    envv = os.getenv(name)
    if envv is None:
        return default
    try:
        return str(envv).strip().lower() in ("1", "true", "yes", "y", "t")
    except Exception:
        return default


def load_config() -> MMConfig:
    """Load configuration from Google Sheets Settings worksheet with fallbacks to env and defaults.

    Required env (secrets):
      - SPREADSHEET_URL
      - PK (optional, for CLOB signing)
      - BROWSER_ADDRESS (optional, wallet address)
    
    All other settings are read from the Settings worksheet in Google Sheets.
    """
    spreadsheet_url = os.getenv("SPREADSHEET_URL")
    if not spreadsheet_url:
        raise ValueError("SPREADSHEET_URL is required")
    
    # Load settings from Google Sheets
    sheet_settings = _get_sheet_settings()
    
    return MMConfig(
        # Sheet and project (from env)
        spreadsheet_url=spreadsheet_url,
        selected_sheet_name=_get_string("SELECTED_SHEET_NAME", "Selected Markets", sheet_settings),
        
        # Thresholds
        min_liquidity=_get_float("MIN_LIQUIDITY", 10000.0, sheet_settings),
        min_weekly_volume=_get_float("MIN_WEEKLY_VOLUME", 50000.0, sheet_settings),
        min_trend=_get_float("MIN_TREND", 0.30, sheet_settings),
        mm_score_min=_get_float("MM_SCORE_MIN", 1_000_000.0, sheet_settings),
        
        # Avellaneda-lite and quoting
        k_vol=_get_float("K_VOL", 2.0, sheet_settings),
        k_fee_ticks=_get_float("K_FEE_TICKS", 1.0, sheet_settings),
        alpha_fair=_get_float("ALPHA_FAIR", 0.2, sheet_settings),
        ewma_vol_window_sec=_get_int("EWMA_VOL_WINDOW_SEC", 600, sheet_settings),
        inv_gamma=_get_float("INV_GAMMA", 1.0, sheet_settings),
        
        # Risk Management
        soft_cap_delta_pct=_get_float("SOFT_CAP_DELTA_PCT", 0.015, sheet_settings),
        hard_cap_delta_pct=_get_float("HARD_CAP_DELTA_PCT", 0.03, sheet_settings),
        daily_loss_limit_pct=_get_float("DAILY_LOSS_LIMIT_PCT", 1.0, sheet_settings),
        
        # Order Management
        order_layers=_get_int("ORDER_LAYERS", 3, sheet_settings),
        base_size_usd=_get_float("BASE_SIZE_USD", 300.0, sheet_settings),
        max_size_usd=_get_float("MAX_SIZE_USD", 1500.0, sheet_settings),
        price_tick=_get_float("PRICE_TICK", 0.01, sheet_settings),
        requote_mid_ticks=_get_int("REQUOTE_MID_TICKS", 1, sheet_settings),
        requote_queue_levels=_get_int("REQUOTE_QUEUE_LEVELS", 2, sheet_settings),
        order_max_age_sec=_get_int("ORDER_MAX_AGE_SEC", 12, sheet_settings),
        # Merger
        merge_scan_interval_sec=_get_int("MERGE_SCAN_INTERVAL_SEC", 120, sheet_settings),
        min_merge_usdc=_get_float("MIN_MERGE_USDC", 0.10, sheet_settings),
        merge_chunk_usdc=_get_float("MERGE_CHUNK_USDC", 0.25, sheet_settings),
        merge_max_retries=_get_int("MERGE_MAX_RETRIES", 3, sheet_settings),
        merge_retry_backoff_ms=_get_int("MERGE_RETRY_BACKOFF_MS", 500, sheet_settings),
        merge_dry_run=_get_bool("MERGE_DRY_RUN", False, sheet_settings),
        # Backoff / cooldowns
        nonretryable_cooldown_sec=_get_int("NONRETRYABLE_COOLDOWN_SEC", 60, sheet_settings),
        
        # Networking
        gamma_base_url=_get_string("GAMMA_BASE_URL", "https://gamma-api.polymarket.com", sheet_settings),
        clob_base_url=_get_string("CLOB_BASE_URL", "https://clob.polymarket.com", sheet_settings),
        clob_ws_url=_get_string("CLOB_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/", sheet_settings),
        
        # Auth (from env only - these are secrets)
        pk=os.getenv("PK"),
        browser_address=os.getenv("BROWSER_ADDRESS"),

        # Logging
        log_level=_get_string("LOG_LEVEL", "INFO", sheet_settings),
        log_file=_get_string("LOG_FILE", "logs/mm_main.log", sheet_settings),
        log_rotation_backups=_get_int("LOG_ROTATION_BACKUPS", 5, sheet_settings),

        # Loop intervals
        selection_loop_sec=_get_int("SELECTION_LOOP_SEC", 900, sheet_settings),
        heartbeat_sec=_get_int("HEARTBEAT_SEC", 5, sheet_settings),
        backfill_throttle_sec=_get_int("BACKFILL_THROTTLE_SEC", 10, sheet_settings),
    )
