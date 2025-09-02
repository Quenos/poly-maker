import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class MMConfig:
    """Configuration values for the market making daemon, loaded from env with defaults."""

    # Sheet and project
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
    requote_mid_ticks: int = 1
    requote_queue_levels: int = 2
    order_max_age_sec: int = 12
    daily_loss_limit_pct: float = 1.0

    # Networking
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    # Auth
    pk: Optional[str] = None
    browser_address: Optional[str] = None


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


def load_config() -> MMConfig:
    """Load configuration from environment variables with defaults.

    Required env:
      - SPREADSHEET_URL
    Optional env (override defaults):
      - MIN_LIQUIDITY, MIN_WEEKLY_VOLUME, MIN_TREND, MM_SCORE_MIN, K_VOL,
        K_FEE_TICKS, ALPHA_FAIR, EWMA_VOL_WINDOW_SEC, INV_GAMMA,
        SOFT_CAP_DELTA_PCT, HARD_CAP_DELTA_PCT, ORDER_LAYERS, BASE_SIZE_USD,
        MAX_SIZE_USD, REQUOTE_MID_TICKS, REQUOTE_QUEUE_LEVELS, ORDER_MAX_AGE_SEC,
        DAILY_LOSS_LIMIT_PCT, PK, BROWSER_ADDRESS
    """
    spreadsheet_url = os.getenv("SPREADSHEET_URL")
    if not spreadsheet_url:
        raise ValueError("SPREADSHEET_URL is required")

    return MMConfig(
        spreadsheet_url=spreadsheet_url,
        selected_sheet_name=os.getenv("SELECTED_SHEET_NAME", "Selected Markets"),
        min_liquidity=_get_float("MIN_LIQUIDITY", 10000.0),
        min_weekly_volume=_get_float("MIN_WEEKLY_VOLUME", 50000.0),
        min_trend=_get_float("MIN_TREND", 0.30),
        mm_score_min=_get_float("MM_SCORE_MIN", 1_000_000.0),
        k_vol=_get_float("K_VOL", 2.0),
        k_fee_ticks=_get_float("K_FEE_TICKS", 1.0),
        alpha_fair=_get_float("ALPHA_FAIR", 0.2),
        ewma_vol_window_sec=_get_int("EWMA_VOL_WINDOW_SEC", 600),
        inv_gamma=_get_float("INV_GAMMA", 1.0),
        soft_cap_delta_pct=_get_float("SOFT_CAP_DELTA_PCT", 0.015),
        hard_cap_delta_pct=_get_float("HARD_CAP_DELTA_PCT", 0.03),
        order_layers=_get_int("ORDER_LAYERS", 3),
        base_size_usd=_get_float("BASE_SIZE_USD", 300.0),
        max_size_usd=_get_float("MAX_SIZE_USD", 1500.0),
        requote_mid_ticks=_get_int("REQUOTE_MID_TICKS", 1),
        requote_queue_levels=_get_int("REQUOTE_QUEUE_LEVELS", 2),
        order_max_age_sec=_get_int("ORDER_MAX_AGE_SEC", 12),
        daily_loss_limit_pct=_get_float("DAILY_LOSS_LIMIT_PCT", 1.0),
        pk=os.getenv("PK"),
        browser_address=os.getenv("BROWSER_ADDRESS"),
    )

