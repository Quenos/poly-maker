import pandas as pd


def filter_markets(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Screen markets using thresholds and MM score.
    MM score = Liquidity * sqrt(max(1, Volume_7d)).
    """
    df = df.copy()
    col_map = {
        "Liquidity": "liquidity",
        "Volume_24h": "volume24h",
        "Volume_7d": "volume7d",
        "Volume_30d": "volume30d",
        "market_id": "market_id",
        "yes_token_id": "yes_token_id",
        "no_token_id": "no_token_id",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = df[src]
    for c in ["liquidity", "volume24h", "volume7d", "volume30d"]:
        series = df[c] if c in df.columns else pd.Series([0.0] * len(df), index=df.index)
        df[c] = pd.to_numeric(series, errors="coerce").fillna(0.0)
    df["trend"] = df["volume7d"] / df["volume30d"].replace(0, 1)
    df["mm_score"] = df["liquidity"] * (df["volume7d"].replace(0, 1).pow(0.5))
    kept = df[(df["liquidity"] >= cfg.min_liquidity) & (df["volume7d"] >= cfg.min_weekly_volume) & (df["trend"] >= cfg.min_trend) & (df["mm_score"] >= cfg.mm_score_min)].copy()
    return kept.reset_index(drop=True)


