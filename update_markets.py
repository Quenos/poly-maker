import time
import json
import logging
from typing import Any, Dict, List

import pandas as pd
import requests
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import get_sel_df, get_all_markets, get_all_results, get_markets, add_volatility_to_df
from gspread_dataframe import set_with_dataframe
import traceback

# Initialize global variables
spreadsheet = get_spreadsheet()
client = get_clob_client()

wk_all = spreadsheet.worksheet("All Markets")
wk_vol = spreadsheet.worksheet("Volatility Markets")

sel_df = get_sel_df(spreadsheet, "Selected Markets")

def update_sheet(data, worksheet):
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    # Create a DataFrame with the maximum size and fill it with empty strings
    padded_data = pd.DataFrame('', index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(worksheet, padded_data, include_index=False, include_column_header=True, resize=True)

def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df['gm_reward_per_100'].mean()
    std_gm = df['gm_reward_per_100'].std()
    
    mean_volatility = df['volatility_sum'].mean()
    std_volatility = df['volatility_sum'].std()
    
    # Standardize the columns
    df['std_gm_reward_per_100'] = (df['gm_reward_per_100'] - mean_gm) / std_gm
    df['std_volatility_sum'] = (df['volatility_sum'] - mean_volatility) / std_volatility
    
    # Define a custom scoring function for best_bid and best_ask
    def proximity_score(value):
        if 0.1 <= value <= 0.25:
            return (0.25 - value) / 0.15
        elif 0.75 <= value <= 0.9:
            return (value - 0.75) / 0.15
        else:
            return 0
    
    df['bid_score'] = df['best_bid'].apply(proximity_score)
    df['ask_score'] = df['best_ask'].apply(proximity_score)
    
    # Create a composite score (higher is better for rewards, lower is better for volatility, with proximity scores)
    df['composite_score'] = (
        df['std_gm_reward_per_100'] - df['std_volatility_sum'] + df['bid_score'] + df['ask_score']
    )
    
    # Sort by the composite score in descending order
    sorted_df = df.sort_values(by='composite_score', ascending=False)
    
    # Drop the intermediate columns used for calculation
    sorted_df = sorted_df.drop(columns=['std_gm_reward_per_100', 'std_volatility_sum', 'bid_score', 'ask_score', 'composite_score'])
    
    return sorted_df

logger = logging.getLogger(__name__)

def _safe_parse_array(value: Any) -> List[Any]:
    """
    Parse a value that may be a JSON-encoded array string into a Python list.
    If it's already a list, return as-is; otherwise return an empty list on failure.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def _map_gamma_market_to_expected_schema(market: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a single Gamma /markets object to the schema expected by find_markets.process_single_row.
    """
    clob_token_ids = _safe_parse_array(market.get("clobTokenIds") or market.get("clob_token_ids"))
    outcomes = _safe_parse_array(market.get("outcomes"))
    outcome_prices = _safe_parse_array(market.get("outcomePrices") or market.get("outcome_prices"))

    tokens: List[Dict[str, Any]] = []
    for idx in range(min(len(clob_token_ids), len(outcomes))):
        token = {
            "token_id": str(clob_token_ids[idx]),
            "outcome": outcomes[idx],
            "price": float(outcome_prices[idx]) if idx < len(outcome_prices) else 0.0,
            "winner": False,
        }
        tokens.append(token)

    rewards_obj: Dict[str, Any] = {
        "rates": [],  # Not exposed by Gamma; downstream handles empty as zero
        "min_size": market.get("rewardsMinSize", market.get("orderMinSize", 0)),
        "max_spread": market.get("rewardsMaxSpread", 0),
    }

    mapped = {
        "question": market.get("question", ""),
        "neg_risk": market.get("negRisk", market.get("neg_risk", False)),
        "tokens": tokens,
        "rewards": rewards_obj,
        "minimum_tick_size": market.get("orderPriceMinTickSize", market.get("minimum_tick_size", 0.01)),
        "end_date_iso": market.get("endDateIso", market.get("end_date_iso")),
        "market_slug": market.get("slug", market.get("market_slug", "")),
        "condition_id": market.get("conditionId", market.get("condition_id", "")),
    }

    return mapped

def get_all_markets_gamma(limit: int = 500) -> pd.DataFrame:
    """
    Fetch markets from Gamma API and map to the schema expected by find_markets.
    """
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "limit": limit,
        "active": True,
        "closed": False,
        "order": "volume",
        "ascending": False,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        markets: List[Dict[str, Any]] = resp.json()
    except Exception as ex:
        logger.error("Failed to fetch Gamma markets: %s", ex)
        # Fall back to empty DataFrame; upstream code will handle empty gracefully
        return pd.DataFrame()

    mapped_rows = [_map_gamma_market_to_expected_schema(m) for m in markets]
    return pd.DataFrame(mapped_rows)

def fetch_and_process_data():
    global spreadsheet, client, wk_all, wk_vol, sel_df
    
    spreadsheet = get_spreadsheet()
    client = get_clob_client()

    wk_all = spreadsheet.worksheet("All Markets")
    wk_vol = spreadsheet.worksheet("Volatility Markets")
    wk_full = spreadsheet.worksheet("Full Markets")
    sel_df = get_sel_df(spreadsheet, "Selected Markets")
    # Prefer Gamma /markets endpoint for market metadata
    all_df = get_all_markets_gamma()
    if len(all_df) == 0:
        print("Gamma markets fetch returned empty; falling back to client sampling markets")
        all_df = get_all_markets(client)
    print("Got all Markets")
    all_results = get_all_results(all_df, client)
    print("Got all Results")
    m_data, all_markets = get_markets(all_results, sel_df, maker_reward=0.75)
    print("Got all orderbook")

    print(f'{pd.to_datetime("now")}: Fetched all markets data of length {len(all_markets)}.')
    new_df = add_volatility_to_df(all_markets)
    new_df['volatility_sum'] = new_df['24_hour'] + new_df['7_day'] + new_df['14_day']

    new_df = new_df.sort_values('volatility_sum', ascending=True)
    new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

    new_df = new_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100', 'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day', 'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size', 'neg_risk', 'market_slug', 'token1', 'token2', 'condition_id']]

    volatility_df = new_df.copy()
    volatility_df = volatility_df[new_df['volatility_sum'] < 20]
    # volatility_df = sort_df(volatility_df)
    volatility_df = volatility_df.sort_values('gm_reward_per_100', ascending=False)
   
    new_df = new_df.sort_values('gm_reward_per_100', ascending=False)
    print(f'{pd.to_datetime("now")}: Fetched select market of length {len(new_df)}.')

    if len(new_df) > 50:
        update_sheet(new_df, wk_all)
        update_sheet(volatility_df, wk_vol)
        update_sheet(m_data, wk_full)
    else:
        print(f'{pd.to_datetime("now")}: Not updating sheet because of length {len(new_df)}.')

if __name__ == "__main__":
    while True:
        try:
            fetch_and_process_data()
            time.sleep(60 * 60)  # Sleep for an hour
        except Exception as e:
            traceback.print_exc()
            print(str(e))
