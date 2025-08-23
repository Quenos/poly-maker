import pandas as pd
from py_clob_client.headers.headers import create_level_2_headers
from py_clob_client.clob_types import RequestArgs

from poly_utils.google_utils import get_spreadsheet
from gspread_dataframe import set_with_dataframe
import requests
import json
import os

from dotenv import load_dotenv
load_dotenv()

spreadsheet = get_spreadsheet()

def get_markets_df(wk_full):
    markets_df = pd.DataFrame(wk_full.get_all_records())
    markets_df = markets_df[['question', 'answer1', 'answer2', 'token1', 'token2']]
    markets_df['token1'] = markets_df['token1'].astype(str)
    markets_df['token2'] = markets_df['token2'].astype(str)
    return markets_df

def get_all_orders(client):
    orders = client.client.get_orders()
    orders_df = pd.DataFrame(orders)

    if len(orders_df) > 0:
        orders_df['order_size'] = orders_df['original_size'].astype('float') - orders_df['size_matched'].astype('float')
        orders_df = orders_df[['asset_id', 'order_size', 'side', 'price']]

        orders_df = orders_df.rename(columns={'side': 'order_side', 'price': 'order_price'})
        return orders_df
    else:
        return pd.DataFrame()
    
def get_all_positions(client):
    try:
        positions = client.get_all_positions()
        positions = positions[['asset', 'size', 'avgPrice', 'curPrice', 'percentPnl']]
        positions = positions.rename(columns={'size': 'position_size'})
        return positions
    except:
        return pd.DataFrame()
    
def combine_dfs(orders_df, positions, markets_df, selected_df):
    """Join orders/positions with market metadata safely.

    The previous implementation asserted that every merged row matched a market
    token, which can fail when new markets appear or sheets are out of sync.
    This version degrades gracefully and still returns a useful summary.
    """
    # Normalize id columns for join
    if 'asset_id' not in orders_df.columns:
        for cand in ['asset', 'token_id', 'tokenId']:
            if cand in orders_df.columns:
                orders_df = orders_df.rename(columns={cand: 'asset_id'})
                break
    pos_id_col = None
    for cand in ['asset', 'asset_id', 'token', 'tokenId']:
        if cand in positions.columns:
            pos_id_col = cand
            break
    if pos_id_col and pos_id_col != 'asset':
        positions = positions.rename(columns={pos_id_col: 'asset'})
    elif not pos_id_col:
        positions = positions.copy()
        positions['asset'] = ''

    merged_df = orders_df.merge(positions, left_on=['asset_id'], right_on=['asset'], how='outer')
    merged_df['asset_id'] = merged_df['asset_id'].combine_first(merged_df['asset'])
    if 'asset' in merged_df.columns:
        merged_df = merged_df.drop(columns='asset', axis=1)

    # First attempt: strict join via token1/token2
    merge_token1 = merged_df.merge(markets_df, left_on='asset_id', right_on='token1', how='inner')
    merge_token1['merged_with'] = 'token1'
    merge_token2 = merged_df.merge(markets_df, left_on='asset_id', right_on='token2', how='inner')
    merge_token2['merged_with'] = 'token2'
    combined_df = pd.concat([merge_token1, merge_token2], ignore_index=True)

    # Fallback: if no matches (or partial), build mapping and fill question/answer
    if combined_df.empty or len(combined_df) < len(merged_df):
        # Build quick maps for question/answers by token
        t1_to_q = dict(zip(markets_df.get('token1', []), markets_df.get('question', [])))
        t2_to_q = dict(zip(markets_df.get('token2', []), markets_df.get('question', [])))
        t1_to_a = dict(zip(markets_df.get('token1', []), markets_df.get('answer1', [])))
        t2_to_a = dict(zip(markets_df.get('token2', []), markets_df.get('answer2', [])))

        fallback = merged_df.copy()
        aid = fallback['asset_id'].astype(str)
        # Question from either token map
        fallback['question'] = aid.map(t1_to_q).fillna(aid.map(t2_to_q)).fillna('')
        # Answer from the side that matches
        ans1 = aid.map(t1_to_a)
        ans2 = aid.map(t2_to_a)
        fallback['answer'] = ans1.where(ans1.notna(), ans2).fillna('')

        keep_cols = ['question', 'answer', 'order_size', 'order_side', 'order_price', 'position_size', 'avgPrice', 'curPrice']
        for col in keep_cols:
            if col not in fallback.columns:
                fallback[col] = 0 if col not in ('question', 'answer', 'order_side') else ''

        # Ensure combined_df has the columns before selecting
        for col in keep_cols:
            if col not in combined_df.columns:
                combined_df[col] = '' if col in ('question', 'answer', 'order_side') else 0

        combined_df = pd.concat([
            combined_df[keep_cols],
            fallback[keep_cols]
        ], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Final shaping
    combined_df['order_side'] = combined_df['order_side'].fillna('')
    combined_df = combined_df.fillna(0)
    combined_df['marketInSelected'] = combined_df['question'].isin(selected_df['question'])
    combined_df = combined_df.sort_values(['marketInSelected', 'question']).reset_index(drop=True)
    return combined_df

def get_earnings(client):
    args = RequestArgs(method='GET', request_path='/rewards/user/markets')
    l2Headers = create_level_2_headers(client.signer, client.creds, args)
    url = "https://polymarket.com/api/rewards/markets"

    cursor = ''
    markets = []

    params = {
        "l2Headers": json.dumps(l2Headers),
        "orderBy": "earnings",
        "position": "DESC",
        "makerAddress": os.getenv('BROWSER_WALLET'),
        "authenticationType": "eoa",
        "nextCursor": cursor,
        "requestPath": "/rewards/user/markets"
    }

    r = requests.get(url,  params=params)
    results = r.json()

    data = pd.DataFrame(results['data'])
    data['earnings'] = data['earnings'].apply(lambda x: x[0]['earnings'])

    data = data[data['earnings'] > 0].reset_index(drop=True)
    data = data[['question', 'earnings', 'earning_percentage']]
    return data



def update_stats_once(client):
    spreadsheet = get_spreadsheet()
    wk_full = spreadsheet.worksheet('Full Markets')
    wk_summary = spreadsheet.worksheet('Summary')


    wk_sel = spreadsheet.worksheet('Selected Markets')
    selected_df = pd.DataFrame(wk_sel.get_all_records())
    
    markets_df = get_markets_df(wk_full)
    # Ensure string types for token columns to avoid join misses
    for col in ('token1', 'token2'):
        if col in markets_df.columns:
            markets_df[col] = markets_df[col].astype(str)
    print("Got spreadsheet...")

    orders_df = get_all_orders(client)
    print("Got Orders...")
    positions = get_all_positions(client)
    print("Got Positions...")

    if len(positions) > 0 or len(orders_df) > 0:
        combined_df = combine_dfs(orders_df, positions, markets_df, selected_df)
        earnings = get_earnings(client.client)
        print("Got Earnings...")
        combined_df = combined_df.merge(earnings, on='question', how='left')

        combined_df = combined_df.fillna(0)
        combined_df = combined_df.round(2)

        combined_df = combined_df.sort_values('earnings', ascending=False)
        combined_df = combined_df[['question', 'answer', 'order_size', 'position_size', 'marketInSelected', 'earnings', 'earning_percentage']]
        wk_summary.clear()

        set_with_dataframe(wk_summary, combined_df, include_index=False, include_column_header=True, resize=True)
    else:
        print("Position or order is empty")