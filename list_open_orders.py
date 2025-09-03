#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
import requests


def main() -> None:
    load_dotenv()
    pk = os.getenv("PK")
    funder = os.getenv("BROWSER_ADDRESS")
    if not pk or not funder:
        print("Missing PK or BROWSER_ADDRESS in environment")
        return

    client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=POLYGON, funder=funder)
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    orders = client.get_orders()
    if not orders:
        print("No open orders.")
        return

    df = pd.DataFrame(orders)
    # Show only active orders
    if "status" in df.columns:
        df = df[df["status"].isin(["LIVE", "OPEN"])].copy()
        if df.empty:
            print("No open orders.")
            return
    print(f"Fetched {len(df)} open orders")
    print(f"Order fields: {list(df.columns)}")
    # Fetch market metadata to map token_id/asset_id/market -> market name (robust to schema changes)
    token_to_name, market_to_name = {}, {}

    def harvest(markets_json):
        for m in markets_json:
            name = m.get("question") or m.get("title") or m.get("name") or m.get("slug") or m.get("questionTitle") or ""
            mid = m.get("id") or m.get("market") or m.get("conditionId")
            if mid:
                market_to_name[str(mid)] = name
            # outcomes/tokens variants
            for key in ("outcomes", "tokens", "outcomeTokens"):
                if key in m and isinstance(m[key], list):
                    for o in m[key]:
                        if isinstance(o, dict):
                            tid = o.get("token_id") or o.get("tokenId") or o.get("id")
                            if tid:
                                token_to_name[str(tid)] = name
    # Try CLOB markets
    try:
        r = requests.get("https://clob.polymarket.com/markets?limit=1000", timeout=10)
        if r.ok:
            data = r.json()
            harvest(data if isinstance(data, list) else data.get("markets", []))
    except Exception:
        pass
    # Try data-api as fallback
    try:
        r = requests.get("https://data-api.polymarket.com/markets?limit=1000", timeout=10)
        if r.ok:
            data = r.json()
            harvest(data if isinstance(data, list) else data.get("markets", []))
    except Exception:
        pass

    # Also query assets endpoint directly for exact token→name mapping
    token_col = None
    for cand in ("token_id", "asset_id", "tokenId", "assetId"):
        if cand in df.columns:
            token_col = cand
            break
    market_col = "market" if "market" in df.columns else None
    if token_col:
        unique_tokens = sorted(set(df[token_col].astype(str)))
        if unique_tokens:
            # Batch in chunks to avoid very long URLs
            for i in range(0, len(unique_tokens), 50):
                chunk = unique_tokens[i:i+50]
                qs = ",".join(chunk)
                for url in (
                    f"https://data-api.polymarket.com/assets?ids={qs}",
                    f"https://clob.polymarket.com/assets?ids={qs}",
                ):
                    try:
                        rr = requests.get(url, timeout=10)
                        if rr.ok:
                            arr = rr.json()
                            if isinstance(arr, dict) and "assets" in arr:
                                arr = arr["assets"]
                            if isinstance(arr, list):
                                for a in arr:
                                    if not isinstance(a, dict):
                                        continue
                                    aid = str(a.get("id") or a.get("token_id") or a.get("tokenId") or "")
                                    if not aid:
                                        continue
                                    # Prefer a question/title at market level, else asset name
                                    mname = a.get("question") or a.get("market_question") or a.get("title") or a.get("name") or ""
                                    token_to_name[aid] = mname
                            break
                    except Exception:
                        continue

    # If names still missing, try per-market lookups using market field
    if market_col:
        unique_markets = sorted(set(df[market_col].astype(str)))
        for mid in unique_markets:
            if mid in market_to_name and market_to_name[mid]:
                continue
            urls = [
                f"https://clob.polymarket.com/markets/{mid}",
                f"https://data-api.polymarket.com/markets/{mid}",
                f"https://clob.polymarket.com/markets?ids={mid}",
                f"https://data-api.polymarket.com/markets?ids={mid}",
            ]
            for url in urls:
                try:
                    resp = requests.get(url, timeout=8)
                    if not resp.ok:
                        continue
                    data = resp.json()
                    if isinstance(data, dict) and "markets" in data:
                        items = data["markets"]
                    elif isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                    harvest(items)
                    if mid in market_to_name and market_to_name[mid]:
                        break
                except Exception:
                    continue
    # Attach names (simple mapping without mask)
    if token_col and any(token_to_name.values()):
        df["market_name"] = df[token_col].astype(str).map(token_to_name)
    elif market_col and any(market_to_name.values()):
        df["market_name"] = df[market_col].astype(str).map(market_to_name)
    else:
        df["market_name"] = ""
    df["market_name"] = df["market_name"].fillna("")
    cols = [
        "order_id",
        "token_id",
        "market_name",
        "outcome",
        "side",
        "price",
        "original_size",
        "size_matched",
        "status",
        "created_at",
    ]
    existing = [c for c in cols if c in df.columns]
    df[existing] = df[existing]

    # Sort newest first if available
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    # Pretty print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
