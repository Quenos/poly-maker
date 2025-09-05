import time
import logging
from typing import List, Tuple, Optional

import pandas as pd
import requests

from store_selected_markets import read_sheet
from poly_utils.google_utils import get_spreadsheet
from mm.state import StateStore


logger = logging.getLogger("mm.selection")


def normalize_selected_markets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns from Selected Markets without filtering.
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
        "token1": "token1",
        "token2": "token2",
        "condition_id": "condition_id",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = df[src]
    for c in ["liquidity", "volume24h", "volume7d", "volume30d"]:
        series = df[c] if c in df.columns else pd.Series([0.0] * len(df), index=df.index)
        df[c] = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # Ensure token/ids are strings
    for c in ["token1", "token2", "yes_token_id", "no_token_id", "condition_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Normalize optional 'suspended' column to boolean (case-insensitive header with synonyms)
    sus_col = None
    try:
        synonyms = {"suspended", "suspend", "suspend_buys", "pause_buys", "pause", "halt_buys"}
        for c in df.columns:
            if str(c).strip().lower() in synonyms:
                sus_col = c
                break
    except Exception:
        sus_col = None
    
    def _to_bool(x) -> bool:
        try:
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s in ("true", "1", "yes", "y")
        except Exception:
            return False
    if sus_col is not None:
        df["suspended"] = df[sus_col].map(_to_bool).fillna(False)
    else:
        df["suspended"] = False
    return df.reset_index(drop=True)


def enrich_gamma(df: pd.DataFrame, gamma_base: str) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        # Build targeted list of clob token ids strictly from token1/token2
        token_ids: List[str] = []
        for col in ("token1", "token2"):
            if col in df.columns:
                token_ids.extend([str(x) for x in df[col].dropna().astype(str).tolist() if str(x)])
        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            # No ids available; return unchanged (no fallback)
            return df

        # Query Gamma in chunks using clob_token_ids filter
        def _chunks(lst: List[str], n: int) -> List[List[str]]:
            return [lst[i:i + n] for i in range(0, len(lst), n)]

        items: List[dict] = []
        for chunk in _chunks(token_ids, 200):
            params = "&".join([f"clob_token_ids={requests.utils.quote(t)}" for t in chunk])
            url = f"{gamma_base}/markets?{params}"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            batch = resp.json() or []
            if batch:
                items.extend(batch)
        gdf = pd.DataFrame(items)
        if gdf.empty:
            return df

        # Join back to original by token id
        left = df.copy()
        # Join strictly on token1
        left_key = "token1"
        if left_key not in left.columns:
            return df
        left[left_key] = left[left_key].astype(str)

        # Gamma clobTokenIds may be list or comma-separated; derive first token as yes side proxy
        def _first_token(x) -> str:
            if isinstance(x, list) and x:
                return str(x[0])
            if isinstance(x, str) and x:
                return x.split(",")[0].strip()
            return ""

        gdf[left_key] = gdf.get("clobTokenIds", "").map(_first_token)
        merged = left.merge(gdf, on=left_key, how="left", suffixes=("", "_g"))
        return merged
    except Exception:
        return df


class SelectionManager:
    def __init__(self, gamma_base: str, state_store: Optional[StateStore] = None, sheet_name: str = "Selected Markets") -> None:
        self.gamma_base = gamma_base
        self.active_tokens: List[str] = []
        self.version: int = 0
        self.ts: float = 0.0
        self.state = state_store
        self.sheet_name = sheet_name
        logger.info("SelectionManager initialized with gamma_base=%s sheet=%s", gamma_base, sheet_name)

    def pull(self) -> Tuple[List[str], pd.DataFrame]:
        ss = get_spreadsheet(read_only=False)
        sel = read_sheet(ss, self.sheet_name)
        norm = normalize_selected_markets(sel)
        # Recompute screening metrics (Trend, MM_SCORE), but we still trade exactly Selected Markets

        enr = enrich_gamma(norm, self.gamma_base)
        # token list strictly from token1/token2
        tokens: List[str] = []
        for col in ("token1", "token2"):
            if col in enr.columns:
                tokens.extend([str(x) for x in enr[col].dropna().astype(str)])
        tokens = list(dict.fromkeys([t for t in tokens if t]))
        
        logger.debug("Pulled %d markets from sheet, derived %d unique tokens", len(sel), len(tokens))
        return tokens, enr

    def diff(self, new_tokens: List[str]) -> Tuple[List[str], List[str]]:
        new_set = set(new_tokens)
        old_set = set(self.active_tokens)
        to_add = sorted(list(new_set - old_set))
        to_remove = sorted(list(old_set - new_set))
        
        if to_add or to_remove:
            logger.info("Selection change detected: +%d markets, -%d markets", len(to_add), len(to_remove))
            if to_add:
                logger.info("Markets ADDED: %s", to_add)
            if to_remove:
                logger.info("Markets REMOVED: %s", to_remove)
        
        return to_add, to_remove

    def snapshot(self, tokens: List[str]) -> Tuple[int, float]:
        if self.state is not None:
            v, ts = self.state.write_selection_snapshot(tokens)
            self.version, self.ts = v, ts
            logger.info("Selection snapshot v%d: %d active markets at %s", v, len(tokens), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)))
        else:
            # Fallback if no state store provided
            self.version += 1
            self.ts = time.time()
            logger.info("Selection snapshot v%d (no state): %d active markets at %s", self.version, len(tokens), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)))
        
        self.active_tokens = tokens
        return self.version, self.ts

    def tick(self) -> Tuple[List[str], List[str]]:
        tokens, _ = self.pull()
        to_add, to_remove = self.diff(tokens)
        if to_add or to_remove:
            self.snapshot(tokens)
        return to_add, to_remove
