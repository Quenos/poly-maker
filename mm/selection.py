import time
import logging
from typing import List, Tuple, Optional

import pandas as pd

from store_selected_markets import read_sheet
from poly_utils.google_utils import get_spreadsheet
from mm.state import StateStore


logger = logging.getLogger("mm.selection")


def normalize_selected_markets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["token1", "token2", "condition_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def enrich_gamma(df: pd.DataFrame, gamma_base: str) -> pd.DataFrame:
    # Placeholder: tests patch this; default behavior is passthrough
    return df


class SelectionManager:
    def __init__(self, gamma_base: str, state_store: Optional[StateStore] = None) -> None:
        self.gamma_base = gamma_base
        self.active_tokens: List[str] = []
        self.version: int = 0
        self.ts: float = 0.0
        self.state = state_store
        logger.info("SelectionManager initialized with gamma_base=%s", gamma_base)

    def pull(self) -> Tuple[List[str], pd.DataFrame]:
        ss = get_spreadsheet(read_only=False)
        sel = read_sheet(ss, "Selected Markets")
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
