#!/usr/bin/env python3
import logging
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe

from poly_utils.google_utils import get_spreadsheet


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_bankroll_rows() -> pd.DataFrame:
    """Return a dataframe with min/optimal bankroll suggestions for given trade/max sizes.

    Assumptions:
    - neg_risk TRUE:
      - min ≈ 3 × trade_size (initial concurrent bids and buffer)
      - optimal ≈ (max_size + trade_size) × 1.15 (15% buffer)
    - neg_risk FALSE:
      - Use multipliers on the TRUE case to reflect lack of netting when quoting both outcomes
      - min_FALSE ≈ 1.5 × min_TRUE, optimal_FALSE ≈ 1.75 × optimal_TRUE
    """
    scenarios: List[Dict[str, float]] = [
        {"trade_size": 20.0, "max_size": 100.0},
        {"trade_size": 50.0, "max_size": 250.0},
        {"trade_size": 100.0, "max_size": 500.0},
    ]

    records: List[Dict[str, float]] = []

    for s in scenarios:
        trade_size = float(s["trade_size"])  # per-order size
        max_size = float(s["max_size"])      # max inventory target

        min_true = 3.0 * trade_size
        optimal_true = (max_size + trade_size) * 1.15

        min_false = 1.5 * min_true
        optimal_false = 1.75 * optimal_true

        records.append({
            "trade_size": trade_size,
            "max_size": max_size,
            "min_bankroll_neg_risk_TRUE_usdc": round(min_true, 2),
            "optimal_bankroll_neg_risk_TRUE_usdc": round(optimal_true, 2),
            "min_bankroll_neg_risk_FALSE_usdc": round(min_false, 2),
            "optimal_bankroll_neg_risk_FALSE_usdc": round(optimal_false, 2),
            "assumptions": "TRUE=min≈3x trade; optimal≈(max+trade)*1.15. FALSE≈1.5x/1.75x of TRUE.",
        })

    return pd.DataFrame.from_records(records)


def upsert_optimal_bankroll_sheet() -> None:
    load_dotenv()
    setup_logging()

    df = compute_bankroll_rows()
    logging.info("Computed bankroll rows:\n%s", df)

    # Open spreadsheet (authenticated mode)
    ss = get_spreadsheet(read_only=False)

    title = "Optimal Bankroll"
    try:
        try:
            ws = ss.worksheet(title)
        except Exception:
            ws = ss.add_worksheet(title=title, rows=50, cols=10)

        # Clear and write fresh content
        try:
            ws.clear()
        except Exception:
            pass

        set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
        logging.info("Wrote %d rows to sheet '%s'", len(df), title)
    except Exception as e:
        logging.error("Failed to update sheet '%s': %s", title, str(e))
        raise


if __name__ == "__main__":
    upsert_optimal_bankroll_sheet()


