import logging
from typing import Optional

import pandas as pd
import numpy as np

from data_updater.google_utils import get_spreadsheet
from store_selected_markets import read_sheet, write_sheet
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv()


def _get_key_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    if "question" in df.columns:
        return "question"
    if "market" in df.columns:
        return "market"
    return None


def _normalize_key_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _sanitize_for_gsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf and NaN with None so the Sheets API can accept the payload."""
    if df is None or df.empty:
        return df
    safe = df.copy()
    # Replace infinities with NA first, then convert all NA/NaN to None (empty cell)
    safe = safe.replace([np.inf, -np.inf], pd.NA)
    safe = safe.where(~safe.isna(), other=None)
    return safe


def _coerce_identifier_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure token/id columns are strings to avoid scientific notation in Sheets."""
    if df is None or df.empty:
        return df
    cols_to_string = [
        "token1",
        "token2",
        "token_id",
        "condition_id",
    ]
    coerced = df.copy()
    for col in cols_to_string:
        if col in coerced.columns:
            coerced[col] = coerced[col].astype(str)
    return coerced


def append_missing_rows() -> None:
    spreadsheet = get_spreadsheet()

    stored_df = read_sheet(spreadsheet, "Stored Sel Markets")
    all_df = read_sheet(spreadsheet, "All Markets")

    if stored_df.empty:
        logger.info("'Stored Sel Markets' is empty; nothing to append")
        return

    key_col_stored = _get_key_column(stored_df)
    key_col_all = _get_key_column(all_df)

    if key_col_stored is None:
        logger.error("'Stored Sel Markets' missing identifier column ('question' or 'market'); aborting")
        return

    if key_col_all is None:
        # If All Markets is entirely empty, initialize with stored schema
        logger.warning("'All Markets' missing identifier column; initializing with Stored schema")
        # Ensure we keep the stored order
        init_df = _coerce_identifier_columns_to_string(stored_df.copy())
        init_df = _sanitize_for_gsheet(init_df)
        write_sheet(spreadsheet, "All Markets", init_df)
        logger.info("Wrote %d rows to 'All Markets' from 'Stored Sel Markets'", len(stored_df))
        return

    stored_keys = set(_normalize_key_series(stored_df[key_col_stored]).tolist())
    all_keys = set(_normalize_key_series(all_df[key_col_all]).tolist()) if not all_df.empty else set()

    missing_keys = [k for k in stored_keys if k not in all_keys]
    if not missing_keys:
        logger.info("No missing rows; 'All Markets' already contains all entries from 'Stored Sel Markets'")
        return

    # Filter stored_df for missing rows
    missing_df = stored_df[_normalize_key_series(stored_df[key_col_stored]).isin(missing_keys)].copy()

    # Align columns to All Markets' schema; fill absent columns with empty strings
    missing_aligned = missing_df.reindex(columns=all_df.columns, fill_value="")

    # Concatenate and reset index
    combined = pd.concat([all_df, missing_aligned], ignore_index=True)

    # Write back to All Markets
    combined = _coerce_identifier_columns_to_string(combined)
    combined = _sanitize_for_gsheet(combined)
    write_sheet(spreadsheet, "All Markets", combined)
    logger.info("Appended %d missing rows to 'All Markets'", len(missing_aligned))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting append of missing rows from 'Stored Sel Markets' to 'All Markets'")
    append_missing_rows()
    logger.info("Done")


if __name__ == "__main__":
    main()
