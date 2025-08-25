import logging
from typing import List

import pandas as pd
from gspread_dataframe import set_with_dataframe

from data_updater.google_utils import get_spreadsheet
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv()


def read_sheet(spreadsheet, title: str) -> pd.DataFrame:
    try:
        wk = spreadsheet.worksheet(title)
        df = pd.DataFrame(wk.get_all_records())
        return df
    except Exception as e:
        logger.error("Failed to read sheet '%s': %s", title, str(e))
        return pd.DataFrame()


def write_sheet(spreadsheet, title: str, df: pd.DataFrame) -> None:
    try:
        try:
            wk = spreadsheet.worksheet(title)
        except Exception:
            wk = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(10, len(df.columns) + 2))
        set_with_dataframe(wk, df, include_index=False, include_column_header=True, resize=True)
        logger.info("Wrote %d rows to sheet '%s'", len(df), title)
    except Exception as e:
        logger.error("Failed to write sheet '%s': %s", title, str(e))


def select_rows(all_markets: pd.DataFrame, selected_questions: List[str]) -> pd.DataFrame:
    if all_markets.empty or not selected_questions:
        return pd.DataFrame(columns=all_markets.columns if not all_markets.empty else [])

    # Ensure string comparison on 'question'
    all_markets = all_markets.copy()
    if "question" in all_markets.columns:
        all_markets["question"] = all_markets["question"].astype(str)
    else:
        logger.warning("'All Markets' sheet missing 'question' column; nothing to select")
        return pd.DataFrame(columns=all_markets.columns)

    selected_set = set(str(q) for q in selected_questions if str(q).strip())
    matched = all_markets[all_markets["question"].isin(selected_set)].copy()

    # Log any selected questions that were not found for transparency
    found = set(matched["question"].astype(str).tolist())
    missing = [q for q in selected_set if q not in found]
    if missing:
        logger.info("%d selected questions not found in All Markets (first few): %s", len(missing), list(missing)[:5])

    return matched


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting Selected->Stored Sel Markets sync")

    spreadsheet = get_spreadsheet()

    sel_df = read_sheet(spreadsheet, "Selected Markets")
    all_df = read_sheet(spreadsheet, "All Markets")

    if sel_df.empty:
        logger.info("Selected Markets is empty; writing empty output")
        write_sheet(spreadsheet, "Stored Sel Markets", pd.DataFrame(columns=all_df.columns if not all_df.empty else []))
        return

    # Prefer 'question' column; fallback to 'market' if present
    selected_questions: List[str] = []
    if "question" in sel_df.columns:
        selected_questions = sel_df["question"].astype(str).tolist()
    elif "market" in sel_df.columns:
        selected_questions = sel_df["market"].astype(str).tolist()
    else:
        logger.error("Selected Markets missing 'question' column; aborting")
        write_sheet(spreadsheet, "Stored Sel Markets", pd.DataFrame(columns=all_df.columns if not all_df.empty else []))
        return

    matched_df = select_rows(all_df, selected_questions)

    # Ensure identifier columns are written as strings (avoid scientific notation)
    if not matched_df.empty:
        for col in ["token1", "token2", "token_id", "condition_id"]:
            if col in matched_df.columns:
                matched_df[col] = matched_df[col].astype(str)

    # Preserve All Markets column order for consistency
    if not matched_df.empty and not all_df.empty:
        matched_df = matched_df.reindex(columns=all_df.columns, fill_value="")

    write_sheet(spreadsheet, "Stored Sel Markets", matched_df)
    logger.info("Done")


if __name__ == "__main__":
    main()
