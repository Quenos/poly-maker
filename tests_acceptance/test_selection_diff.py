from unittest.mock import patch
from mm.selection import SelectionManager


@patch("mm.selection.get_spreadsheet")
@patch("mm.selection.read_sheet")
@patch("mm.selection.enrich_gamma")
def test_selection_diff(enrich_gamma, read_sheet, get_spreadsheet):
    import pandas as pd
    df1 = pd.DataFrame({"token1": ["a"], "token2": ["b"]})
    df2 = pd.DataFrame({"token1": ["a", "c"], "token2": ["b", "d"]})
    read_sheet.side_effect = [df1, df2]
    enrich_gamma.side_effect = [df1, df2]
    get_spreadsheet.return_value = object()
    sel = SelectionManager("https://gamma-api.polymarket.com")
    add1, rem1 = sel.tick()
    assert set(add1) == {"a", "b"} and not rem1
    add2, rem2 = sel.tick()
    assert set(add2) == {"c", "d"} and not rem2



