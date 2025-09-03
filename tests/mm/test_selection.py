from unittest.mock import patch

from mm.selection import SelectionManager


@patch("mm.selection.get_spreadsheet")
@patch("mm.selection.read_sheet")
@patch("mm.selection.enrich_gamma")
def test_selection_add_remove(enrich_gamma, read_sheet, get_spreadsheet):
    # Mock Selected Markets
    import pandas as pd
    df1 = pd.DataFrame({"token1": ["a", "b"], "token2": ["c", "d"]})
    df2 = pd.DataFrame({"token1": ["a"], "token2": ["c"]})
    read_sheet.side_effect = [df1, df2]
    get_spreadsheet.return_value = object()
    enrich_gamma.side_effect = [df1, df2]

    sel = SelectionManager(gamma_base="https://gamma-api.polymarket.com")
    # First pull
    to_add, to_remove = sel.tick()
    assert set(to_add) == set(["a", "b", "c", "d"]) and not to_remove
    # Second pull (removed b,d)
    to_add2, to_remove2 = sel.tick()
    assert not to_add2 and set(to_remove2) == set(["b", "d"]) 



