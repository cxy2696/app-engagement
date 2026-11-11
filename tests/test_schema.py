import pandas as pd
import pytest

from engagegraph.data.schema import SchemaError, clean_interactions, normalize_label


def test_normalizes_fullwidth_and_whitespace():
    assert normalize_label("  Ｔｉｋ\u200b  Ｔｏｋ  ") == "Tik Tok"  # noqa: RUF001


def test_cleans_duplicates_negatives_and_missing():
    frame = pd.DataFrame(
        {
            "u": ["a", "a", "b", "c", None, "d"],
            "i": [" X ", "X", "Y", "Z", "W", "V"],
            "w": [10, 5, -1, 2, 7, None],
        }
    )
    cleaned, report = clean_interactions(frame, user_col="u", item_col="i", weight_col="w")
    assert report.rows_in == 6
    assert report.rows_out == 2
    assert report.duplicates_collapsed == 1
    assert report.dropped_negative_weight == 1
    assert report.dropped_missing_id == 1
    assert report.dropped_missing_weight == 1
    assert cleaned.loc[0, "weight"] == 15
    assert set(cleaned["item"]) == {"X", "Z"}


def test_missing_columns_raise():
    with pytest.raises(SchemaError):
        clean_interactions(pd.DataFrame({"user": ["a"]}), item_col="item", weight_col="w")


def test_pii_warning():
    frame = pd.DataFrame({"u": ["person@example.com"], "i": ["App"], "w": [10.0]})
    _, report = clean_interactions(frame, user_col="u", item_col="i", weight_col="w")
    assert report.suspected_pii_user_ids == 1
    assert report.warnings
