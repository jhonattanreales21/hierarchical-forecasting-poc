"""Unit tests for shared upload validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from shared.upload_validation import (
    detect_granularity,
    find_date_column,
    summarize_csv,
)


def test_find_date_column_exact_match():
    df = pd.DataFrame({"Date": ["2023-01"], "value": [1]})
    assert find_date_column(df) == "Date"


def test_find_date_column_substring_match():
    df = pd.DataFrame({"order_date": ["2023-01-01"], "value": [1]})
    assert find_date_column(df) == "order_date"


def test_find_date_column_absent():
    df = pd.DataFrame({"value": [1], "qty": [2]})
    assert find_date_column(df) is None


def test_detect_granularity_daily():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    assert detect_granularity(pd.Series(dates)) == "daily"


def test_detect_granularity_weekly():
    dates = pd.date_range("2023-01-01", periods=20, freq="W")
    assert detect_granularity(pd.Series(dates)) == "weekly"


def test_detect_granularity_monthly():
    dates = pd.date_range("2023-01-01", periods=18, freq="MS")
    assert detect_granularity(pd.Series(dates)) == "monthly"


def test_detect_granularity_monthly_from_year_month_strings():
    dates = pd.Series(["2023-01", "2023-02", "2023-03", "2023-04"])
    assert detect_granularity(dates) == "monthly"


def test_detect_granularity_unknown_with_single_date():
    assert detect_granularity(pd.Series(["2023-01-01"])) == "unknown"


def test_detect_granularity_irregular():
    dates = pd.Series(["2023-01-01", "2023-04-15", "2023-12-31"])
    assert detect_granularity(dates) == "irregular"


def test_summarize_csv_monthly():
    df = pd.DataFrame(
        {
            "Date": ["2023-01", "2023-02", "2023-03"],
            "demand": [10, 20, 30],
            "sku": ["sku1", "sku1", "sku1"],
        }
    )
    summary = summarize_csv(df)
    assert summary["n_rows"] == 3
    assert summary["n_columns"] == 3
    assert summary["date_column"] == "Date"
    assert summary["granularity"] == "monthly"
    assert summary["date_min"] == "2023-01-01"
    assert summary["date_max"] == "2023-03-01"


def test_summarize_csv_without_date_column():
    df = pd.DataFrame({"value": [1, 2], "qty": [3, 4]})
    summary = summarize_csv(df)
    assert summary["date_column"] is None
    assert summary["granularity"] == "unknown"
    assert summary["date_min"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
