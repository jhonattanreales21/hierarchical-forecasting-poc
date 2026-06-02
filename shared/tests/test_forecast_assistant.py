import pandas as pd

from shared.forecast_assistant import (
    detect_period,
    is_in_scope,
    rename_business_columns,
    scenario_label,
    select_historical_context,
    transform_scenario_rows,
)


def test_detect_period_month_and_year():
    detected = detect_period("Why did demand spike in Jul 2025?")

    assert detected.years == [2025]
    assert detected.month == 7


def test_select_historical_context_uses_month_window():
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2025-06-01", periods=3, freq="MS"),
            "y": [10, 30, 12],
        }
    )

    result = select_historical_context("Explain July 2025", df)

    assert result["month"].dt.strftime("%Y-%m").tolist() == [
        "2025-06",
        "2025-07",
        "2025-08",
    ]


def test_select_historical_context_year_uses_highest_demand_months():
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "y": [10, 90, 30, 80, 20, 70],
        }
    )

    result = select_historical_context("What happened in 2025?", df, max_rows=3)

    assert set(result["y"]) == {70, 80, 90}


def test_transform_scenario_rows_uses_business_language():
    forecast = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2026-06-01")],
            "expected_market_share": [1.0],
            "pfizer_limited": [1],
            "surgifoam_limited": [0],
            "rebate_target": [0],
            "yhat": [1451.7],
        }
    )

    result = transform_scenario_rows(forecast)

    assert result.loc[0, "expected market share"] == "100%"
    assert result.loc[0, "market-share scenario label"] == "severe competitor-stockout scenario"
    assert result.loc[0, "competitor supply constraint effect"] == "active"


def test_scope_guardrail_rejects_unrelated_questions():
    assert not is_in_scope("Can you write me a poem?")


def test_scenario_label_for_normal_share():
    assert scenario_label(0.5) == "normalized market-share assumption"


def test_rename_business_columns_coalesces_duplicate_month_columns():
    df = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2026-06-01")],
            "month": [pd.Timestamp("2026-06-01")],
            "yhat": [10.2],
        }
    )

    result = rename_business_columns(df)

    assert result.columns.tolist().count("month") == 1
    assert result.loc[0, "month"] == "2026-06"
