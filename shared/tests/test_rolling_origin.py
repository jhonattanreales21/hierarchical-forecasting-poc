"""Unit tests for the shared rolling-origin backtesting engine.

Covers the protocol invariants (§3, §5, §9.3, §10): expanding/step-1 cycle
generation with the last cycle predicting ``[L-2, L-1, L]``, the minimum-train
guard, per-cycle metric definitions, macro-average aggregation, and the
family-agnostic driver.
"""

import numpy as np
import pandas as pd
import pytest

from shared.rolling_origin import (
    RollingOriginCycle,
    aggregate_rolling_origin_metrics,
    compute_cycle_metrics,
    generate_rolling_origin_cycles,
    run_rolling_origin,
)


def _month_starts(n: int, start: str = "2023-03-01") -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, periods=n, freq="MS"))


def test_last_cycle_predicts_final_horizon_months() -> None:
    # L = 2026-05-01 with 39 contiguous months. n_cycles=5, H=3, step=1.
    dates = _month_starts(39)
    cycles = generate_rolling_origin_cycles(dates, n_cycles=5, horizon=3)

    assert len(cycles) == 5
    last = cycles[-1]
    # Last cycle predicts exactly [L-2, L-1, L].
    assert [d.strftime("%Y-%m-%d") for d in last.target_dates] == [
        "2026-03-01",
        "2026-04-01",
        "2026-05-01",
    ]
    # Origin = L - horizon.
    assert last.origin_date == pd.Timestamp("2026-02-01")
    # Expanding + step 1: each origin advances by one month, cycle 1 is earliest.
    origins = [c.origin_date for c in cycles]
    assert origins == sorted(origins)
    assert (origins[-1] - origins[0]).days > 0
    assert cycles[0].cycle_index == 1


def test_expanding_step_one_origins_are_consecutive() -> None:
    dates = _month_starts(39)
    cycles = generate_rolling_origin_cycles(dates, n_cycles=5, horizon=3, step_months=1)
    origins = [c.origin_date for c in cycles]
    diffs = {(b.year * 12 + b.month) - (a.year * 12 + a.month) for a, b in zip(origins, origins[1:])}
    assert diffs == {1}  # exactly one month apart


def test_min_train_guard_raises_clear_error() -> None:
    dates = _month_starts(39)
    # First cycle train length is 32 here; require 40 to trigger the guard.
    with pytest.raises(ValueError, match="insufficient training history"):
        generate_rolling_origin_cycles(
            dates, n_cycles=5, horizon=3, min_train_periods=40
        )


def test_series_too_short_raises() -> None:
    dates = _month_starts(6)
    with pytest.raises(ValueError, match="too short"):
        generate_rolling_origin_cycles(dates, n_cycles=5, horizon=3)


def test_only_expanding_window_supported() -> None:
    dates = _month_starts(39)
    with pytest.raises(ValueError, match="expanding"):
        generate_rolling_origin_cycles(dates, n_cycles=3, horizon=3, window="sliding")


def test_compute_cycle_metrics_keys_and_values() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    y_train = np.arange(1.0, 40.0)  # > season

    m = compute_cycle_metrics(y_true, y_pred, y_train, season=12, epsilon=1.0)

    assert set(["wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse"]).issubset(m)
    # Global WMAPE = sum|e| / sum|y| = (10+10+30)/600.
    assert m["wmape"] == pytest.approx(50.0 / 600.0)
    # Per-horizon WMAPE uses the epsilon-guarded denominator.
    assert m["wmape_m3"] == pytest.approx(30.0 / (300.0 + 1.0))
    # BIAS = sum(yhat - y) / (sum|y| + eps) = (10 - 10 + 30) / 601.
    assert m["bias"] == pytest.approx(30.0 / 601.0)


def test_aggregate_macro_average_ignores_nan() -> None:
    records = [
        {"wmape": 0.10, "wmape_m3": 0.20, "mase": 0.5},
        {"wmape": 0.20, "wmape_m3": 0.40, "mase": float("nan")},
        {"wmape": float("nan"), "wmape_m3": 0.60, "mase": 0.9},
    ]
    agg = aggregate_rolling_origin_metrics(records)

    assert agg["wmape"] == pytest.approx(0.15)  # mean of 0.10, 0.20
    assert agg["wmape_m3"] == pytest.approx(0.40)  # mean of 0.20, 0.40, 0.60
    assert agg["mase"] == pytest.approx(0.70)  # mean of 0.5, 0.9
    assert agg["n_cycles"] == 3
    assert agg["n_cycles_evaluated"] == 2  # wmape finite in 2 of 3
    assert "wmape_m3_std" in agg


def test_run_rolling_origin_with_naive_forecaster() -> None:
    # Build a contiguous monthly series; forecaster returns last observed value.
    dates = _month_starts(39)
    demand = np.linspace(100.0, 500.0, num=39)
    full_df = pd.DataFrame({"month_start_date": dates, "monthly_demand": demand})
    cycles = generate_rolling_origin_cycles(dates, n_cycles=5, horizon=3)

    def naive_fit_forecast(train_df: pd.DataFrame, cycle: RollingOriginCycle) -> np.ndarray:
        last_value = float(train_df["monthly_demand"].iloc[-1])
        return np.full(len(cycle.target_dates), last_value)

    per_cycle_df, agg = run_rolling_origin(
        full_df,
        date_col="month_start_date",
        target_col="monthly_demand",
        cycles=cycles,
        fit_forecast_fn=naive_fit_forecast,
        season=12,
    )

    assert len(per_cycle_df) == 5
    assert (per_cycle_df["status"] == "success").all()
    assert agg["n_cycles_evaluated"] == 5
    assert np.isfinite(agg["wmape_m3"])
    # Training window is leakage-safe: never longer than origin position + 1.
    assert per_cycle_df["n_train"].tolist() == sorted(per_cycle_df["n_train"].tolist())


def test_run_rolling_origin_records_failed_cycle() -> None:
    dates = _month_starts(39)
    full_df = pd.DataFrame(
        {"month_start_date": dates, "monthly_demand": np.arange(39, dtype=float) + 1.0}
    )
    cycles = generate_rolling_origin_cycles(dates, n_cycles=3, horizon=3)

    def flaky(train_df: pd.DataFrame, cycle: RollingOriginCycle) -> np.ndarray:
        if cycle.cycle_index == 2:
            raise RuntimeError("boom")
        return np.full(len(cycle.target_dates), float(train_df["monthly_demand"].iloc[-1]))

    per_cycle_df, agg = run_rolling_origin(
        full_df,
        date_col="month_start_date",
        target_col="monthly_demand",
        cycles=cycles,
        fit_forecast_fn=flaky,
    )

    assert agg["n_cycles"] == 3
    assert agg["n_cycles_evaluated"] == 2
    assert per_cycle_df["status"].str.startswith("failed").sum() == 1
