"""Recursive target-feature construction for CatBoost multi-step forecasting.

These helpers rebuild target-derived features (lags, rolling statistics, and trend
diffs/pct-changes) step-by-step from a demand buffer of observed values plus prior
predictions. They are shared by two consumers so the recursion is identical
everywhere:

* ``forecast_inference`` — production monthly CatBoost inference.
* ``model_selection`` — rolling-origin (operational lead-time) M-h evaluation of
  CatBoost candidates on the held-out test set.

Keeping a single implementation guarantees the operational-lead-time metrics used
for champion selection reflect exactly how the CatBoost champion forecasts in
production (predicted lags feed later steps), rather than an optimistic one-shot
forecast that uses actual lags.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_demand_buffer(
    history_df: pd.DataFrame | None,
    target_col: str,
    date_col: str,
) -> list[float]:
    """Build the recursive demand buffer from a historical CatBoost-ready DataFrame.

    Returns an ordered list of observed demand float values (oldest first).
    An empty list is returned when ``history_df`` is None or lacks the target column.
    """
    if history_df is None or history_df.empty:
        logger.warning(
            "CatBoost recursion: history_df is empty or None. "
            "All recursive lag and rolling features will be NaN."
        )
        return []

    if target_col not in history_df.columns:
        logger.warning(
            "CatBoost recursion: history_df does not contain target column %r; "
            "using empty demand buffer.",
            target_col,
        )
        return []

    sort_col = date_col if date_col in history_df.columns else None
    sorted_df = history_df.sort_values(sort_col) if sort_col else history_df
    return [float(v) for v in sorted_df[target_col].dropna().to_numpy()]


def compute_recursive_target_features(  # noqa: PLR0913
    demand_buffer: list[float],
    target_lags: list[int],
    rolling_windows: list[int],
    include_std: bool,
    include_min_max: bool,
    trend_diffs: list[int],
    trend_pct_changes: list[int],
) -> dict[str, float]:
    """Compute all target-derived features for the next forecast step.

    Mirrors the training-time feature engineering in
    ``_compute_catboost_target_features`` (model_input_preparation):

    * ``demand_lag_k``        = ``buffer[-k]``   (value k periods before current)
    * ``rolling_mean_w``      = mean of ``buffer[-w:]``  (w periods ending at current-1)
    * ``rolling_std_w``       = ddof-1 std of the same window
    * ``rolling_min/max_w``   = min/max of the same window
    * ``rolling_mean_3_vs_12``= rolling_mean_3 / rolling_mean_12
    * ``demand_diff_p``       = ``buffer[-1] - buffer[-(p+1)]``
    * ``demand_pct_change_p`` = ``(buffer[-1] - buffer[-(p+1)]) / buffer[-(p+1)]``

    At step h, ``demand_buffer`` contains all observed and previously predicted
    values up to period T+h-1. ``buffer[-k]`` therefore equals demand at T+h-k.

    NaN is returned for any feature requiring more history than is available,
    matching the lag-warmup behaviour during training.
    """
    features: dict[str, float] = {}
    n = len(demand_buffer)

    for lag in target_lags:
        features[f"demand_lag_{lag}"] = float(demand_buffer[-lag]) if n >= lag else np.nan

    rolling_means: dict[int, float] = {}
    for w in rolling_windows:
        if n >= w:
            arr = np.array(demand_buffer[-w:], dtype=float)
            mean_val = float(np.mean(arr))
            features[f"rolling_mean_{w}"] = mean_val
            rolling_means[w] = mean_val
            if include_std:
                # ddof=1 matches pandas rolling().std() default
                features[f"rolling_std_{w}"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
            if include_min_max:
                features[f"rolling_min_{w}"] = float(np.min(arr))
                features[f"rolling_max_{w}"] = float(np.max(arr))
        else:
            features[f"rolling_mean_{w}"] = np.nan
            rolling_means[w] = np.nan
            if include_std:
                features[f"rolling_std_{w}"] = np.nan
            if include_min_max:
                features[f"rolling_min_{w}"] = np.nan
                features[f"rolling_max_{w}"] = np.nan

    if 3 in rolling_windows and 12 in rolling_windows:
        rm3 = rolling_means.get(3, np.nan)
        rm12 = rolling_means.get(12, np.nan)
        if (
            not np.isnan(rm3)
            and not np.isnan(rm12)
            and rm12 != 0.0
        ):
            features["rolling_mean_3_vs_12"] = rm3 / rm12
        else:
            features["rolling_mean_3_vs_12"] = np.nan

    for period in trend_diffs:
        if n > period:
            features[f"demand_diff_{period}"] = (
                float(demand_buffer[-1]) - float(demand_buffer[-(period + 1)])
            )
        else:
            features[f"demand_diff_{period}"] = np.nan

    for period in trend_pct_changes:
        if n > period:
            prev = float(demand_buffer[-(period + 1)])
            curr = float(demand_buffer[-1])
            features[f"demand_pct_change_{period}"] = (
                (curr - prev) / prev if prev != 0.0 and not np.isnan(prev) else np.nan
            )
        else:
            features[f"demand_pct_change_{period}"] = np.nan

    return features


def extract_periods_from_column_names(column_names: list[str], prefix: str) -> list[int]:
    """Extract integer period suffixes from column names like ``demand_diff_12``."""
    periods: list[int] = []
    for col in column_names:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            try:
                periods.append(int(suffix))
            except ValueError:
                pass
    return sorted(set(periods))
