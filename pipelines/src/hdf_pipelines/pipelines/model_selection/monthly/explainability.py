"""Per-family explainability helpers for monthly family champions.

This module isolates the family-specific "driver importance" computation used by the
``generate_monthly_family_champion_explanations`` node so the heavy SHAP / statistics
logic stays out of the already-large ``nodes.py``. Each helper returns a tidy
``importance`` DataFrame with a consistent core schema so the app can render a single
feature-importance bar regardless of family. The ``importance_type`` column records how
each value was computed, because the statistic is *not* comparable across families:

- CatBoost: SHAP values via :class:`shap.TreeExplainer` → ``mean(|SHAP|)`` per feature.
- Prophet: ``model.predict`` component frame → mean absolute *centered* contribution
  (deviation from each component's own mean, so the trend level does not dominate).
- SARIMAX: fitted coefficients → ``|coefficient|`` per exogenous term, with the signed
  coefficient, standard error, and p-value retained for inspection.

All functions are pure: fitted models and frames in, DataFrames (and a SHAP explainer)
out. They never read the catalog or touch disk.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Unified long-form schema for the family-champion importance table. Optional
# family-specific columns (coefficient/std_err/pvalue) are present for every row and
# filled with NaN where they do not apply, so the parquet schema stays stable.
IMPORTANCE_COLUMNS: list[str] = [
    "family",
    "champion_id",
    "feature",
    "importance",
    "importance_type",
    "rank",
    "coefficient",
    "std_err",
    "pvalue",
    "computed_at",
]

# Prophet component columns considered as forecast drivers (when include_components).
_PROPHET_COMPONENTS: tuple[str, ...] = ("trend", "yearly", "weekly", "daily", "holidays")


def compute_catboost_shap_importance(
    model: Any,
    x_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Any, float]:
    """Compute SHAP values and global importance for a CatBoost champion.

    Uses :class:`shap.TreeExplainer`, which is exact for tree ensembles and needs no
    background dataset. Global importance is the mean absolute SHAP value per feature.

    Args:
        model: Fitted ``CatBoostRegressor`` (the champion's full-history refit).
        x_df: Feature matrix in the exact training column order; numeric only.
        feature_names: Ordered feature names matching ``x_df`` columns.

    Returns:
        Tuple ``(importance_df, shap_values_df, explainer, base_value)``:
        - ``importance_df``: columns ``feature``, ``importance`` (``mean(|SHAP|)``),
          ``importance_type`` (``"mean_abs_shap"``).
        - ``shap_values_df``: per-observation SHAP values, one column per feature.
        - ``explainer``: the fitted ``shap.TreeExplainer`` (persisted for reuse).
        - ``base_value``: the explainer's scalar expected value.
    """
    import shap  # local import keeps the heavy SHAP/numba import off pipeline startup

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_df)
    shap_arr = np.asarray(shap_values, dtype=float)

    # Normalise shape: some tree backends return (n, k, 1) or an extra base column.
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[..., 0]
    if shap_arr.shape[1] == len(feature_names) + 1:
        # CatBoost convention appends the base value as a trailing column.
        shap_arr = shap_arr[:, :-1]

    mean_abs = np.abs(shap_arr).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": mean_abs.astype(float),
            "importance_type": "mean_abs_shap",
        }
    )
    shap_values_df = pd.DataFrame(shap_arr, columns=list(feature_names))

    expected = explainer.expected_value
    base_value = float(np.ravel(np.asarray(expected, dtype=float))[0])

    return importance_df, shap_values_df, explainer, base_value


def compute_prophet_regressor_importance(
    model: Any,
    full_train_df: pd.DataFrame,
    active_regressors: list[str],
    include_components: bool = True,
    date_col: str = "ds",
) -> pd.DataFrame:
    """Compute driver importance for a Prophet champion from its component frame.

    Calls ``model.predict`` on the full-history frame and measures each driver's mean
    absolute *centered* contribution (deviation from its own mean). Centering keeps the
    additive trend's baseline level from dwarfing the regressor and seasonality effects,
    making the magnitudes comparable as "how much this driver moves the forecast".

    Args:
        model: Fitted Prophet champion model.
        full_train_df: Prophet-format full history (``ds`` + regressor columns).
        active_regressors: Regressor names the champion uses.
        include_components: When True, also include trend/seasonality/holiday components
            (bracketed feature names, e.g. ``[trend]``).
        date_col: Date column name in ``full_train_df``.

    Returns:
        DataFrame with ``feature``, ``importance``, ``importance_type``
        (``"mean_abs_contribution"``). Empty when no drivers are available.
    """
    regressors = [r for r in active_regressors if r in full_train_df.columns]
    prepared = full_train_df.copy()
    if date_col in prepared.columns:
        prepared[date_col] = pd.to_datetime(prepared[date_col])
    predict_cols = [date_col, *regressors] if date_col in prepared.columns else regressors
    forecast = model.predict(prepared[predict_cols])

    rows: list[dict[str, Any]] = []
    for name in regressors:
        if name in forecast.columns:
            rows.append(
                {
                    "feature": name,
                    "importance": _mean_abs_centered(forecast[name]),
                    "importance_type": "mean_abs_contribution",
                }
            )

    if include_components:
        for comp in _PROPHET_COMPONENTS:
            if comp in forecast.columns:
                rows.append(
                    {
                        "feature": f"[{comp}]",
                        "importance": _mean_abs_centered(forecast[comp]),
                        "importance_type": "mean_abs_contribution",
                    }
                )

    return pd.DataFrame(rows, columns=["feature", "importance", "importance_type"])


def compute_sarimax_coefficient_importance(
    results: Any,
    exog_names: list[str],
) -> pd.DataFrame:
    """Compute exogenous-driver importance for a SARIMAX champion from its coefficients.

    SARIMAX fit on numpy arrays names exogenous parameters ``x1..xk``; these are mapped
    back to readable column names by position. Structural terms (AR/MA/trend/sigma2) are
    intentionally excluded — they are not exogenous demand drivers and mixing their
    scales would make the importance bar misleading. Importance is ``|coefficient|``.

    Args:
        results: Fitted statsmodels SARIMAX results object.
        exog_names: Ordered exogenous column names used in the fit (may be empty).

    Returns:
        DataFrame with ``feature``, ``importance`` (``|coef|``), ``importance_type``
        (``"abs_coefficient"``), ``coefficient``, ``std_err``, ``pvalue``. Empty when the
        champion uses no exogenous regressors.
    """
    param_names = [str(n) for n in getattr(results, "param_names", [])]
    if not param_names:
        return pd.DataFrame(
            columns=["feature", "importance", "importance_type", "coefficient", "std_err", "pvalue"]
        )

    params = np.asarray(results.params, dtype=float)
    bse = np.asarray(getattr(results, "bse", np.full(len(param_names), np.nan)), dtype=float)
    pvalues = np.asarray(
        getattr(results, "pvalues", np.full(len(param_names), np.nan)), dtype=float
    )

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(param_names):
        match = re.fullmatch(r"x(\d+)", name)
        if not match:
            continue
        pos = int(match.group(1)) - 1
        feature = exog_names[pos] if 0 <= pos < len(exog_names) else name
        coef = float(params[i])
        rows.append(
            {
                "feature": feature,
                "importance": abs(coef),
                "importance_type": "abs_coefficient",
                "coefficient": coef,
                "std_err": float(bse[i]) if i < len(bse) else float("nan"),
                "pvalue": float(pvalues[i]) if i < len(pvalues) else float("nan"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["feature", "importance", "importance_type", "coefficient", "std_err", "pvalue"],
    )


def assemble_family_importance_table(
    per_family: dict[str, dict],
    computed_at: str,
) -> pd.DataFrame:
    """Combine per-family importance frames into one ranked long-form table.

    Args:
        per_family: Mapping ``family -> {"champion_id": str, "importance": DataFrame}``.
            The importance DataFrame must carry at least ``feature``, ``importance``,
            ``importance_type``.
        computed_at: ISO timestamp stamped on every row.

    Returns:
        Long-form DataFrame with :data:`IMPORTANCE_COLUMNS`. Within each family, rows are
        ranked by descending importance (rank 1 = most important). Empty when no family
        produced importance values.
    """
    frames: list[pd.DataFrame] = []
    for family, info in per_family.items():
        importance = info.get("importance")
        if importance is None or importance.empty:
            continue
        ordered = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        ordered.insert(0, "family", family)
        ordered.insert(1, "champion_id", str(info.get("champion_id", "")))
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        ordered["computed_at"] = computed_at
        frames.append(ordered)

    if not frames:
        return pd.DataFrame(columns=IMPORTANCE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    for col in ("coefficient", "std_err", "pvalue"):
        if col not in out.columns:
            out[col] = np.nan
    return out[IMPORTANCE_COLUMNS]


def _mean_abs_centered(series: pd.Series) -> float:
    """Return the mean absolute deviation of a series from its own mean."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return 0.0
    centered = values - np.nanmean(values)
    return float(np.nanmean(np.abs(centered)))
