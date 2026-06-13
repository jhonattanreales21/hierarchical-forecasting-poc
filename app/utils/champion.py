"""Model-family-agnostic champion and forecast normalization helpers.

This module centralises the logic that lets the Streamlit app consume the
current monthly *production champion* without assuming a fixed model family
(Prophet, SARIMAX, or CatBoost). It deliberately avoids importing Streamlit so
the functions stay pure and unit-testable.

The app reads several artifacts whose key names have drifted between the legacy
Prophet-only contract and the current generic monthly contract. The helpers here
extract a single normalized "champion identity" with defensive fallbacks where
they do not reintroduce deprecated metric names.
"""

from typing import Any, Optional

import pandas as pd

# Canonical generic monthly forecast columns -> internal chart columns.
_FORECAST_RENAME_MAP: dict[str, str] = {
    "date": "ds",
    "forecast": "yhat",
    "forecast_lower": "yhat_lower",
    "forecast_upper": "yhat_upper",
}

_LOWER_BOUND_COLS = ("yhat_lower", "forecast_lower")
_UPPER_BOUND_COLS = ("yhat_upper", "forecast_upper")


def standardize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize current and legacy forecast output schemas for the app.

    The current monthly forecast artifacts expose generic columns
    (``date``, ``forecast``, ``forecast_lower``, ``forecast_upper``); legacy
    Prophet artifacts already use ``ds``/``yhat``. This maps either onto the
    internal ``ds``/``yhat`` shape expected by the chart component without
    implying the project is Prophet-shaped.

    Args:
        df: Raw forecast DataFrame from any monthly forecast artifact.

    Returns:
        DataFrame with ``ds`` parsed as datetime where present.
    """
    if df is None or df.empty:
        return df
    out = df.rename(
        columns={k: v for k, v in _FORECAST_RENAME_MAP.items() if k in df.columns}
    )
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"])
    return out


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None (treating NaN as None)."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        return value
    return None


def _summary_value(summary: Optional[pd.DataFrame], column: str) -> Any:
    """Safely pull a scalar from the single-row selection summary DataFrame."""
    if summary is None or summary.empty or column not in summary.columns:
        return None
    value = summary[column].iloc[0]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def standardize_champion_metadata(meta: dict) -> dict:
    """Normalize current and legacy champion metadata keys.

    Current metadata stores rolling-origin metrics under ``metrics``; legacy metadata
    used ``test_metrics``. A model-family-agnostic ``forecast_precision``
    (``1 - WMAPE``) and a ``business_success_flag`` are derived when absent so the
    app's precision KPIs work regardless of which family won.

    Args:
        meta: Raw champion metadata dict (may be empty).

    Returns:
        Metadata dict with ``evaluation_metrics`` populated and derived business flags.
    """
    if not meta:
        return meta
    out = dict(meta)
    metrics = dict(
        out.get("evaluation_metrics") or out.get("metrics") or out.get("test_metrics") or {}
    )

    wmape = metrics.get("wmape")
    if "forecast_precision" not in metrics and wmape is not None:
        try:
            metrics["forecast_precision"] = 1.0 - float(wmape)
        except (TypeError, ValueError):
            pass

    out["evaluation_metrics"] = metrics

    precision_threshold = out.get("business_success_precision_threshold", 0.85)
    out["business_success_precision_threshold"] = precision_threshold
    if "business_success_flag" not in out and wmape is not None:
        try:
            out["business_success_flag"] = float(wmape) <= (1.0 - precision_threshold)
        except (TypeError, ValueError):
            out["business_success_flag"] = False
    return out


def extract_champion_identity(
    meta: dict,
    inference_meta: dict,
    selection_summary: Optional[pd.DataFrame] = None,
) -> dict:
    """Build a normalized, model-family-agnostic champion identity.

    Pulls champion family, id, selection metric, evaluation mode, refit status,
    run provenance, hyperparameters, active regressors, and interval availability
    from whichever artifact exposes them, using defensive fallbacks rather than
    assuming one metadata shape.

    Args:
        meta: Champion metadata dict (``champion_monthly_metadata.json``).
        inference_meta: Inference run metadata dict.
        selection_summary: Optional single-row production selection summary.

    Returns:
        Dict with normalized champion-identity fields. Missing values are ``None``.
    """
    meta = meta or {}
    inf = inference_meta or {}
    selection = meta.get("selection", {}) or {}
    metrics = (
        meta.get("evaluation_metrics")
        or meta.get("metrics")
        or meta.get("test_metrics")
        or {}
    )
    contract = meta.get("inference_contract", {}) or {}

    model_family = _first_not_none(
        meta.get("model_family"),
        inf.get("model_family"),
        _summary_value(selection_summary, "production_champion_family"),
    )
    champion_id = _first_not_none(
        meta.get("champion_id"),
        inf.get("champion_id"),
        _summary_value(selection_summary, "production_champion_id"),
    )
    selection_metric = _first_not_none(
        selection.get("primary_metric"),
        inf.get("selection_metric"),
        meta.get("selection_metric"),
        _summary_value(selection_summary, "primary_metric"),
    )
    selection_metric_value = _first_not_none(
        metrics.get(selection_metric) if selection_metric else None,
        inf.get("selection_metric_value"),
        selection.get("primary_metric_value"),
        _summary_value(selection_summary, "primary_metric_value"),
    )

    refit = meta.get("refit", {}) or {}

    forecast_generated_at = _first_not_none(
        inf.get("forecast_generated_at"),
        inf.get("forecast_created_at"),
        meta.get("forecast_generated_at"),
    )
    run_id = _first_not_none(inf.get("run_id"), meta.get("run_id"))

    supported_horizons = inf.get("supported_horizons")
    if not supported_horizons and inf.get("horizons"):
        try:
            supported_horizons = sorted(int(h) for h in inf["horizons"].keys())
        except (TypeError, ValueError):
            supported_horizons = None
    if not supported_horizons:
        supported_horizons = contract.get("forecast_horizons")

    has_prediction_interval = _first_not_none(
        inf.get("has_prediction_interval"),
        contract.get("has_prediction_intervals"),
        meta.get("has_prediction_interval"),
    )
    interval_method = _first_not_none(
        inf.get("interval_method"),
        contract.get("interval_method"),
        meta.get("interval_method"),
    )

    return {
        "model_family": model_family,
        "champion_id": champion_id,
        "champion_level": meta.get("champion_level"),
        "selection_metric": selection_metric,
        "selection_metric_value": selection_metric_value,
        "selection_reason": selection.get("selection_reason")
        or _summary_value(selection_summary, "selection_reason"),
        "selected_at": _first_not_none(
            selection.get("selected_at"),
            meta.get("selected_at"),
            _summary_value(selection_summary, "selection_timestamp"),
        ),
        "evaluation": meta.get("evaluation", {}) or {},
        "evaluation_metrics": metrics,
        "hyperparameters": meta.get("hyperparameters") or {},
        "active_regressors": _first_not_none(
            meta.get("active_regressors"),
            contract.get("active_regressors"),
        )
        or [],
        "training_cutoff": meta.get("training_cutoff"),
        "refit": refit,
        "forecast_generated_at": forecast_generated_at,
        "run_id": run_id,
        "supported_horizons": supported_horizons or [],
        "has_prediction_interval": has_prediction_interval,
        "interval_method": interval_method,
        "active_families": _summary_value(selection_summary, "active_families"),
    }


def family_label(model_family: Optional[str]) -> str:
    """Return a human-friendly label for a model family, or a generic fallback."""
    if not model_family:
        return "production champion"
    pretty = {"prophet": "Prophet", "sarimax": "SARIMAX", "catboost": "CatBoost"}
    return pretty.get(str(model_family).lower(), str(model_family))


def forecast_has_intervals(future_fc: pd.DataFrame, identity: dict) -> bool:
    """Decide whether prediction-interval bands should be rendered.

    Intervals are only shown when metadata does not deny them, both bound columns
    exist, and the bounds are neither entirely null nor a zero-width degenerate
    band. This keeps interval language and shaded bands out of the UI for
    champions that do not produce intervals.

    Args:
        future_fc: Future forecast DataFrame (standardized or raw).
        identity: Normalized champion identity from ``extract_champion_identity``.

    Returns:
        True if interval bands are safe and meaningful to render.
    """
    if future_fc is None or future_fc.empty:
        return False
    if identity.get("has_prediction_interval") is False:
        return False

    lower_col = next((c for c in _LOWER_BOUND_COLS if c in future_fc.columns), None)
    upper_col = next((c for c in _UPPER_BOUND_COLS if c in future_fc.columns), None)
    if lower_col is None or upper_col is None:
        return False

    lower = pd.to_numeric(future_fc[lower_col], errors="coerce")
    upper = pd.to_numeric(future_fc[upper_col], errors="coerce")
    if lower.isna().all() or upper.isna().all():
        return False
    if float((upper.fillna(0) - lower.fillna(0)).abs().sum()) == 0.0:
        return False
    return True
