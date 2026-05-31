"""Forecast inference nodes: metadata-driven monthly champion forecasting.

Phase 6 makes monthly inference generic. A single node loads the production
champion (Prophet or SARIMAX) through generic champion artifacts, dispatches
prediction to the correct family adapter based on ``champion_monthly_metadata``,
and emits a standardized monthly forecast schema shared across families.

Weekly and daily inference remain future scope (see the stubs at the end of this
module); they are intentionally not implemented in this phase.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .adapters import SUPPORTED_MONTHLY_FAMILIES, dispatch_monthly_prediction

logger = logging.getLogger(__name__)

# Horizons (in months) produced by monthly inference.
_SUPPORTED_HORIZONS: tuple[int, ...] = (3, 6, 12)

# Champion metadata fields required before dispatch can proceed.
_REQUIRED_METADATA_KEYS: tuple[str, ...] = ("model_family", "champion_id")

# Canonical monthly forecast output schema, shared by every supported family.
_STANDARD_FORECAST_COLUMNS: list[str] = [
    "date",
    "forecast",
    "forecast_lower",
    "forecast_upper",
    "model_family",
    "granularity",
    "horizon",
    "horizon_label",
    "forecast_generated_at",
    "champion_id",
    "selection_metric",
    "selection_metric_value",
    "sku",
    "has_prediction_interval",
    "interval_method",
    "run_id",
    "source_dataset",
]


# ── Public node ───────────────────────────────────────────────────────────────


def generate_monthly_champion_forecasts(  # noqa: PLR0913
    champion_monthly_model: Any,
    champion_monthly_metadata: dict,
    monthly_future_3m: pd.DataFrame,
    monthly_future_6m: pd.DataFrame,
    monthly_future_12m: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate official monthly forecasts from the selected production champion.

    Loads the generic champion model and metadata, dispatches inference to the
    family-specific adapter named by ``champion_monthly_metadata["model_family"]``,
    standardizes the per-horizon output to the canonical monthly forecast schema,
    and returns one forecast table per horizon plus a latest forecast and an audit
    metadata artifact.

    The dispatch is driven by metadata, not by the Python type of the model object,
    so Prophet and SARIMAX champions flow through the same node.

    Args:
        champion_monthly_model: Production champion artifact. Prophet champions are
            the fitted model; SARIMAX champions are the candidate entry dict that
            carries the fitted results under ``"model"``.
        champion_monthly_metadata: Generic champion metadata from model selection.
            Must contain ``model_family`` and ``champion_id``.
        monthly_future_3m: Future feature frame for the 3-month horizon.
        monthly_future_6m: Future feature frame for the 6-month horizon.
        monthly_future_12m: Future feature frame for the 12-month horizon.
        params: Contents of ``forecast_inference.monthly`` from the parameter file.

    Returns:
        Five-element tuple ``(forecast_3m, forecast_6m, forecast_12m,
        forecast_latest, inference_metadata)``. Each forecast frame uses the
        canonical schema in ``_STANDARD_FORECAST_COLUMNS``; ``forecast_latest`` is a
        copy of the forecast for the configured ``default_horizon``;
        ``inference_metadata`` is a JSON-serialisable audit dict.

    Raises:
        ValueError: If champion metadata is invalid, the model family is
            unsupported, or any future frame fails validation.
    """
    _validate_monthly_champion_metadata(champion_monthly_metadata, params)

    model_family = str(champion_monthly_metadata["model_family"]).strip().lower()
    champion_id = str(champion_monthly_metadata["champion_id"])
    created_at = datetime.now(tz=UTC).isoformat()
    run_id = (
        f"monthly_{model_family}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:6]}"
    )

    logger.info(
        "Monthly champion inference — family=%s  champion=%s  run_id=%s",
        model_family,
        champion_id,
        run_id,
    )

    # The canonical generic future frames are not built yet, so monthly inference
    # temporarily consumes the Prophet future frames. The family adapters extract
    # the date index (and any exogenous columns) from these frames.
    # TODO(model_input): emit granularity-generic monthly_future_{3,6,12}m frames
    #   and route them here instead of the Prophet-specific frames.
    horizon_futures: dict[int, tuple[pd.DataFrame, str]] = {
        3: (monthly_future_3m, "monthly_future_3m"),
        6: (monthly_future_6m, "monthly_future_6m"),
        12: (monthly_future_12m, "monthly_future_12m"),
    }

    forecasts: dict[int, pd.DataFrame] = {}
    horizon_summaries: dict[int, dict] = {}

    for horizon, (future_df, source_dataset) in horizon_futures.items():
        _validate_future_frame(future_df, horizon, source_dataset)

        core = dispatch_monthly_prediction(
            model=champion_monthly_model,
            metadata=champion_monthly_metadata,
            future_df=future_df,
            params=params,
            horizon=horizon,
        )
        standardized = _standardize_forecast_schema(
            core_df=core,
            metadata=champion_monthly_metadata,
            table_horizon=horizon,
            run_id=run_id,
            created_at=created_at,
            source_dataset=source_dataset,
        )
        _validate_standard_forecast_output(standardized, horizon)

        forecasts[horizon] = standardized
        horizon_summaries[horizon] = _summarize_forecast(standardized, source_dataset)
        logger.info(
            "Horizon %dm — rows=%d  %s → %s",
            horizon,
            len(standardized),
            horizon_summaries[horizon]["start_date"],
            horizon_summaries[horizon]["end_date"],
        )

    default_horizon = _resolve_default_horizon(params, available=list(forecasts))
    forecast_latest = forecasts[default_horizon].copy()

    inference_metadata = _build_inference_metadata(
        metadata=champion_monthly_metadata,
        params=params,
        run_id=run_id,
        created_at=created_at,
        default_horizon=default_horizon,
        horizon_summaries=horizon_summaries,
        latest_forecast=forecast_latest,
        horizon_futures=horizon_futures,
    )

    logger.info(
        "Monthly inference complete — run_id=%s  latest_horizon=%dm  family=%s",
        run_id,
        default_horizon,
        model_family,
    )
    return (
        forecasts[3],
        forecasts[6],
        forecasts[12],
        forecast_latest,
        inference_metadata,
    )


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_monthly_champion_metadata(metadata: dict, params: dict) -> None:
    """Validate champion metadata before dispatch.

    Args:
        metadata: ``champion_monthly_metadata`` produced by model selection.
        params: Contents of ``forecast_inference.monthly`` (for supported families).

    Raises:
        ValueError: If a required field is missing or the family is unsupported.
    """
    missing = [k for k in _REQUIRED_METADATA_KEYS if not metadata.get(k)]
    if missing:
        raise ValueError(
            "champion_monthly_metadata is missing required field(s) "
            f"{missing}. Monthly inference cannot dispatch to a prediction adapter "
            f"without them. Present fields: {sorted(metadata.keys())}."
        )

    supported = [
        str(f).lower()
        for f in params.get("supported_families", SUPPORTED_MONTHLY_FAMILIES)
    ]
    model_family = str(metadata["model_family"]).strip().lower()
    if model_family not in supported:
        raise ValueError(
            f"Monthly champion model_family '{model_family}' is not supported. "
            f"Supported families: {supported}."
        )


def _validate_future_frame(future_df: pd.DataFrame, horizon: int, name: str) -> None:
    """Validate a future feature frame before family dispatch.

    Performs only granularity-agnostic checks; family-specific checks (regressors,
    exogenous columns) live in the adapters.

    Args:
        future_df: Future feature frame for a single horizon.
        horizon: Forecast horizon in months.
        name: Logical dataset name used in error messages.

    Raises:
        ValueError: If the frame is empty, lacks a date column, or leaks a target.
    """
    if future_df is None or future_df.empty:
        raise ValueError(
            f"Future frame '{name}' for the {horizon}-month horizon is empty."
        )

    date_candidates = ("ds", "month_start_date", "date")
    if not any(col in future_df.columns for col in date_candidates):
        raise ValueError(
            f"Future frame '{name}' has no recognisable date column "
            f"(looked for {date_candidates}); found {list(future_df.columns)}."
        )

    leaked_targets = [c for c in ("y", "monthly_demand") if c in future_df.columns]
    if leaked_targets:
        raise ValueError(
            f"Future frame '{name}' must not contain target column(s) {leaked_targets}; "
            "future inference inputs are feature-only."
        )


def _validate_standard_forecast_output(forecast_df: pd.DataFrame, horizon: int) -> None:
    """Validate a standardized forecast frame against the canonical contract.

    Args:
        forecast_df: Standardized forecast frame for one horizon.
        horizon: Forecast horizon in months (for error messages).

    Raises:
        ValueError: If required columns are absent, dates are unparseable, the point
            forecast is non-numeric, or interval columns are missing.
    """
    missing = [c for c in _STANDARD_FORECAST_COLUMNS if c not in forecast_df.columns]
    if missing:
        raise ValueError(
            f"Standardized {horizon}-month forecast is missing required columns: {missing}."
        )
    if forecast_df.empty:
        raise ValueError(f"Standardized {horizon}-month forecast is empty.")

    try:
        pd.to_datetime(forecast_df["date"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Standardized {horizon}-month forecast has unparseable 'date' values."
        ) from exc

    if not pd.api.types.is_numeric_dtype(forecast_df["forecast"]):
        raise ValueError(
            f"Standardized {horizon}-month forecast 'forecast' column must be numeric."
        )


# ── Schema and metadata helpers ───────────────────────────────────────────────


def _standardize_forecast_schema(  # noqa: PLR0913
    core_df: pd.DataFrame,
    metadata: dict,
    table_horizon: int,
    run_id: str,
    created_at: str,
    source_dataset: str,
) -> pd.DataFrame:
    """Decorate an adapter's core output with the canonical monthly forecast schema.

    Adds run/selection/traceability columns, guarantees nullable interval columns
    exist, computes the per-row 1-based ``horizon`` step, and enforces column order.

    Args:
        core_df: Core forecast frame returned by a family adapter.
        metadata: Champion metadata used for traceability columns.
        table_horizon: This table's horizon in months (3, 6, or 12).
        run_id: Unique inference run identifier.
        created_at: ISO-formatted generation timestamp.
        source_dataset: Logical name of the source future frame.

    Returns:
        Forecast frame with exactly ``_STANDARD_FORECAST_COLUMNS``.
    """
    df = core_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce").astype(float)

    for interval_col in ("forecast_lower", "forecast_upper"):
        if interval_col not in df.columns:
            df[interval_col] = np.nan

    if "has_prediction_interval" not in df.columns:
        df["has_prediction_interval"] = (
            df[["forecast_lower", "forecast_upper"]].notna().all(axis=1)
        )
    if "interval_method" not in df.columns:
        df["interval_method"] = None
    if "sku" not in df.columns:
        df["sku"] = None

    selection_metric, selection_value = _extract_selection_metric(metadata)

    df["model_family"] = str(metadata.get("model_family", "")).strip().lower()
    df["granularity"] = "monthly"
    df["horizon"] = _row_horizon(df["date"])
    df["horizon_label"] = f"{table_horizon}m"
    df["forecast_generated_at"] = created_at
    df["champion_id"] = str(metadata.get("champion_id", ""))
    df["selection_metric"] = selection_metric
    df["selection_metric_value"] = selection_value
    df["run_id"] = run_id
    df["source_dataset"] = source_dataset

    return df[_STANDARD_FORECAST_COLUMNS]


def _row_horizon(dates: pd.Series) -> list[int]:
    """Return 1-based month-ahead horizon indices ordered by date.

    horizon=1 is the earliest forecast period. Ties on date share an index.
    """
    sorted_unique = sorted(pd.to_datetime(dates).unique())
    rank = {d: i + 1 for i, d in enumerate(sorted_unique)}
    return [rank[d] for d in pd.to_datetime(dates)]


def _extract_selection_metric(metadata: dict) -> tuple[str, float | None]:
    """Resolve the production selection metric name and value from champion metadata."""
    selection = metadata.get("selection", {})
    metric = str(
        selection.get("primary_metric") or metadata.get("selection_metric") or ""
    )
    value: Any = None
    metrics = metadata.get("metrics", {})
    if metric and isinstance(metrics, dict) and metric in metrics:
        value = metrics.get(metric)
    if value is None:
        value = metadata.get("selection_metric_value")
    return metric, _safe_float(value)


def _resolve_default_horizon(params: dict, available: list[int]) -> int:
    """Resolve the latest-forecast horizon, falling back to the largest available."""
    default_horizon = int(params.get("default_horizon", max(available)))
    if default_horizon not in available:
        fallback = max(available)
        logger.warning(
            "default_horizon=%d is not among produced horizons %s; using %d.",
            default_horizon,
            available,
            fallback,
        )
        return fallback
    return default_horizon


def _summarize_forecast(forecast_df: pd.DataFrame, source_dataset: str) -> dict:
    """Build a compact per-horizon summary for the inference metadata artifact."""
    dates = pd.to_datetime(forecast_df["date"])
    return {
        "source_dataset": source_dataset,
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "rows": int(len(forecast_df)),
    }


def _build_inference_metadata(  # noqa: PLR0913
    metadata: dict,
    params: dict,
    run_id: str,
    created_at: str,
    default_horizon: int,
    horizon_summaries: dict[int, dict],
    latest_forecast: pd.DataFrame,
    horizon_futures: dict[int, tuple[pd.DataFrame, str]],
) -> dict:
    """Build the JSON-serialisable monthly inference audit artifact.

    Args:
        metadata: Champion metadata.
        params: Contents of ``forecast_inference.monthly``.
        run_id: Unique inference run identifier.
        created_at: ISO-formatted generation timestamp.
        default_horizon: Horizon used for the latest forecast.
        horizon_summaries: Per-horizon summary dicts.
        latest_forecast: Standardized latest forecast frame (for interval flags).
        horizon_futures: Mapping of horizon → (future frame, source dataset name).

    Returns:
        JSON-serialisable inference metadata dict.
    """
    model_family = str(metadata.get("model_family", "")).strip().lower()
    selection_metric, selection_value = _extract_selection_metric(metadata)

    has_interval = bool(latest_forecast["has_prediction_interval"].any())
    interval_methods = [
        m for m in latest_forecast["interval_method"].dropna().unique().tolist()
    ]
    interval_method = interval_methods[0] if interval_methods else None

    notes: list[str] = [
        "Monthly inference temporarily consumes Prophet future frames as the "
        "compatibility source; generic monthly_future_*m frames are the intended "
        "canonical input.",
    ]
    if model_family == "sarimax":
        notes.append(
            "SARIMAX forecasts are generated from the champion fitted results object; "
            "a full-history refit is recommended before production deployment."
        )

    return {
        "granularity": "monthly",
        "model_family": model_family,
        "champion_id": str(metadata.get("champion_id", "")),
        "run_id": run_id,
        "forecast_generated_at": created_at,
        "supported_horizons": list(_SUPPORTED_HORIZONS),
        "default_horizon": default_horizon,
        "output_schema_version": str(
            params.get("output_schema_version", "monthly_forecast_v1")
        ),
        "selection_metric": selection_metric,
        "selection_metric_value": selection_value,
        "has_prediction_interval": has_interval,
        "interval_method": interval_method,
        "horizons": {str(h): summary for h, summary in horizon_summaries.items()},
        "source_future_frames": {
            str(h): name for h, (_, name) in horizon_futures.items()
        },
        "source_champion_metadata_fields": sorted(metadata.keys()),
        "notes": notes,
    }


def _safe_float(value: Any) -> float | None:
    """Convert to a finite Python float, returning None for missing/non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


# ── Future inference stage stubs ──────────────────────────────────────────────


def generate_weekly_forecast(
    champion_weekly_model: Any,
    weekly_inference_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generate weekly demand forecasts using the elected weekly champion model.

    Args:
        champion_weekly_model: Serialised weekly champion model artifact.
        weekly_inference_df: Inference-ready weekly feature DataFrame.
        parameters: Weekly inference configuration.

    Returns:
        Standardized weekly forecast DataFrame.
    """
    raise NotImplementedError(
        "Weekly inference is future scope. Mirror the metadata-driven monthly "
        "dispatch for the weekly champion and emit granularity='weekly' outputs."
    )


def allocate_daily_forecast(
    forecast_weekly_reconciled: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Disaggregate reconciled weekly forecasts to daily estimates.

    Args:
        forecast_weekly_reconciled: Reconciled weekly forecast DataFrame.
        feature_weekly: Weekly feature DataFrame for historical day-of-week shares.
        parameters: Daily allocation configuration.

    Returns:
        Daily forecast DataFrame, or an empty DataFrame when allocation is disabled.
    """
    raise NotImplementedError(
        "Daily allocation is future scope. Distribute each reconciled week across "
        "7 days using historical day-of-week share fractions."
    )
