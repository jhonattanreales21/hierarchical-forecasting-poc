"""Forecast inference nodes: generate forward-looking predictions from champion models."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Keys required to be present in champion metadata before inference can proceed.
_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "champion_id",
    "active_regressors",
    "model_family",
)

# Standardised column order for all monthly Prophet forecast outputs.
_FORECAST_COLUMN_ORDER: list[str] = [
    "ds",
    "sku",
    "horizon_month",
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "model_family",
    "model_granularity",
    "champion_id",
    "forecast_run_id",
    "forecast_created_at",
    "forecast_horizon_months",
    "selection_metric",
    "selection_metric_value",
    "business_success_flag",
    "source_dataset",
]


# ── Public node ───────────────────────────────────────────────────────────────


def generate_monthly_prophet_forecasts(  # noqa: PLR0912, PLR0913
    monthly_prophet_champion_model: Any,
    monthly_prophet_champion_metadata: dict,
    monthly_prophet_future_3m: pd.DataFrame,
    monthly_prophet_future_6m: pd.DataFrame,
    monthly_prophet_future_12m: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate official monthly demand forecasts using the selected Prophet champion.

    Loads the champion model and metadata, validates future regressor inputs for each
    configured horizon, runs Prophet inference, annotates outputs with traceability
    metadata, and returns one forecast table per horizon plus a latest forecast and
    an inference metadata artifact.

    Args:
        monthly_prophet_champion_model: Fitted Prophet model from Stage 5 model selection.
        monthly_prophet_champion_metadata: Champion metadata dict from Stage 5, must
            include champion_id, active_regressors, model_family, selection_metric,
            selection_metric_value, and business_success_flag.
        monthly_prophet_future_3m: Future feature DataFrame for the 3-month horizon.
        monthly_prophet_future_6m: Future feature DataFrame for the 6-month horizon.
        monthly_prophet_future_12m: Future feature DataFrame for the 12-month horizon.
        params: Contents of forecast_inference.monthly_prophet from the parameter file.

    Returns:
        Five-element tuple:

        1. ``forecast_3m`` — Annotated forecast DataFrame for the 3-month horizon.
        2. ``forecast_6m`` — Annotated forecast DataFrame for the 6-month horizon.
        3. ``forecast_12m`` — Annotated forecast DataFrame for the 12-month horizon.
        4. ``forecast_latest`` — Copy of the forecast for the configured
           ``latest_output_horizon_months`` (defaults to 12). Used by downstream consumers
           (Streamlit, FastAPI, reporting) as the single authoritative forecast table.
        5. ``inference_metadata`` — JSON-serialisable dict summarising the inference run.

    Raises:
        ValueError: If champion metadata is missing required keys, or if any future
            dataset fails validation.
    """
    _validate_champion_metadata(monthly_prophet_champion_metadata)

    champion_id: str = monthly_prophet_champion_metadata["champion_id"]
    active_regressors: list[str] = list(
        monthly_prophet_champion_metadata["active_regressors"]
    )
    date_col: str = params.get("date_column", "ds")
    sku_col: str = params.get("sku_column", "sku")
    output_cfg: dict = params.get("output", {})
    val_cfg: dict = params.get("validation", {})
    latest_horizon: int = int(output_cfg.get("latest_output_horizon_months", 6))

    logger.info(
        "Starting Monthly Prophet forecast inference — champion: %s", champion_id
    )
    logger.info("Active regressors (%d): %s", len(active_regressors), active_regressors)

    forecast_run_id = (
        f"monthly_prophet_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:6]}"
    )
    created_at = datetime.now(tz=UTC).isoformat()

    horizon_futures: dict[int, tuple[pd.DataFrame, str]] = {
        3: (monthly_prophet_future_3m, "monthly_prophet_future_3m"),
        6: (monthly_prophet_future_6m, "monthly_prophet_future_6m"),
        12: (monthly_prophet_future_12m, "monthly_prophet_future_12m"),
    }

    forecasts: dict[int, pd.DataFrame] = {}
    horizon_summaries: dict[int, dict] = {}

    for horizon_months, (future_df, dataset_name) in horizon_futures.items():
        logger.info(
            "Validating future dataset '%s' — shape: %s",
            dataset_name,
            future_df.shape,
        )
        _validate_future_dataset(
            future_df=future_df,
            active_regressors=active_regressors,
            expected_horizon=horizon_months,
            dataset_name=dataset_name,
            date_col=date_col,
            sku_col=sku_col,
            params=val_cfg,
        )

        forecast_df = _forecast_one_horizon(
            model=monthly_prophet_champion_model,
            future_df=future_df,
            active_regressors=active_regressors,
            champion_metadata=monthly_prophet_champion_metadata,
            forecast_run_id=forecast_run_id,
            created_at=created_at,
            horizon_months=horizon_months,
            source_dataset=dataset_name,
            date_col=date_col,
            sku_col=sku_col,
            output_cfg=output_cfg,
        )

        forecasts[horizon_months] = forecast_df
        horizon_summaries[horizon_months] = _summarize_forecast_output(
            forecast_df, "ds", dataset_name
        )

        logger.info(
            "Horizon %dm — rows: %d | %s → %s",
            horizon_months,
            len(forecast_df),
            horizon_summaries[horizon_months]["start_date"],
            horizon_summaries[horizon_months]["end_date"],
        )

    forecast_3m = forecasts[3]
    forecast_6m = forecasts[6]
    forecast_12m = forecasts[12]

    # forecast_latest mirrors the configured latest_output_horizon_months (default 12)
    # so downstream consumers always have one canonical table to query.
    if latest_horizon not in forecasts:
        raise ValueError(
            f"latest_output_horizon_months={latest_horizon} is not in the generated "
            f"horizons {list(forecasts.keys())}. Check forecast_inference.monthly_prophet."
        )
    forecast_latest = forecasts[latest_horizon].copy()

    inference_metadata = _build_inference_metadata(
        champion_metadata=monthly_prophet_champion_metadata,
        forecast_run_id=forecast_run_id,
        created_at=created_at,
        active_regressors=active_regressors,
        horizon_summaries=horizon_summaries,
        latest_horizon=latest_horizon,
    )

    logger.info(
        "Inference complete — forecast_run_id: %s | 3m rows: %d | 6m rows: %d | 12m rows: %d",
        forecast_run_id,
        len(forecast_3m),
        len(forecast_6m),
        len(forecast_12m),
    )
    logger.info(
        "Latest forecast: %dm horizon → dataset: monthly_prophet_forecast_latest",
        latest_horizon,
    )

    return forecast_3m, forecast_6m, forecast_12m, forecast_latest, inference_metadata


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_champion_metadata(champion_metadata: dict) -> None:
    """Raise ValueError if required champion metadata keys are missing or invalid.

    Args:
        champion_metadata: Champion metadata dict produced by Stage 5 model selection.
    """
    missing = [k for k in _REQUIRED_METADATA_KEYS if k not in champion_metadata]
    if missing:
        raise ValueError(
            f"Champion metadata is missing required keys: {missing}. "
            f"Found keys: {list(champion_metadata.keys())}"
        )
    if not isinstance(champion_metadata.get("active_regressors"), list):
        raise ValueError(
            "champion_metadata['active_regressors'] must be a list of column names."
        )
    if not champion_metadata["active_regressors"]:
        raise ValueError(
            "champion_metadata['active_regressors'] is empty. "
            "At least one regressor is required for Monthly Prophet inference."
        )


def _validate_future_dataset(  # noqa: PLR0912, PLR0913
    future_df: pd.DataFrame,
    active_regressors: list[str],
    expected_horizon: int,
    dataset_name: str,
    date_col: str,
    sku_col: str,
    params: dict,
) -> None:
    """Validate a future input DataFrame before Prophet inference.

    Performs structural, type, uniqueness, null, and horizon length checks.
    Raises a descriptive ValueError on the first failing check.

    Args:
        future_df: Future feature DataFrame to validate.
        active_regressors: Regressor columns that must be present and non-null.
        expected_horizon: Expected number of future months for this dataset.
        dataset_name: Dataset name used in error messages.
        date_col: Name of the date column.
        sku_col: Name of the SKU column.
        params: Validation flags (fail_on_missing_regressors, fail_on_null_regressors,
            fail_if_future_contains_y).
    """
    fail_on_missing = bool(params.get("fail_on_missing_regressors", True))
    fail_on_nulls = bool(params.get("fail_on_null_regressors", True))
    fail_if_y = bool(params.get("fail_if_future_contains_y", True))

    if future_df.empty:
        raise ValueError(f"Future dataset '{dataset_name}' is empty.")

    for required_col in [date_col, sku_col]:
        if required_col not in future_df.columns:
            raise ValueError(
                f"Future dataset '{dataset_name}' is missing required column "
                f"'{required_col}'."
            )

    # y must not be present in future inference inputs — it indicates a data pipeline
    # error and could silently cause the model to overfit during batch inference.
    if fail_if_y and "y" in future_df.columns:
        raise ValueError(
            f"Future dataset '{dataset_name}' contains column 'y' (target). "
            "Future inference inputs must not include the target column."
        )

    # Datetime check
    try:
        pd.to_datetime(future_df[date_col])
    except Exception as exc:
        raise ValueError(
            f"Column '{date_col}' in '{dataset_name}' could not be parsed as datetime."
        ) from exc

    # Active regressors
    missing_regressors = [r for r in active_regressors if r not in future_df.columns]
    if missing_regressors and fail_on_missing:
        raise ValueError(
            f"Future dataset '{dataset_name}' is missing active regressor columns: "
            f"{missing_regressors}. All active regressors must be present for inference."
        )

    present_regressors = [r for r in active_regressors if r in future_df.columns]

    # Null check on active regressors
    if fail_on_nulls and present_regressors:
        null_cols = [r for r in present_regressors if future_df[r].isnull().any()]
        if null_cols:
            raise ValueError(
                f"Future dataset '{dataset_name}' has null values in active regressors: "
                f"{null_cols}. Future regressors must be complete for inference."
            )

    # Unique ds per sku
    sku_values = future_df[sku_col].unique()
    for sku_val in sku_values:
        sku_mask = future_df[sku_col] == sku_val
        n_dates = int(sku_mask.sum())
        n_unique = int(future_df.loc[sku_mask, date_col].nunique())
        if n_dates != n_unique:
            raise ValueError(
                f"Future dataset '{dataset_name}' has duplicate '{date_col}' values "
                f"for SKU '{sku_val}'."
            )

    # Horizon length check (per SKU)
    for sku_val in sku_values:
        sku_n = int((future_df[sku_col] == sku_val).sum())
        if sku_n != expected_horizon:
            raise ValueError(
                f"Future dataset '{dataset_name}' has {sku_n} rows for SKU "
                f"'{sku_val}' but expected {expected_horizon} months. "
                "Check model_input_preparation outputs."
            )

    # Consecutive month-start dates check
    future_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(future_df[date_col]).unique())
    )
    for i in range(1, len(future_dates)):
        expected_next = future_dates[i - 1] + pd.offsets.MonthBegin(1)
        if future_dates[i] != expected_next:
            raise ValueError(
                f"Future dataset '{dataset_name}' has non-consecutive month-start dates. "
                f"Expected {expected_next.date()} after {future_dates[i - 1].date()}, "
                f"found {future_dates[i].date()}."
            )

    logger.info(
        "Validation passed for '%s' — %d rows | %s → %s",
        dataset_name,
        len(future_df),
        str(future_dates[0].date()),
        str(future_dates[-1].date()),
    )


def _build_prophet_prediction_input(
    future_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str,
) -> pd.DataFrame:
    """Build the Prophet prediction input containing only ds and active regressors.

    y must not be in the prediction input — Prophet does not use the target during
    inference, and its presence would indicate a data pipeline error upstream.

    Args:
        future_df: Future feature DataFrame with date_col, sku, and active regressors.
        active_regressors: Regressor columns registered with the champion model.
        date_col: Name of the date column in future_df.

    Returns:
        DataFrame with only ds (renamed from date_col) and active regressors.
    """
    predict_df = future_df[[date_col] + active_regressors].copy()
    predict_df = predict_df.rename(columns={date_col: "ds"})
    predict_df["ds"] = pd.to_datetime(predict_df["ds"])
    return predict_df


def _add_horizon_month(
    forecast_df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """Add a 1-based horizon_month integer column to the forecast DataFrame.

    horizon_month=1 is the first forecasted month after the latest historical period.
    Ordering is determined by sorting date_col ascending — it does not depend on the
    calendar month number.

    Args:
        forecast_df: Forecast DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with an additional horizon_month integer column.
    """
    sorted_dates = sorted(pd.to_datetime(forecast_df[date_col]).unique())
    date_to_horizon = {d: i + 1 for i, d in enumerate(sorted_dates)}
    result = forecast_df.copy()
    result["horizon_month"] = pd.to_datetime(result[date_col]).map(date_to_horizon)
    return result


def _add_forecast_metadata(  # noqa: PLR0913
    forecast_df: pd.DataFrame,
    future_df: pd.DataFrame,
    champion_metadata: dict,
    forecast_run_id: str,
    created_at: str,
    horizon_months: int,
    source_dataset: str,
    date_col: str,
    sku_col: str,
    output_cfg: dict,
) -> pd.DataFrame:
    """Annotate a forecast DataFrame with champion and inference traceability columns.

    Prophet predict() does not preserve non-ds input columns, so sku is reattached
    here by left-joining on the ds key from the original future input.

    Args:
        forecast_df: Raw forecast DataFrame from Prophet (has ds, yhat, etc.).
        future_df: Source future DataFrame used to reattach sku.
        champion_metadata: Champion metadata dict from Stage 5.
        forecast_run_id: Unique run identifier for this inference execution.
        created_at: ISO-formatted forecast creation timestamp.
        horizon_months: Number of months in this forecast horizon.
        source_dataset: Catalog name of the source future dataset.
        date_col: Name of the date column in future_df.
        sku_col: Name of the SKU column.
        output_cfg: Output configuration dict (model_family, model_granularity).

    Returns:
        Annotated forecast DataFrame with all required traceability columns.
    """
    result = forecast_df.copy()

    # Prophet predict() drops all non-ds columns; reattach sku by joining on the date
    # key. forecast_df has ds (renamed by _build_prophet_prediction_input); future_df
    # has date_col which may differ from "ds".
    if sku_col not in result.columns:
        sku_map = (
            future_df[[date_col, sku_col]]
            .copy()
            .assign(ds=lambda df: pd.to_datetime(df[date_col]))
            .drop(columns=[date_col] if date_col != "ds" else [])
            .drop_duplicates(subset=["ds"])
        )
        result = result.merge(sku_map, on="ds", how="left")

    result["model_family"] = str(output_cfg.get("model_family", "prophet"))
    result["model_granularity"] = str(output_cfg.get("model_granularity", "monthly"))
    result["champion_id"] = str(champion_metadata.get("champion_id", ""))
    result["forecast_run_id"] = forecast_run_id
    result["forecast_created_at"] = created_at
    result["forecast_horizon_months"] = horizon_months
    result["selection_metric"] = str(champion_metadata.get("selection_metric", ""))
    result["selection_metric_value"] = champion_metadata.get("selection_metric_value")
    result["business_success_flag"] = bool(
        champion_metadata.get("business_success_flag", False)
    )
    result["source_dataset"] = source_dataset

    return result


def _forecast_one_horizon(  # noqa: PLR0913
    model: Any,
    future_df: pd.DataFrame,
    active_regressors: list[str],
    champion_metadata: dict,
    forecast_run_id: str,
    created_at: str,
    horizon_months: int,
    source_dataset: str,
    date_col: str,
    sku_col: str,
    output_cfg: dict,
) -> pd.DataFrame:
    """Run Prophet inference for a single horizon and return an annotated forecast.

    Args:
        model: Fitted Prophet champion model.
        future_df: Future feature DataFrame for this horizon.
        active_regressors: Active regressor column names registered with the model.
        champion_metadata: Champion metadata dict from Stage 5.
        forecast_run_id: Unique run identifier for this inference execution.
        created_at: ISO-formatted creation timestamp.
        horizon_months: Number of months in this forecast horizon.
        source_dataset: Catalog name of the source future dataset.
        date_col: Name of the date column in future_df.
        sku_col: Name of the SKU column.
        output_cfg: Output configuration dict.

    Returns:
        Annotated forecast DataFrame with standardised column order.
    """
    predict_input = _build_prophet_prediction_input(
        future_df, active_regressors, date_col
    )
    raw_forecast = model.predict(predict_input)

    # Keep only the Prophet point forecast and prediction interval columns; all other
    # Prophet components (trend, seasonality decomposition, regressor contributions)
    # are dropped to maintain a clean, minimal output schema.
    include_intervals = bool(output_cfg.get("include_prediction_intervals", True))
    keep_cols = ["ds", "yhat"]
    if include_intervals:
        for interval_col in ("yhat_lower", "yhat_upper"):
            if interval_col in raw_forecast.columns:
                keep_cols.append(interval_col)

    forecast_df = raw_forecast[keep_cols].copy()

    # Ensure prediction interval columns exist even when not computed, so the output
    # schema stays consistent across different Prophet configurations.
    for interval_col in ("yhat_lower", "yhat_upper"):
        if interval_col not in forecast_df.columns:
            forecast_df[interval_col] = None

    forecast_df = _add_horizon_month(forecast_df, "ds")

    forecast_df = _add_forecast_metadata(
        forecast_df=forecast_df,
        future_df=future_df,
        champion_metadata=champion_metadata,
        forecast_run_id=forecast_run_id,
        created_at=created_at,
        horizon_months=horizon_months,
        source_dataset=source_dataset,
        date_col=date_col,
        sku_col=sku_col,
        output_cfg=output_cfg,
    )

    present_cols = [c for c in _FORECAST_COLUMN_ORDER if c in forecast_df.columns]
    return forecast_df[present_cols]


def _summarize_forecast_output(
    forecast_df: pd.DataFrame,
    date_col: str,
    output_dataset: str,
) -> dict:
    """Build a compact summary dict for one forecast horizon.

    Args:
        forecast_df: Annotated forecast DataFrame.
        date_col: Name of the date column.
        output_dataset: Catalog name of the output dataset.

    Returns:
        Dict with output_dataset, start_date, end_date, and rows.
    """
    dates = pd.to_datetime(forecast_df[date_col])
    return {
        "output_dataset": output_dataset,
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "rows": len(forecast_df),
    }


def _build_inference_metadata(  # noqa: PLR0913
    champion_metadata: dict,
    forecast_run_id: str,
    created_at: str,
    active_regressors: list[str],
    horizon_summaries: dict[int, dict],
    latest_horizon: int,
) -> dict:
    """Build the JSON-serialisable inference metadata artifact.

    Args:
        champion_metadata: Champion metadata dict from Stage 5.
        forecast_run_id: Unique run identifier for this inference execution.
        created_at: ISO-formatted creation timestamp.
        active_regressors: Active regressor column names used during inference.
        horizon_summaries: Dict mapping horizon_months → summary dict.
        latest_horizon: Horizon used for the latest forecast output.

    Returns:
        JSON-serialisable inference metadata dict consumed by app and reporting layers.
    """
    return {
        "model_family": str(champion_metadata.get("model_family", "prophet")),
        "model_granularity": str(champion_metadata.get("granularity", "monthly")),
        "champion_id": str(champion_metadata.get("champion_id", "")),
        "forecast_run_id": forecast_run_id,
        "forecast_created_at": created_at,
        "active_regressors": active_regressors,
        "horizons": {str(h): summary for h, summary in horizon_summaries.items()},
        "latest_output": {
            "dataset": "monthly_prophet_forecast_latest",
            "horizon_months": latest_horizon,
        },
        "selection": {
            "selection_metric": str(champion_metadata.get("selection_metric", "")),
            "selection_metric_value": champion_metadata.get("selection_metric_value"),
            "business_success_flag": bool(
                champion_metadata.get("business_success_flag", False)
            ),
        },
    }


# ── Future inference stage stubs ──────────────────────────────────────────────
# The following functions are placeholder stubs for weekly inference and daily
# allocation that will be implemented in later stages of the project.


def load_inference_inputs(
    feature_monthly: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the input DataFrames for inference (no target leakage).

    Args:
        feature_monthly: Full monthly feature DataFrame including the most recent rows.
        feature_weekly: Full weekly feature DataFrame including the most recent rows.
        parameters: Config from forecast_inference key. Expected keys: mode,
            include_intervals, confidence_level.

    Returns:
        Tuple of (monthly_inference_df, weekly_inference_df) containing only the
        feature columns required for prediction, with future date rows appended
        for the forecast horizon.
    """
    raise NotImplementedError(
        "Identify the most recent date in each granularity, extend the date range "
        "by the configured horizon, build future feature rows (lag features from "
        "the most recent actuals), and return the inference-ready DataFrames."
    )


def generate_monthly_forecast(
    champion_monthly_model: Any,
    monthly_inference_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generate monthly demand forecasts using the elected champion model.

    Args:
        champion_monthly_model: Serialised champion model from data/06_models/champions/.
        monthly_inference_df: Inference-ready monthly DataFrame from load_inference_inputs.
        parameters: Config from forecast_inference key with include_intervals and
            confidence_level.

    Returns:
        DataFrame with columns [ds, y_forecast, y_lower, y_upper, model_name,
        granularity, horizon_step] for each forecast month.
    """
    raise NotImplementedError(
        "Dispatch to the appropriate prediction method based on model type (Prophet: "
        "model.predict(); CatBoost: model.predict(); SARIMAX: results.forecast()). "
        "If include_intervals=True, compute prediction intervals at confidence_level. "
        "Return a standardised forecast DataFrame."
    )


def generate_weekly_forecast(
    champion_weekly_model: Any,
    weekly_inference_df: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generate weekly demand forecasts using the elected champion model.

    Args:
        champion_weekly_model: Serialised champion model from data/06_models/champions/.
        weekly_inference_df: Inference-ready weekly DataFrame from load_inference_inputs.
        parameters: Config from forecast_inference key with include_intervals and
            confidence_level.

    Returns:
        DataFrame with columns [ds, y_forecast, y_lower, y_upper, model_name,
        granularity, horizon_step] for each forecast week.
    """
    raise NotImplementedError(
        "Apply the same dispatch logic as generate_monthly_forecast but for the weekly "
        "champion model. Return a standardised forecast DataFrame with granularity='weekly'."
    )


def allocate_daily_forecast(
    forecast_weekly_reconciled: pd.DataFrame,
    feature_weekly: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Disaggregate reconciled weekly forecasts to daily estimates.

    Args:
        forecast_weekly_reconciled: Reconciled weekly forecast DataFrame.
        feature_weekly: Weekly feature DataFrame used to compute historical daily shares.
        parameters: Config from forecast_inference key. Expected key: daily_allocation
            with 'enabled' (bool) and 'method' ('historical_shares').

    Returns:
        DataFrame with daily forecast rows, or an empty DataFrame if
        daily_allocation.enabled is False.
    """
    raise NotImplementedError(
        "If parameters['daily_allocation']['enabled'] is False, return an empty "
        "DataFrame. Otherwise, compute historical day-of-week share fractions from "
        "feature_weekly and distribute each week's forecast proportionally across "
        "7 daily rows."
    )
