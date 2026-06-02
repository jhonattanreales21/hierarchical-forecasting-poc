"""Family-specific monthly inference adapters and metadata-driven dispatch.

monthly inference generic: the family is resolved from ``champion_monthly_metadata``
rather than from the model object's Python type.

Each adapter returns a *core* forecast DataFrame using the canonical column names
(``date``, ``forecast``, ``forecast_lower``, ``forecast_upper``, ``sku``,
``has_prediction_interval``, ``interval_method``).  The calling node decorates these
with run/selection metadata and enforces the full standard schema via
``_standardize_forecast_schema`` in :mod:`nodes`.

"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from shared.catboost_recursive import (
    build_demand_buffer as _build_demand_buffer,
)
from shared.catboost_recursive import (
    compute_recursive_target_features as _compute_recursive_features_for_step,
)
from shared.catboost_recursive import (
    extract_periods_from_column_names as _extract_periods_from_column_names,
)

logger = logging.getLogger(__name__)

SUPPORTED_MONTHLY_FAMILIES: tuple[str, ...] = ("prophet", "sarimax", "catboost")

# Columns every adapter must return before standardisation decorates the frame.
_ADAPTER_CORE_COLUMNS: tuple[str, ...] = (
    "date",
    "forecast",
    "forecast_lower",
    "forecast_upper",
    "sku",
    "has_prediction_interval",
    "interval_method",
)


# ── Dispatch ──────────────────────────────────────────────────────────────────


def dispatch_monthly_prediction(
    model: Any,
    metadata: dict,
    future_df: pd.DataFrame,
    params: dict,
    horizon: int,
    *,
    history_df: pd.DataFrame | None = None,
    catboost_split_metadata: dict | None = None,
) -> pd.DataFrame:
    """Route a monthly forecast request to the correct family adapter.

    Dispatch is driven exclusively by ``metadata["model_family"]`` — never by the
    Python type of ``model`` — so the same inference node serves every supported
    champion family.

    Args:
        model: The champion model artifact. For Prophet this is the fitted model;
            for SARIMAX it is the candidate entry dict (``{"config", "model", ...}``)
            or a fitted results object; for CatBoost it is the candidate entry dict
            carrying ``"model"`` (a fitted ``CatBoostRegressor``) and
            ``"feature_columns"``.
        metadata: ``champion_monthly_metadata`` produced by model selection. Must
            contain a non-empty ``model_family``.
        future_df: Future feature frame for this horizon (date column plus any
            regressor/exogenous columns).
        params: Contents of ``forecast_inference.monthly``.
        horizon: Forecast horizon in months (3, 6, or 12); used for error messages.
        history_df: Historical CatBoost-ready DataFrame used to seed the recursive
            demand buffer. Only consumed by the CatBoost adapter; ignored by others.
        catboost_split_metadata: Split metadata produced by
            ``adapt_monthly_data_for_catboost``; provides column names, lag settings,
            rolling settings, and future-required column lists. Only consumed by the
            CatBoost adapter; ignored by others.

    Returns:
        Core forecast DataFrame with the canonical adapter columns.

    Raises:
        ValueError: If ``model_family`` is missing or unsupported.
    """
    model_family = str(metadata.get("model_family", "")).strip().lower()

    if not model_family:
        raise ValueError(
            "champion_monthly_metadata is missing required field 'model_family'. "
            "Monthly inference cannot dispatch to a prediction adapter without this "
            "value."
        )

    if model_family == "prophet":
        return predict_monthly_prophet(model, future_df, metadata, params, horizon)
    if model_family == "sarimax":
        return predict_monthly_sarimax(model, future_df, metadata, params, horizon)
    if model_family == "catboost":
        return predict_monthly_catboost(
            model,
            future_df,
            metadata,
            params,
            horizon,
            history_df=history_df,
            catboost_split_metadata=catboost_split_metadata,
        )

    raise ValueError(
        f"Unsupported monthly champion model_family: {model_family}. "
        f"Supported families are: {', '.join(SUPPORTED_MONTHLY_FAMILIES)}."
    )


# ── Prophet adapter ───────────────────────────────────────────────────────────


def predict_monthly_prophet(
    model: Any,
    future_df: pd.DataFrame,
    metadata: dict,
    params: dict,
    horizon: int,
) -> pd.DataFrame:
    """Generate monthly forecasts from a Prophet champion model.

    Preserves the existing Prophet MVP behaviour: only ``ds`` plus the registered
    active regressors are passed to ``predict()``, and native prediction intervals
    are mapped to the canonical interval columns when present.

    Args:
        model: Fitted Prophet champion model.
        future_df: Future regressor frame containing the date column and regressors.
        metadata: Champion metadata; ``active_regressors`` is used when present.
        params: Contents of ``forecast_inference.monthly``.
        horizon: Forecast horizon in months (for error messages).

    Returns:
        Core forecast DataFrame with the canonical adapter columns.

    Raises:
        ValueError: If the future frame is empty or the prediction column is absent.
    """
    prophet_cfg: dict = params.get("prophet", {})
    sku_col: str = params.get("sku_column", "sku")
    pred_col: str = prophet_cfg.get("prediction_column", "yhat")
    lower_col: str = prophet_cfg.get("lower_column", "yhat_lower")
    upper_col: str = prophet_cfg.get("upper_column", "yhat_upper")

    if future_df.empty:
        raise ValueError(
            f"Prophet monthly inference received an empty {horizon}-month future frame."
        )

    date_col = _resolve_date_column(
        future_df, [prophet_cfg.get("date_column", "ds"), "ds", "month_start_date"]
    )
    active_regressors = _resolve_prophet_regressors(model, metadata, prophet_cfg)

    predict_input = pd.DataFrame({"ds": pd.to_datetime(future_df[date_col].to_numpy())})
    for regressor in active_regressors:
        if regressor in future_df.columns:
            predict_input[regressor] = future_df[regressor].to_numpy()

    raw = model.predict(predict_input)
    if pred_col not in raw.columns:
        raise ValueError(
            f"Prophet prediction output is missing the point-forecast column "
            f"'{pred_col}'. Available columns: {list(raw.columns)}."
        )

    has_lower = lower_col in raw.columns
    has_upper = upper_col in raw.columns
    has_interval = bool(has_lower and has_upper)

    core = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["ds"].to_numpy()),
            "forecast": pd.to_numeric(raw[pred_col], errors="coerce")
            .astype(float)
            .to_numpy(),
            "forecast_lower": (
                pd.to_numeric(raw[lower_col], errors="coerce").to_numpy()
                if has_lower
                else np.nan
            ),
            "forecast_upper": (
                pd.to_numeric(raw[upper_col], errors="coerce").to_numpy()
                if has_upper
                else np.nan
            ),
        }
    )
    core["has_prediction_interval"] = has_interval
    core["interval_method"] = "prophet_native" if has_interval else None
    core = _attach_sku(core, future_df, date_col, sku_col)

    logger.info(
        "Prophet adapter produced %d rows for the %d-month horizon (intervals=%s).",
        len(core),
        horizon,
        has_interval,
    )
    return core[list(_ADAPTER_CORE_COLUMNS)]


# ── SARIMAX adapter ───────────────────────────────────────────────────────────


def predict_monthly_sarimax(
    model: Any,
    future_df: pd.DataFrame,
    metadata: dict,
    params: dict,
    horizon: int,
) -> pd.DataFrame:
    """Generate monthly forecasts from a SARIMAX champion model.

    The champion artifact may be the candidate entry dict emitted by SARIMAX
    training (``{"config", "model", ...}``) or a bare fitted statsmodels results
    object. Forecast steps are taken from the future frame length. Confidence
    intervals are produced from ``get_forecast().conf_int`` when available, and left
    null otherwise (never fabricated).

    Args:
        model: SARIMAX champion candidate dict or fitted results object.
        future_df: Future frame providing the date index and any exogenous columns.
        metadata: Champion metadata; ``active_regressors`` may name exogenous columns.
        params: Contents of ``forecast_inference.monthly``.
        horizon: Forecast horizon in months (for validation and error messages).

    Returns:
        Core forecast DataFrame with the canonical adapter columns.

    Raises:
        ValueError: If the future frame is empty, the model cannot forecast, or
            required exogenous columns are missing/invalid.
    """
    sarimax_cfg: dict = params.get("sarimax", {})
    sku_col: str = params.get("sku_column", "sku")

    if future_df.empty:
        raise ValueError(
            f"SARIMAX monthly inference received an empty {horizon}-month future frame."
        )

    results, config = _unwrap_sarimax_model(model)
    date_col = _resolve_date_column(
        future_df, [sarimax_cfg.get("date_column", "month_start_date"), "ds"]
    )

    ordered = future_df.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values(date_col).reset_index(drop=True)
    steps = len(ordered)

    exog_cols = _resolve_sarimax_exog(results, config, metadata, sarimax_cfg)
    future_exog: np.ndarray | None = None
    if exog_cols:
        _validate_future_exog(ordered, exog_cols, steps, horizon)
        future_exog = ordered[exog_cols].to_numpy(dtype=float)

    predicted_mean, lower, upper, interval_method = _sarimax_forecast(
        results, steps, future_exog, sarimax_cfg
    )
    has_interval = lower is not None and upper is not None

    core = pd.DataFrame(
        {
            "date": ordered[date_col].to_numpy(),
            "forecast": np.asarray(predicted_mean, dtype=float)[:steps],
            "forecast_lower": lower if lower is not None else np.full(steps, np.nan),
            "forecast_upper": upper if upper is not None else np.full(steps, np.nan),
        }
    )
    core["has_prediction_interval"] = bool(has_interval)
    core["interval_method"] = interval_method
    core["sku"] = ordered[sku_col].to_numpy() if sku_col in ordered.columns else None

    logger.info(
        "SARIMAX adapter produced %d rows for the %d-month horizon "
        "(exog=%s, intervals=%s).",
        len(core),
        horizon,
        exog_cols if exog_cols else "(none)",
        has_interval,
    )
    return core[list(_ADAPTER_CORE_COLUMNS)]


# ── CatBoost adapter ──────────────────────────────────────────────────────────


def predict_monthly_catboost(
    model: Any,
    future_df: pd.DataFrame,
    metadata: dict,
    params: dict,
    horizon: int,
    *,
    history_df: pd.DataFrame | None = None,
    catboost_split_metadata: dict | None = None,
) -> pd.DataFrame:
    """Generate monthly forecasts from a CatBoost champion using recursive inference.

    Builds target-derived lag, rolling, and trend features recursively from the
    demand history plus prior predictions, then merges future-known calendar and
    exogenous features for each forecast step. The exact training ``feature_columns``
    order is preserved to ensure the feature matrix matches training.

    Prediction intervals are not produced; ``forecast_lower`` and ``forecast_upper``
    are set to ``NaN`` with ``interval_method=None``.

    Args:
        model: CatBoost champion artifact — a candidate dict carrying ``"model"``
            (fitted ``CatBoostRegressor``) and ``"feature_columns"`` (ordered list),
            or a bare ``CatBoostRegressor`` when ``feature_columns`` can be resolved
            from ``metadata``.
        future_df: Future feature frame for this horizon (date column plus
            future-known calendar/exogenous columns; must not contain the target).
        metadata: ``champion_monthly_metadata``. ``feature_columns`` is read when
            absent from the model artifact.
        params: Contents of ``forecast_inference.monthly``.
        horizon: Forecast horizon in months (3, 6, or 12).
        history_df: CatBoost-ready historical DataFrame (e.g.
            ``monthly_catboost_full_train``) used to seed the recursive demand
            buffer. Target column must be present.
        catboost_split_metadata: Split metadata from ``adapt_monthly_data_for_catboost``;
            provides ``date_column``, ``target_column``, ``lag_settings``,
            ``rolling_settings``, ``trend_feature_columns``, and
            ``future_required_columns``.

    Returns:
        Core forecast DataFrame with the canonical adapter columns.

    Raises:
        ValueError: If the future frame is empty, required future columns are missing,
            or the model artifact cannot be resolved.
    """
    catboost_cfg: dict = params.get("catboost", {})
    sku_col: str = params.get("sku_column", "sku")

    if future_df.empty:
        raise ValueError(
            f"CatBoost monthly inference received an empty {horizon}-month future frame."
        )

    cb_model, feature_columns = _unwrap_catboost_model(model, metadata)

    split_meta = catboost_split_metadata or {}
    target_col = str(split_meta.get("target_column") or catboost_cfg.get("target_column") or "monthly_demand")
    raw_date_col = str(
        split_meta.get("date_column") or catboost_cfg.get("date_column") or "month_start_date"
    )
    date_col = _resolve_date_column(future_df, [raw_date_col, "month_start_date", "ds"])

    ordered = future_df.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values(date_col).reset_index(drop=True)

    future_required = list(split_meta.get("future_required_columns") or [])
    _validate_catboost_future_required_columns(ordered, future_required, horizon, feature_columns, metadata)

    # Resolve recursive feature settings from split metadata or fallback defaults.
    lag_settings = dict(split_meta.get("lag_settings") or {})
    target_lags = sorted(int(x) for x in lag_settings.get("target_lags", [1, 2, 3, 6, 12]))

    rolling_settings = dict(split_meta.get("rolling_settings") or {})
    rolling_windows = sorted(int(x) for x in rolling_settings.get("windows", [3, 6, 12]))
    include_std = bool(rolling_settings.get("include_std", True))
    include_min_max = bool(rolling_settings.get("include_min_max", True))

    trend_feature_cols = list(split_meta.get("trend_feature_columns") or [])
    trend_diffs = _extract_periods_from_column_names(trend_feature_cols, "demand_diff_")
    trend_pct_changes = _extract_periods_from_column_names(trend_feature_cols, "demand_pct_change_")

    demand_buffer = _build_demand_buffer(history_df, target_col, date_col)

    min_required_buffer = max(target_lags, default=0)
    if min_required_buffer and len(demand_buffer) < min_required_buffer:
        logger.warning(
            "CatBoost demand buffer has %d observations but max lag is %d. "
            "Leading recursive steps will produce NaN lag features.",
            len(demand_buffer),
            min_required_buffer,
        )

    # Recursive forecast loop: one step at a time, appending each prediction.
    forecast_rows: list[dict] = []
    for _, row in ordered.iterrows():
        recursive_feats = _compute_recursive_features_for_step(
            demand_buffer=demand_buffer,
            target_lags=target_lags,
            rolling_windows=rolling_windows,
            include_std=include_std,
            include_min_max=include_min_max,
            trend_diffs=trend_diffs,
            trend_pct_changes=trend_pct_changes,
        )

        # Build feature vector in exact training column order.
        feature_values = []
        for col in feature_columns:
            if col in recursive_feats:
                feature_values.append(recursive_feats[col])
            elif col in row.index:
                feature_values.append(row[col])
            else:
                feature_values.append(np.nan)

        X = np.array([feature_values], dtype=float)
        prediction = float(cb_model.predict(X)[0])
        forecast_rows.append({"date": row[date_col], "forecast": prediction})
        demand_buffer.append(prediction)

    core = pd.DataFrame(forecast_rows)
    core["date"] = pd.to_datetime(core["date"])
    core["forecast"] = core["forecast"].astype(float)
    core["forecast_lower"] = np.nan
    core["forecast_upper"] = np.nan
    core["has_prediction_interval"] = False
    core["interval_method"] = None
    core["sku"] = ordered[sku_col].to_numpy() if sku_col in ordered.columns else None

    logger.info(
        "CatBoost adapter produced %d rows for the %d-month horizon "
        "(recursive, %d features, intervals=None).",
        len(core),
        horizon,
        len(feature_columns),
    )
    return core[list(_ADAPTER_CORE_COLUMNS)]


# ── CatBoost private helpers ──────────────────────────────────────────────────


def _unwrap_catboost_model(model: Any, metadata: dict) -> tuple[Any, list[str]]:
    """Extract the fitted CatBoostRegressor and ordered feature_columns.

    Resolution order for ``feature_columns``:
    1. ``model["feature_columns"]`` when the artifact is a candidate dict.
    2. ``metadata["feature_columns"]`` from champion metadata.
    3. ``model.feature_names_`` attribute on the fitted model.

    Raises:
        ValueError: If no ``feature_columns`` can be resolved or the model object
            cannot be identified as a CatBoostRegressor.
    """
    feature_columns_from_meta = list(metadata.get("feature_columns") or [])

    if isinstance(model, dict):
        cb_model = model.get("model")
        if cb_model is None:
            raise ValueError(
                "CatBoost champion artifact dict has no 'model' key holding a fitted "
                f"CatBoostRegressor. Keys present: {list(model.keys())}."
            )
        feature_columns = list(model.get("feature_columns") or feature_columns_from_meta)
    elif hasattr(model, "predict"):
        cb_model = model
        feature_columns = list(feature_columns_from_meta)
    else:
        raise ValueError(
            "CatBoost champion artifact is neither a candidate dict nor a model "
            f"exposing predict(); received type {type(model)!r}."
        )

    if not feature_columns:
        try:
            names = list(getattr(cb_model, "feature_names_", None) or [])
            if names:
                feature_columns = names
        except Exception:  # noqa: BLE001
            pass

    if not feature_columns:
        champion_id = str(metadata.get("champion_id", "unknown"))
        raise ValueError(
            "Cannot resolve CatBoost feature_columns for inference. "
            "The champion artifact must carry 'feature_columns' in the candidate dict "
            "or champion_monthly_metadata must include 'feature_columns'. "
            f"Champion ID: {champion_id}."
        )

    return cb_model, feature_columns


def _validate_catboost_future_required_columns(
    future_df: pd.DataFrame,
    future_required_columns: list[str],
    horizon: int,
    feature_columns: list[str],
    metadata: dict,
) -> None:
    """Validate that all future-required (non-recursive) columns are present.

    These columns cannot be computed recursively from demand history and must
    come from the generic future monthly frames.

    Raises:
        ValueError: Naming every missing column, the horizon, and the champion id.
    """
    if not future_required_columns:
        return

    missing = [c for c in future_required_columns if c not in future_df.columns]
    if not missing:
        return

    champion_id = str(metadata.get("champion_id", "unknown"))
    feature_preview = feature_columns[:8]
    suffix = "..." if len(feature_columns) > 8 else ""
    raise ValueError(
        f"CatBoost monthly inference ({horizon}-month horizon, champion={champion_id}) "
        f"requires future calendar/exogenous columns that are missing from the future "
        f"frame: {missing}. These columns must be provided externally — they cannot be "
        f"computed recursively from demand history. "
        f"Model feature_columns (first 8): {feature_preview}{suffix}."
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _resolve_date_column(future_df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate date column present in the future frame.

    Each adapter passes a family-preferred name first (e.g. ``month_start_date``
    for SARIMAX, ``ds`` for Prophet) followed by fallbacks so either frame format
    is accepted without error.

    Args:
        future_df: Future feature frame.
        candidates: Ordered candidate date-column names.

    Returns:
        The first candidate present in ``future_df``.

    Raises:
        ValueError: If none of the candidate columns are present.
    """
    for candidate in candidates:
        if candidate and candidate in future_df.columns:
            return candidate
    raise ValueError(
        "Future frame is missing a usable date column. Looked for "
        f"{[c for c in candidates if c]}; found columns {list(future_df.columns)}."
    )


def _resolve_prophet_regressors(
    model: Any, metadata: dict, prophet_cfg: dict
) -> list[str]:
    """Resolve the Prophet active regressors, preferring explicit metadata.

    Resolution order: champion metadata ``active_regressors`` → the regressors
    registered on the fitted Prophet model (``model.extra_regressors``) → the
    configured ``prophet.active_regressors`` parameter. An empty result is valid
    (Prophet can predict from ``ds`` alone).
    """
    from_metadata = metadata.get("active_regressors")
    if from_metadata:
        return list(from_metadata)

    extra = getattr(model, "extra_regressors", None)
    if isinstance(extra, dict) and extra:
        return list(extra.keys())

    return list(prophet_cfg.get("active_regressors", []) or [])


def _attach_sku(
    core: pd.DataFrame, future_df: pd.DataFrame, date_col: str, sku_col: str
) -> pd.DataFrame:
    """Reattach the SKU column to a forecast frame by joining on the date key.

    Prophet ``predict()`` drops every non-``ds`` input column, so the SKU is
    re-merged from the future frame. If the future frame has no SKU column the
    output SKU is set to null.
    """
    if sku_col not in future_df.columns:
        core["sku"] = None
        return core

    sku_map = (
        future_df[[date_col, sku_col]]
        .assign(date=lambda df: pd.to_datetime(df[date_col]))
        .drop_duplicates(subset=["date"])[["date", sku_col]]
        .rename(columns={sku_col: "sku"})
    )
    merged = core.merge(sku_map, on="date", how="left")
    return merged


def _unwrap_sarimax_model(model: Any) -> tuple[Any, dict]:
    """Return the fitted SARIMAX results object and its config from the champion artifact.

    SARIMAX training persists each candidate as a dict carrying the fitted results
    under ``"model"`` and the search config under ``"config"``. A bare fitted
    results object is also accepted for flexibility/testing.

    Raises:
        ValueError: If no forecastable results object can be located.
    """
    if isinstance(model, dict):
        results = model.get("model")
        config = dict(model.get("config", {}))
        if results is None:
            raise ValueError(
                "SARIMAX champion artifact dict has no 'model' key holding a fitted "
                f"results object. Keys present: {list(model.keys())}."
            )
        return results, config

    if hasattr(model, "get_forecast") or hasattr(model, "forecast"):
        return model, {}

    raise ValueError(
        "SARIMAX champion artifact is neither a candidate dict nor a fitted results "
        f"object exposing get_forecast()/forecast(); received type {type(model)!r}."
    )


def _resolve_sarimax_exog(
    results: Any, config: dict, metadata: dict, sarimax_cfg: dict
) -> list[str]:
    """Resolve the exogenous column names the SARIMAX champion requires for inference.

    Exogenous inputs are required when the fitted model carries exogenous regressors
    (``k_exog > 0``) or its config sets ``use_exog``. Column names are taken from the
    SARIMAX parameter block first, then from champion metadata ``active_regressors``.

    Raises:
        ValueError: If the model was fit with exogenous regressors but their names
            cannot be resolved, or the resolved count disagrees with the fitted model.
    """
    inner = getattr(results, "model", None)
    k_exog = int(getattr(inner, "k_exog", 0) or 0)
    use_exog = bool(config.get("use_exog", False)) or k_exog > 0

    if not use_exog:
        return []

    names = list(
        sarimax_cfg.get("exogenous_columns") or metadata.get("active_regressors") or []
    )

    if k_exog and not names:
        raise ValueError(
            f"SARIMAX champion was fit with {k_exog} exogenous regressor(s) but no "
            "exogenous column names are available from parameters or champion "
            "metadata. Set forecast_inference.monthly.sarimax.exogenous_columns or "
            "record active_regressors in champion_monthly_metadata."
        )
    if k_exog and len(names) != k_exog:
        raise ValueError(
            f"SARIMAX champion was fit with {k_exog} exogenous regressor(s) but the "
            f"resolved exogenous contract names {len(names)} column(s): {names}."
        )
    return names


def _validate_future_exog(
    future_df: pd.DataFrame, exog_cols: list[str], steps: int, horizon: int
) -> None:
    """Validate that future exogenous inputs satisfy the SARIMAX training contract.

    Checks presence, numeric dtype, completeness, and row count.

    Raises:
        ValueError: On the first failing check, naming the offending columns.
    """
    missing = [c for c in exog_cols if c not in future_df.columns]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise ValueError(
            f"SARIMAX monthly inference requires future exogenous columns "
            f"{exog_cols}, but {missing} {verb} missing from the {horizon}-month "
            "future frame."
        )

    non_numeric = [
        c for c in exog_cols if not pd.api.types.is_numeric_dtype(future_df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"SARIMAX future exogenous columns must be numeric; non-numeric columns "
            f"in the {horizon}-month future frame: {non_numeric}."
        )

    null_cols = [c for c in exog_cols if future_df[c].isnull().any()]
    if null_cols:
        raise ValueError(
            f"SARIMAX future exogenous columns contain null values in the "
            f"{horizon}-month future frame: {null_cols}."
        )

    if len(future_df) != steps:
        raise ValueError(
            f"SARIMAX future exogenous frame has {len(future_df)} rows but "
            f"{steps} forecast steps were requested for the {horizon}-month horizon."
        )


def _sarimax_forecast(
    results: Any,
    steps: int,
    future_exog: np.ndarray | None,
    sarimax_cfg: dict,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str | None]:
    """Run the SARIMAX forecast and optionally extract confidence intervals.

    Prefers ``get_forecast`` (which exposes ``conf_int``); falls back to ``forecast``
    (point forecast only). Intervals are produced only when configured and available.

    Returns:
        Tuple ``(predicted_mean, lower, upper, interval_method)``. ``lower``/``upper``
        are ``None`` when intervals are unavailable, and ``interval_method`` describes
        how intervals were derived (or ``None``).
    """
    interval_strategy = str(sarimax_cfg.get("interval_strategy", "conf_int")).lower()
    confidence_level = float(sarimax_cfg.get("confidence_level", 0.90))
    alpha = 1.0 - confidence_level

    if hasattr(results, "get_forecast"):
        forecast_out = results.get_forecast(steps=steps, exog=future_exog)
        predicted_mean = np.asarray(forecast_out.predicted_mean, dtype=float)

        if interval_strategy == "conf_int":
            try:
                conf_int = np.asarray(forecast_out.conf_int(alpha=alpha), dtype=float)
                lower = conf_int[:steps, 0]
                upper = conf_int[:steps, 1]
                return predicted_mean, lower, upper, "sarimax_get_forecast_conf_int"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SARIMAX confidence intervals unavailable (%s); emitting null "
                    "intervals.",
                    exc,
                )
        return predicted_mean, None, None, None

    if hasattr(results, "forecast"):
        predicted_mean = np.asarray(
            results.forecast(steps=steps, exog=future_exog), dtype=float
        )
        return predicted_mean, None, None, None

    raise ValueError(
        "SARIMAX champion results object exposes neither get_forecast() nor "
        "forecast(); cannot generate monthly predictions."
    )
