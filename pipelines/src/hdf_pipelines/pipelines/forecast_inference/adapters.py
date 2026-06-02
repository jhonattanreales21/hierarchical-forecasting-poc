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

logger = logging.getLogger(__name__)

# Supported monthly families. CatBoost is intentionally excluded.
SUPPORTED_MONTHLY_FAMILIES: tuple[str, ...] = ("prophet", "sarimax")

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
) -> pd.DataFrame:
    """Route a monthly forecast request to the correct family adapter.

    Dispatch is driven exclusively by ``metadata["model_family"]`` — never by the
    Python type of ``model`` — so the same inference node serves every supported
    champion family.

    Args:
        model: The champion model artifact. For Prophet this is the fitted model;
            for SARIMAX it is the candidate entry dict (``{"config", "model", ...}``)
            or a fitted results object.
        metadata: ``champion_monthly_metadata`` produced by model selection. Must
            contain a non-empty ``model_family``.
        future_df: Future feature frame for this horizon (date column plus any
            regressor/exogenous columns).
        params: Contents of ``forecast_inference.monthly``.
        horizon: Forecast horizon in months (3, 6, or 12); used for error messages.

    Returns:
        Core forecast DataFrame with the canonical adapter columns.

    Raises:
        ValueError: If ``model_family`` is missing, is CatBoost, or is unsupported.
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
        raise ValueError(
            "Monthly champion model_family 'catboost' is not supported. "
            f"Supported families are: {', '.join(SUPPORTED_MONTHLY_FAMILIES)}."
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
