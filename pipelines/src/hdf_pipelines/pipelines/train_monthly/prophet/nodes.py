"""Monthly Prophet training and tuning nodes.

Implements grid-search hyperparameter tuning for Prophet on monthly demand data.
Trains all candidate configurations on the training split, evaluates them on the
validation split, ranks by the configured selection metric, and persists the top-N
pre-champion configurations and their fitted models for the model-selection stage.
"""

import itertools
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress verbose Stan/cmdstanpy progress output during Prophet fitting
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def train_and_evaluate_monthly_prophet_candidates(  # noqa: PLR0915
    monthly_prophet_train: pd.DataFrame,
    monthly_prophet_validation: pd.DataFrame,
    monthly_prophet_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, Any]:
    """Train and evaluate all Prophet candidate configurations on the validation set.

    Iterates over the full Cartesian product of the configured hyperparameter grid,
    fits each candidate on the training split, forecasts the validation period, and
    computes validation metrics. The top-N pre-champion candidates (ranked by the
    configured selection metric) are persisted for the downstream model-selection stage.

    Args:
        monthly_prophet_train: Prophet-ready training data with columns ds, y, sku,
            and all active regressor columns.
        monthly_prophet_validation: Held-out validation data with the same schema.
        monthly_prophet_split_metadata: Metadata from model_input_preparation
            (date ranges, active_regressors). Used for cross-checking regressors;
            params is the authoritative source for pipeline configuration.
        params: Contents of ``train_monthly.prophet`` from the parameter file.

    Returns:
        Five-element tuple:

        1. ``tuning_results`` — DataFrame, one row per candidate (ranked).
        2. ``validation_metrics`` — DataFrame, detailed per-candidate metrics.
        3. ``prechampion_configs`` — Dict with top-N pre-champion configurations.
        4. ``candidate_models`` — Dict mapping candidate_id → fitted Prophet model
           (top-N only).
        5. ``best_prophet_model`` — Rank-1 Prophet model for downstream compatibility
           with the model-selection stage.

    Raises:
        ValueError: When inputs are invalid, required columns are missing, or the
            candidate grid is empty.
        RuntimeError: When every candidate fails to train.
    """
    # ── extract configuration ─────────────────────────────────────────────────────
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")
    active_regressors: list[str] = list(params.get("active_regressors", []))
    regressor_mode: str = params.get("regressors", {}).get("mode", "additive")

    tuning_cfg: dict = params.get("tuning", {})
    selection_metric: str = tuning_cfg.get("selection_metric", "mape")
    top_n: int = int(tuning_cfg.get("top_n_prechampions", 3))
    candidate_grid: dict = tuning_cfg.get("candidate_grid", {})

    metrics_cfg: dict = params.get("metrics", {})
    epsilon: float = float(metrics_cfg.get("epsilon", 1.0))
    precision_threshold: float = float(
        metrics_cfg.get("business_success_precision_threshold", 0.85)
    )
    horizon_metrics_cfg: dict = metrics_cfg.get("horizon_metrics", {})
    horizons: list[int] = (
        [int(h) for h in horizon_metrics_cfg.get("horizons", [2, 3])]
        if horizon_metrics_cfg.get("enabled", True)
        else []
    )

    # ── validate inputs ───────────────────────────────────────────────────────────
    _validate_prophet_training_inputs(
        monthly_prophet_train,
        monthly_prophet_validation,
        active_regressors,
        date_col,
        target_col,
    )

    # Cross-check regressors against split metadata (informational only)
    metadata_regressors = monthly_prophet_split_metadata.get("active_regressors", [])
    if set(active_regressors) != set(metadata_regressors):
        logger.warning(
            "Active regressors in params differ from split metadata. "
            "Using params as the authoritative source. "
            "Params: %s | Metadata: %s",
            active_regressors,
            metadata_regressors,
        )

    # ── prepare dataframes ────────────────────────────────────────────────────────
    train_df = monthly_prophet_train.copy()
    val_df = monthly_prophet_validation.copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    val_df[date_col] = pd.to_datetime(val_df[date_col])
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    val_df = val_df.sort_values(date_col).reset_index(drop=True)

    logger.info(
        "Train  : %d rows | %s → %s",
        len(train_df),
        train_df[date_col].min().date(),
        train_df[date_col].max().date(),
    )
    logger.info(
        "Val    : %d rows | %s → %s",
        len(val_df),
        val_df[date_col].min().date(),
        val_df[date_col].max().date(),
    )
    logger.info(
        "Active regressors (%d): %s", len(active_regressors), active_regressors
    )

    # ── build candidate grid ──────────────────────────────────────────────────────
    candidate_configs = _build_candidate_grid(candidate_grid)
    if not candidate_configs:
        raise ValueError(
            "Candidate grid is empty — check "
            "train_monthly.prophet.tuning.candidate_grid in the parameter file."
        )
    logger.info("Candidate configurations to train: %d", len(candidate_configs))

    # Prophet requires columns named exactly 'ds' and 'y'
    train_fit_df = train_df[[date_col, target_col] + active_regressors].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    # Prediction input must include 'ds' and regressors but NOT 'y'
    val_pred_df = val_df[[date_col] + active_regressors].rename(
        columns={date_col: "ds"}
    )
    y_true = val_df[target_col].values.astype(float)

    # ── train and evaluate each candidate ────────────────────────────────────────
    tuning_rows: list[dict] = []
    metrics_rows: list[dict] = []
    trained_models: dict[str, Prophet] = {}

    for idx, config in enumerate(candidate_configs):
        candidate_id = f"prophet_candidate_{idx + 1:03d}"
        trained_at = datetime.now(tz=UTC).isoformat()
        logger.info(
            "[%d/%d] Training %s …",
            idx + 1,
            len(candidate_configs),
            candidate_id,
        )

        try:
            model = _create_prophet_model(config, active_regressors, regressor_mode)
            model.fit(train_fit_df)

            forecast = _forecast_validation_period(model, val_pred_df, active_regressors)
            y_pred = forecast["yhat"].values.astype(float)

            base = _compute_forecast_metrics(y_true, y_pred, epsilon, precision_threshold)
            horiz = _compute_horizon_metrics(
                val_df,
                y_pred,
                date_col,
                target_col,
                horizons,
                epsilon,
                precision_threshold,
            )

            logger.info(
                "%s → mape=%.4f  rmse=%.2f  wmape=%.4f  precision=%.4f",
                candidate_id,
                base["mape"],
                base["rmse"],
                base["wmape"],
                base["forecast_precision"],
            )

            trained_models[candidate_id] = model
            tuning_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "error_message": None,
                    "selection_metric": selection_metric,
                    "selection_metric_value": base[selection_metric],
                    **base,
                    # Only include horizon mape columns in tuning summary
                    **{k: v for k, v in horiz.items() if k.endswith("_mape")},
                    **config,
                    "active_regressors": ",".join(active_regressors),
                    "trained_at": trained_at,
                }
            )
            metrics_rows.append(
                {
                    "candidate_id": candidate_id,
                    **base,
                    **horiz,
                    "validation_start_date": str(val_df[date_col].min().date()),
                    "validation_end_date": str(val_df[date_col].max().date()),
                    "validation_rows": len(val_df),
                }
            )

        except Exception as exc:  # noqa: BLE001 — intentional: isolate per-candidate failures
            logger.error(
                "Candidate %s failed: %s | config=%s", candidate_id, exc, config
            )
            tuning_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error_message": str(exc),
                    "selection_metric": selection_metric,
                    "selection_metric_value": None,
                    **config,
                    "active_regressors": ",".join(active_regressors),
                    "trained_at": trained_at,
                }
            )

    if not any(r["status"] == "success" for r in tuning_rows):
        raise RuntimeError(
            "All Prophet candidates failed during training. Check logs for details."
        )

    # ── rank candidates and select pre-champions ──────────────────────────────────
    tuning_df = pd.DataFrame(tuning_rows)
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    ranked_df = _rank_candidates(tuning_df, selection_metric, top_n)

    prechampion_ids: list[str] = ranked_df.loc[
        ranked_df["is_prechampion"].eq(True), "candidate_id"
    ].tolist()
    logger.info("Pre-champion candidates (rank 1–%d): %s", top_n, prechampion_ids)

    # Report whether any pre-champion clears the business success threshold
    prechampion_precisions = ranked_df.loc[
        ranked_df["is_prechampion"].eq(True), "forecast_precision"
    ].dropna()
    if not prechampion_precisions.empty and (
        prechampion_precisions >= precision_threshold
    ).any():
        logger.info(
            "At least one pre-champion meets the business success threshold "
            "(≥%.0f%% precision).",
            precision_threshold * 100,
        )
    else:
        logger.warning(
            "No pre-champion exceeds the %.0f%% precision threshold. "
            "Best precision among pre-champions: %.4f",
            precision_threshold * 100,
            prechampion_precisions.max() if not prechampion_precisions.empty else 0.0,
        )

    # ── build output artifacts ────────────────────────────────────────────────────
    prechampion_configs = _build_prechampion_configs(
        ranked_df, metrics_df, active_regressors, selection_metric, top_n
    )
    # Save only top-N pre-champion models to keep the artifact lightweight
    candidate_models = {
        cid: trained_models[cid]
        for cid in prechampion_ids
        if cid in trained_models
    }
    best_model = trained_models.get(prechampion_ids[0]) if prechampion_ids else None

    logger.info(
        "Outputs — tuning_results=%s  validation_metrics=%s  "
        "prechampions=%d  saved_models=%d",
        ranked_df.shape,
        metrics_df.shape,
        len(prechampion_configs.get("prechampions", [])),
        len(candidate_models),
    )

    return ranked_df, metrics_df, prechampion_configs, candidate_models, best_model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_prophet_training_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for any invalid training or validation input.

    Args:
        train_df: Training DataFrame to validate.
        val_df: Validation DataFrame to validate.
        active_regressors: Regressor columns that must exist in both DataFrames.
        date_col: Name of the date column.
        target_col: Name of the target column (required in train, not in val).
    """
    if train_df.empty:
        raise ValueError("Training DataFrame is empty.")
    if val_df.empty:
        raise ValueError("Validation DataFrame is empty.")

    required_train = [date_col, target_col] + active_regressors
    missing_train = [c for c in required_train if c not in train_df.columns]
    if missing_train:
        raise ValueError(
            f"Required columns missing from training data: {missing_train}"
        )

    # Validation needs date + regressors for prediction; y is not required
    required_val = [date_col] + active_regressors
    missing_val = [c for c in required_val if c not in val_df.columns]
    if missing_val:
        raise ValueError(
            f"Required columns missing from validation data: {missing_val}"
        )

    if not pd.api.types.is_numeric_dtype(train_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {train_df[target_col].dtype}."
        )

    # Null checks on active regressors
    for col in active_regressors:
        null_train = int(train_df[col].isnull().sum())
        if null_train > 0:
            raise ValueError(
                f"Active regressor {col!r} has {null_train} null value(s) in training data."
            )
        null_val = int(val_df[col].isnull().sum())
        if null_val > 0:
            raise ValueError(
                f"Active regressor {col!r} has {null_val} null value(s) in validation data."
            )


def _build_candidate_grid(grid_params: dict) -> list[dict]:
    """Expand a hyperparameter grid dict into a list of candidate configurations.

    Args:
        grid_params: Mapping from hyperparameter name to list of values.
            Key order is preserved (follows the YAML declaration order).

    Returns:
        List of dicts where each entry is one combination of hyperparameter values.
        The order is deterministic (itertools.product over the declared key order).
    """
    if not grid_params:
        return []
    keys = list(grid_params.keys())
    # Convert OmegaConf list containers to plain Python lists
    value_lists = [list(grid_params[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _create_prophet_model(
    candidate_config: dict,
    active_regressors: list[str],
    regressor_mode: str,
) -> Prophet:
    """Instantiate a Prophet model and register all active regressors.

    Args:
        candidate_config: Hyperparameter values for this candidate
            (changepoint_prior_scale, seasonality_prior_scale, etc.).
        active_regressors: Column names to register via add_regressor().
        regressor_mode: Mode for all regressors ('additive' or 'multiplicative').

    Returns:
        Configured but unfitted Prophet model.
    """
    model = Prophet(
        changepoint_prior_scale=float(
            candidate_config.get("changepoint_prior_scale", 0.05)
        ),
        seasonality_prior_scale=float(
            candidate_config.get("seasonality_prior_scale", 10.0)
        ),
        holidays_prior_scale=float(
            candidate_config.get("holidays_prior_scale", 10.0)
        ),
        seasonality_mode=str(candidate_config.get("seasonality_mode", "additive")),
        yearly_seasonality=bool(candidate_config.get("yearly_seasonality", True)),
        weekly_seasonality=bool(candidate_config.get("weekly_seasonality", False)),
        daily_seasonality=bool(candidate_config.get("daily_seasonality", False)),
        interval_width=float(candidate_config.get("interval_width", 0.8)),
    )
    # Register regressors before fit; all share the same mode for this MVP
    for regressor_name in active_regressors:
        model.add_regressor(regressor_name, mode=regressor_mode)
    return model


def _forecast_validation_period(
    model: Prophet,
    val_df: pd.DataFrame,
    active_regressors: list[str],
) -> pd.DataFrame:
    """Generate validation-period forecasts from a fitted Prophet model.

    Args:
        model: Fitted Prophet model.
        val_df: Validation input with columns 'ds' and all active regressors.
            Must not contain the target column 'y'.
        active_regressors: Regressor column names registered with the model.

    Returns:
        Prophet forecast DataFrame with yhat, yhat_lower, yhat_upper, etc.
    """
    predict_cols = ["ds"] + active_regressors
    return model.predict(val_df[predict_cols])


def _compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float,
    precision_threshold: float,
) -> dict:
    """Compute scalar forecast accuracy metrics for a single candidate.

    MAPE uses an epsilon additive guard in the denominator to avoid division by
    zero on months with zero demand. forecast_precision = 1 - mape.

    Args:
        y_true: Observed values.
        y_pred: Forecasted values, same shape as y_true.
        epsilon: Small constant added to |y_true| in percentage metric denominators.
        precision_threshold: Threshold above which business_success_flag is True.

    Returns:
        Dict with mae, rmse, mape, wmape, forecast_precision, business_success_flag.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_true - y_pred)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(abs_errors / (np.abs(y_true) + epsilon)))
    wmape = float(np.sum(abs_errors) / (np.sum(np.abs(y_true)) + epsilon))
    forecast_precision = 1.0 - mape
    business_success_flag = bool(forecast_precision >= precision_threshold)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "wmape": wmape,
        "forecast_precision": forecast_precision,
        "business_success_flag": business_success_flag,
    }


def _compute_horizon_metrics(  # noqa: PLR0913
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    date_col: str,
    target_col: str,
    horizons: list[int],
    epsilon: float,
    precision_threshold: float,
) -> dict:
    """Compute point metrics at specified forecast horizons.

    Horizon h is the h-th validation month when sorted ascending by date
    (horizon 1 = first month, horizon 2 = second, …).  Single-point metrics are
    computed by treating that month as both y_true and y_pred (length-1 arrays).

    Args:
        val_df: Validation DataFrame with date and target columns (sorted).
        y_pred: Forecasted values aligned positionally with val_df sorted by date.
        date_col: Date column name in val_df.
        target_col: Target column name in val_df.
        horizons: List of 1-based horizon integers to evaluate.
        epsilon: Division guard for percentage metrics.
        precision_threshold: Threshold for business success flag.

    Returns:
        Dict with horizon_{h}_mae/mape/wmape/forecast_precision for each h in horizons.
    """
    val_sorted = val_df.sort_values(date_col).reset_index(drop=True)
    y_true_all = val_sorted[target_col].values.astype(float)
    result: dict = {}

    for h in horizons:
        idx = h - 1  # convert 1-based horizon to 0-based index
        if idx >= len(val_sorted):
            logger.warning(
                "Validation set has %d rows; horizon-%d metrics unavailable.",
                len(val_sorted),
                h,
            )
            result[f"horizon_{h}_mae"] = None
            result[f"horizon_{h}_mape"] = None
            result[f"horizon_{h}_wmape"] = None
            result[f"horizon_{h}_forecast_precision"] = None
        else:
            pt_metrics = _compute_forecast_metrics(
                np.array([y_true_all[idx]]),
                np.array([y_pred[idx]]),
                epsilon,
                precision_threshold,
            )
            result[f"horizon_{h}_mae"] = pt_metrics["mae"]
            result[f"horizon_{h}_mape"] = pt_metrics["mape"]
            result[f"horizon_{h}_wmape"] = pt_metrics["wmape"]
            result[f"horizon_{h}_forecast_precision"] = pt_metrics["forecast_precision"]

    return result


def _rank_candidates(
    tuning_df: pd.DataFrame,
    selection_metric: str,
    top_n: int,
) -> pd.DataFrame:
    """Rank successful candidates by selection metric and mark top-N as pre-champions.

    Error metrics (mae, rmse, mape, wmape) are ranked ascending (lower is better).
    Precision-style metrics (forecast_precision) are ranked descending (higher is better).
    Failed candidates are appended last with rank=None and is_prechampion=False.

    Args:
        tuning_df: DataFrame with one row per candidate including a 'status' column.
        selection_metric: Column name to rank by (e.g. 'mape').
        top_n: Number of candidates to flag as pre-champions.

    Returns:
        Updated DataFrame with 'rank' (int, nullable) and 'is_prechampion' (bool) columns.
    """
    success_mask = tuning_df["status"] == "success"
    success_df = tuning_df[success_mask].copy()
    failed_df = tuning_df[~success_mask].copy()

    ascending = selection_metric not in {"forecast_precision"}
    success_df = success_df.sort_values(
        "selection_metric_value", ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    success_df["rank"] = list(range(1, len(success_df) + 1))
    success_df["is_prechampion"] = success_df["rank"] <= top_n

    failed_df["rank"] = None
    failed_df["is_prechampion"] = False

    return pd.concat([success_df, failed_df], ignore_index=True)


def _build_prechampion_configs(
    ranked_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    active_regressors: list[str],
    selection_metric: str,
    top_n: int,
) -> dict:
    """Build the prechampion_configs artifact consumed by the model-selection stage.

    Args:
        ranked_df: Ranked tuning results with is_prechampion flags.
        metrics_df: Detailed per-candidate validation metrics.
        active_regressors: Ordered list of regressor column names.
        selection_metric: Metric used for ranking.
        top_n: Configured number of pre-champions.

    Returns:
        Nested dict ready for JSON serialisation.
    """
    prechampions_df = ranked_df[ranked_df["is_prechampion"].eq(True)].sort_values(
        "rank"
    )

    prechampions = []
    for _, row in prechampions_df.iterrows():
        cid = str(row["candidate_id"])

        # Convert metrics row to plain dict for safe JSON serialisation
        if not metrics_df.empty and cid in metrics_df["candidate_id"].values:
            m: dict = metrics_df[metrics_df["candidate_id"] == cid].iloc[0].to_dict()
        else:
            m = {}

        prechampions.append(
            {
                "candidate_id": cid,
                "rank": _safe_int(row.get("rank")),
                "validation_metrics": {
                    "mae": _safe_float(m.get("mae")),
                    "rmse": _safe_float(m.get("rmse")),
                    "mape": _safe_float(m.get("mape")),
                    "wmape": _safe_float(m.get("wmape")),
                    "forecast_precision": _safe_float(m.get("forecast_precision")),
                    "horizon_2_mape": _safe_float(m.get("horizon_2_mape")),
                    "horizon_3_mape": _safe_float(m.get("horizon_3_mape")),
                },
                "model_params": {
                    "changepoint_prior_scale": _safe_float(
                        row.get("changepoint_prior_scale")
                    ),
                    "seasonality_prior_scale": _safe_float(
                        row.get("seasonality_prior_scale")
                    ),
                    "holidays_prior_scale": _safe_float(
                        row.get("holidays_prior_scale")
                    ),
                    "seasonality_mode": str(row.get("seasonality_mode", "")),
                    "yearly_seasonality": bool(row.get("yearly_seasonality", True)),
                    "weekly_seasonality": bool(row.get("weekly_seasonality", False)),
                    "daily_seasonality": bool(row.get("daily_seasonality", False)),
                    "interval_width": _safe_float(row.get("interval_width")),
                },
                "active_regressors": active_regressors,
            }
        )

    return {
        "model_family": "prophet",
        "granularity": "monthly",
        "selection_metric": selection_metric,
        "top_n_prechampions": top_n,
        "prechampions": prechampions,
    }


def _safe_float(value: Any) -> float | None:
    """Convert to Python float, returning None for missing or non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert to Python int, returning None for missing values."""
    if value is None:
        return None
    try:
        f = float(value)
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
