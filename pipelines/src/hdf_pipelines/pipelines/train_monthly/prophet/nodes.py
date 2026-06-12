"""Monthly Prophet training and tuning nodes (rolling-origin protocol).

Implements Optuna Bayesian hyperparameter tuning for Prophet on monthly demand.
Each trial is scored by a **rolling-origin backtest**: for every cycle the model
is refit on history through the cycle origin and forecasts the next ``H`` months;
per-cycle metrics are macro-averaged. The Optuna objective is the averaged
``WMAPE_M3`` (protocol §3.5, §4). The top-N pre-champions are persisted (refit on
full history) for the model-selection stage, which chooses champions directly from
these rolling-origin metrics — there is no separate hold-out test stage.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from shared.rolling_origin import RollingOriginCycle, run_rolling_origin

from hdf_pipelines.pipelines.train_monthly.nodes import (
    build_monthly_rolling_origin_cycles,
    extract_rolling_origin_metric_set,
    make_pruning_callback,
    supported_rolling_origin_metrics,
)
from hdf_pipelines.utils import (
    build_rolling_origin_pruner,
    create_optuna_study,
    serialize_optuna_trial,
    suggest_trial_params,
    validate_objective_metric_direction,
    validate_optuna_search_space,
)

# Suppress verbose Stan/cmdstanpy progress output during Prophet fitting
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Public node function


def train_and_evaluate_monthly_prophet_candidates(  # noqa: PLR0915
    monthly_prophet_full_train: pd.DataFrame,
    monthly_prophet_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, Any]:
    """Tune Prophet with a rolling-origin backtest and persist pre-champions.

    Runs one ephemeral Optuna study. Each trial is evaluated by refitting Prophet
    at every rolling-origin cycle and forecasting the next ``H`` months; the
    objective is the macro-averaged ``WMAPE_M3``. The top-N pre-champions are refit
    on full history (through ``L``) and persisted for model selection.

    Args:
        monthly_prophet_full_train: Prophet-ready full history (through ``L``) with
            columns ds, y, sku, and all active regressor columns.
        monthly_prophet_split_metadata: Metadata from model_input_preparation
            (date range, active_regressors). Used for cross-checking regressors;
            params is the authoritative source for pipeline configuration.
        params: Contents of ``train_monthly.prophet`` from the parameter file.

    Returns:
        Six-element tuple:

        1. ``tuning_results`` — DataFrame, one row per candidate (ranked).
        2. ``rolling_origin_metrics`` — DataFrame, per-candidate macro-averaged metrics.
        3. ``prechampion_configs`` — Dict with top-N pre-champion configurations.
        4. ``candidate_models`` — Dict mapping candidate_id → full-history Prophet model
           (top-N only).
        5. ``training_metadata`` — Dict summarizing the Optuna study and trials.
        6. ``best_prophet_model`` — Rank-1 full-history Prophet model.

    Raises:
        ValueError: When inputs or the Optuna configuration are invalid.
        RuntimeError: When every trial fails to evaluate.
    """
    # ── extract configuration ─────────────────────────────────────────────────────
    date_col: str = params.get("date_column", "ds")
    target_col: str = params.get("target_column", "y")
    active_regressors: list[str] = list(params.get("active_regressors", []))
    regressor_mode: str = params.get("regressors", {}).get("mode", "additive")

    tuning_cfg: dict = params.get("tuning", {})
    optimizer: str = str(tuning_cfg.get("optimizer", "optuna")).lower()
    objective_cfg: dict = tuning_cfg.get("objective", {})
    selection_metric: str = str(objective_cfg.get("metric", "wmape_m3"))
    objective_direction: str = str(objective_cfg.get("direction", "minimize")).lower()
    max_trials: int = int(tuning_cfg.get("max_trials", 30))
    top_n: int = int(tuning_cfg.get("top_n_prechampions", 3))
    sampler_cfg: dict = dict(tuning_cfg.get("sampler", {}))
    pruning_cfg: dict = dict(tuning_cfg.get("pruning", {}))
    rolling_origin_cfg: dict = dict(tuning_cfg.get("rolling_origin", {}))
    horizon: int = int(rolling_origin_cfg.get("horizon", 3))
    search_space = validate_optuna_search_space(
        dict(tuning_cfg.get("search_space", {}))
    )
    fixed_params: dict = dict(tuning_cfg.get("fixed_params", {}))

    metrics_cfg: dict = params.get("metrics", {})
    epsilon: float = float(metrics_cfg.get("epsilon", 1.0))
    mase_seasonal_period: int = int(metrics_cfg.get("mase_seasonal_period", 12))

    supported_metrics = supported_rolling_origin_metrics(horizon)
    selection_metric, objective_direction = validate_objective_metric_direction(
        selection_metric,
        objective_direction,
        supported_metrics,
    )

    if optimizer != "optuna":
        raise ValueError(
            f"Unsupported monthly Prophet optimizer {optimizer!r}. Only 'optuna' is supported."
        )
    if max_trials <= 0:
        raise ValueError(
            "train_monthly.prophet.tuning.max_trials must be a positive integer."
        )

    # ── validate inputs ───────────────────────────────────────────────────────────
    _validate_prophet_training_inputs(
        monthly_prophet_full_train, active_regressors, date_col, target_col
    )

    metadata_regressors = monthly_prophet_split_metadata.get("active_regressors", [])
    if set(active_regressors) != set(metadata_regressors):
        logger.warning(
            "Active regressors in params differ from split metadata. "
            "Using params as the authoritative source. Params: %s | Metadata: %s",
            active_regressors, metadata_regressors,
        )

    # ── prepare full-history frame and rolling-origin cycles ──────────────────────
    full_df = monthly_prophet_full_train.copy()
    full_df[date_col] = pd.to_datetime(full_df[date_col])
    full_df = full_df.sort_values(date_col).reset_index(drop=True)

    cycles = build_monthly_rolling_origin_cycles(full_df, date_col, rolling_origin_cfg)
    logger.info(
        "Full history: %d rows | %s → %s | %d rolling-origin cycles (H=%d), "
        "last cycle targets %s",
        len(full_df),
        full_df[date_col].min().date(),
        full_df[date_col].max().date(),
        len(cycles), horizon,
        [d.strftime("%Y-%m-%d") for d in cycles[-1].target_dates],
    )
    logger.info(
        "Optuna study — metric=%s  direction=%s  max_trials=%d  pruning=%s",
        selection_metric, objective_direction, max_trials,
        bool(pruning_cfg.get("enabled", False)),
    )

    # ── train and evaluate each candidate ────────────────────────────────────────
    metrics_rows: list[dict] = []
    trial_configs: dict[str, dict] = {}
    study = create_optuna_study(
        objective_direction,
        sampler_cfg,
        pruner=build_rolling_origin_pruner(pruning_cfg),
    )

    def objective(trial: optuna.Trial) -> float:
        candidate_id = f"prophet_candidate_{trial.number + 1:03d}"
        config = suggest_trial_params(trial, search_space, fixed_params)
        trial.set_user_attr("candidate_id", candidate_id)
        trial.set_user_attr("trained_at", datetime.now(tz=UTC).isoformat())
        trial.set_user_attr("config", dict(config))
        trial_configs[candidate_id] = dict(config)

        def _fit_forecast(train_df: pd.DataFrame, cycle: RollingOriginCycle) -> np.ndarray:
            return _prophet_cycle_forecast(
                config, train_df, cycle, full_df, date_col, target_col,
                active_regressors, regressor_mode,
            )

        on_cycle_end = make_pruning_callback(trial, pruning_cfg, metric_key=selection_metric)
        _, aggregated = run_rolling_origin(
            full_df, date_col, target_col, cycles, _fit_forecast,
            season=mase_seasonal_period, epsilon=epsilon, on_cycle_end=on_cycle_end,
        )
        metric_set = extract_rolling_origin_metric_set(aggregated, horizon)
        selection_value = metric_set.get(selection_metric)

        if selection_value is None:
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error_message", "all rolling-origin cycles failed")
            raise RuntimeError(f"{candidate_id}: all rolling-origin cycles failed")

        trial.set_user_attr("status", "success")
        trial.set_user_attr("metrics", metric_set)
        metrics_rows.append(
            {"candidate_id": candidate_id, "trial_number": trial.number, **metric_set}
        )
        logger.info(
            "[%d/%d] %s → wmape_m3=%.4f  wmape=%.4f  mase=%s  bias=%.4f  (%s/%s cycles)",
            trial.number + 1, max_trials, candidate_id,
            metric_set["wmape_m3"], metric_set["wmape"],
            f"{metric_set['mase']:.4f}" if metric_set["mase"] is not None else "nan",
            metric_set["bias"], metric_set["n_cycles_evaluated"], metric_set["n_cycles"],
        )
        return float(selection_value)

    study.optimize(objective, n_trials=max_trials, catch=(Exception,))

    tuning_rows = _build_tuning_rows(
        study, selection_metric, objective_direction, optimizer,
        fixed_params, active_regressors,
    )
    if not any(r["status"] == "success" for r in tuning_rows):
        raise RuntimeError(
            "All Prophet Optuna trials failed during rolling-origin evaluation. "
            "Check logs for details."
        )

    # ── rank candidates and select pre-champions ──────────────────────────────────
    tuning_df = pd.DataFrame(tuning_rows)
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    ranked_df = _rank_candidates(tuning_df, objective_direction, top_n)

    prechampion_ids: list[str] = ranked_df.loc[
        ranked_df["is_prechampion"].eq(True), "candidate_id"
    ].tolist()
    logger.info("Pre-champion candidates (rank 1–%d): %s", top_n, prechampion_ids)

    # ── refit pre-champions on full history and build artifacts ───────────────────
    candidate_models: dict[str, Prophet] = {}
    for cid in prechampion_ids:
        cfg = trial_configs.get(cid)
        if cfg is None:
            continue
        model = _create_prophet_model(cfg, active_regressors, regressor_mode)
        model.fit(
            full_df[[date_col, target_col, *active_regressors]].rename(
                columns={date_col: "ds", target_col: "y"}
            )
        )
        candidate_models[cid] = model

    prechampion_configs = _build_prechampion_configs(
        ranked_df, metrics_df, active_regressors, selection_metric, top_n
    )
    training_metadata = _build_training_metadata(
        study, ranked_df, selection_metric, objective_direction, optimizer,
        sampler_cfg, max_trials, top_n, fixed_params, rolling_origin_cfg,
    )
    best_model = candidate_models.get(prechampion_ids[0]) if prechampion_ids else None

    logger.info(
        "Outputs — tuning_results=%s  rolling_origin_metrics=%s  prechampions=%d  "
        "saved_models=%d  trials=%d",
        ranked_df.shape, metrics_df.shape,
        len(prechampion_configs.get("prechampions", [])),
        len(candidate_models), len(study.trials),
    )

    return (
        ranked_df,
        metrics_df,
        prechampion_configs,
        candidate_models,
        training_metadata,
        best_model,
    )


# Private helpers


def _validate_prophet_training_inputs(
    full_df: pd.DataFrame,
    active_regressors: list[str],
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for any invalid full-history training input."""
    if full_df.empty:
        raise ValueError("Prophet full-history DataFrame is empty.")

    required = [date_col, target_col] + active_regressors
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(f"Required columns missing from full-history data: {missing}")

    if not pd.api.types.is_numeric_dtype(full_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {full_df[target_col].dtype}."
        )
    for col in active_regressors:
        n_null = int(full_df[col].isnull().sum())
        if n_null > 0:
            raise ValueError(
                f"Active regressor {col!r} has {n_null} null value(s) in full history."
            )


def _prophet_cycle_forecast(  # noqa: PLR0913
    config: dict,
    train_df: pd.DataFrame,
    cycle: RollingOriginCycle,
    full_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    active_regressors: list[str],
    regressor_mode: str,
) -> np.ndarray:
    """Refit Prophet on a cycle's train window and forecast its target months.

    Future regressor values for the (observed) target months are taken from the
    full-history frame, so the forecast uses exactly the known exogenous inputs.

    Returns:
        Array of ``len(cycle.target_dates)`` point forecasts in chronological order.
    """
    fit_df = train_df[[date_col, target_col, *active_regressors]].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    model = _create_prophet_model(config, active_regressors, regressor_mode)
    model.fit(fit_df)

    target_set = pd.DatetimeIndex(cycle.target_dates)
    target_rows = (
        full_df[full_df[date_col].isin(target_set)]
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    future_df = target_rows[[date_col, *active_regressors]].rename(
        columns={date_col: "ds"}
    )
    forecast = model.predict(future_df)
    return forecast["yhat"].to_numpy(dtype=float)


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
        holidays_prior_scale=float(candidate_config.get("holidays_prior_scale", 10.0)),
        seasonality_mode=str(candidate_config.get("seasonality_mode", "additive")),
        yearly_seasonality=bool(candidate_config.get("yearly_seasonality", True)),
        weekly_seasonality=bool(candidate_config.get("weekly_seasonality", False)),
        daily_seasonality=bool(candidate_config.get("daily_seasonality", False)),
        interval_width=float(candidate_config.get("interval_width", 0.8)),
    )
    for regressor_name in active_regressors:
        prior_scale_val = candidate_config.get(f"prior_scale_{regressor_name}")
        kwargs: dict = {"mode": regressor_mode}
        if prior_scale_val is not None:
            kwargs["prior_scale"] = float(prior_scale_val)
        model.add_regressor(regressor_name, **kwargs)
    return model


def _rank_candidates(
    tuning_df: pd.DataFrame,
    objective_direction: str,
    top_n: int,
) -> pd.DataFrame:
    """Rank successful candidates by selection metric and mark top-N pre-champions.

    Failed/pruned candidates are appended last with rank=None and is_prechampion=False.

    Args:
        tuning_df: DataFrame with one row per candidate including a 'status' column.
        objective_direction: Optuna study direction ('minimize' or 'maximize').
        top_n: Number of candidates to flag as pre-champions.

    Returns:
        Updated DataFrame with 'rank' (int, nullable) and 'is_prechampion' (bool).
    """
    success_mask = tuning_df["status"] == "success"
    success_df = tuning_df[success_mask].copy()
    other_df = tuning_df[~success_mask].copy()

    ascending = objective_direction == "minimize"
    success_df = success_df.sort_values(
        "selection_metric_value", ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    success_df["rank"] = list(range(1, len(success_df) + 1))
    success_df["is_prechampion"] = success_df["rank"] <= top_n

    other_df["rank"] = None
    other_df["is_prechampion"] = False

    return pd.concat([success_df, other_df], ignore_index=True)


def _build_prechampion_configs(
    ranked_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    active_regressors: list[str],
    selection_metric: str,
    top_n: int,
) -> dict:
    """Build the prechampion_configs artifact consumed by model selection.

    Stores each pre-champion's hyperparameters and macro-averaged rolling-origin
    metrics so selection can rank families directly on these metrics.

    Returns:
        Nested dict ready for JSON serialisation.
    """
    prechampions_df = ranked_df[ranked_df["is_prechampion"].eq(True)].sort_values(
        "rank"
    )

    prechampions = []
    for _, row in prechampions_df.iterrows():
        cid = str(row["candidate_id"])
        if not metrics_df.empty and cid in metrics_df["candidate_id"].values:
            m: dict = metrics_df[metrics_df["candidate_id"] == cid].iloc[0].to_dict()
        else:
            m = {}

        regressor_prior_scales = {
            str(k): _safe_float(row.get(k))
            for k in row.index
            if str(k).startswith("prior_scale_")
        }
        prechampions.append(
            {
                "candidate_id": cid,
                "rank": _safe_int(row.get("rank")),
                "rolling_origin_metrics": _metric_set_from_row(m),
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
                    **regressor_prior_scales,
                },
                "active_regressors": active_regressors,
            }
        )

    return {
        "model_family": "prophet",
        "granularity": "monthly",
        "selection_stage": "rolling_origin",
        "selection_metric": selection_metric,
        "top_n_prechampions": top_n,
        "prechampions": prechampions,
    }


def _metric_set_from_row(m: dict) -> dict:
    """Extract the rolling-origin metric set from a metrics row dict."""
    keys = [
        "wmape", "wmape_m1", "wmape_m2", "wmape_m3", "mase", "bias", "rmse",
        "wmape_m3_std", "wmape_std", "n_cycles", "n_cycles_evaluated",
    ]
    return {k: _safe_float(m.get(k)) for k in keys}


def _build_tuning_rows(  # noqa: PLR0913
    study: optuna.study.Study,
    selection_metric: str,
    objective_direction: str,
    optimizer: str,
    fixed_params: dict[str, Any],
    active_regressors: list[str],
) -> list[dict]:
    """Build ranked tuning rows from Optuna trial history."""
    rows: list[dict] = []
    for trial in study.trials:
        candidate_id = str(
            trial.user_attrs.get(
                "candidate_id", f"prophet_candidate_{trial.number + 1:03d}"
            )
        )
        config = dict(trial.user_attrs.get("config", {}))
        if not config:
            config = dict(trial.params)
            config.update(fixed_params)
        metrics = dict(trial.user_attrs.get("metrics", {}))
        rows.append(
            {
                "candidate_id": candidate_id,
                "trial_number": int(trial.number),
                "status": _trial_status(trial),
                "error_message": trial.user_attrs.get("error_message"),
                "optimizer": optimizer,
                "objective_direction": objective_direction,
                "selection_metric": selection_metric,
                "selection_metric_value": _safe_float(metrics.get(selection_metric)),
                **metrics,
                **config,
                "active_regressors": ",".join(active_regressors),
                "trained_at": trial.user_attrs.get("trained_at"),
            }
        )
    return rows


def _trial_status(trial: optuna.trial.FrozenTrial) -> str:
    """Map an Optuna trial state to a status label."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        return "success"
    if trial.state == optuna.trial.TrialState.PRUNED:
        return "pruned"
    return "failed"


def _build_training_metadata(  # noqa: PLR0913
    study: optuna.study.Study,
    ranked_df: pd.DataFrame,
    selection_metric: str,
    objective_direction: str,
    optimizer: str,
    sampler_cfg: dict[str, Any],
    max_trials: int,
    top_n: int,
    fixed_params: dict[str, Any],
    rolling_origin_cfg: dict[str, Any],
) -> dict:
    """Summarize the Optuna study into a Kedro JSON artifact."""
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    best_candidate_id = None
    best_trial_number = None
    best_value = None
    if not ranked_df.empty:
        best_row = ranked_df[ranked_df["rank"] == 1]
        if not best_row.empty:
            best_candidate_id = str(best_row.iloc[0]["candidate_id"])
            best_trial_number = _safe_int(best_row.iloc[0].get("trial_number"))
            best_value = _safe_float(best_row.iloc[0].get("selection_metric_value"))

    return {
        "model_family": "prophet",
        "granularity": "monthly",
        "evaluation_mode": "rolling_origin",
        "optimizer": optimizer,
        "objective": {"metric": selection_metric, "direction": objective_direction},
        "rolling_origin": {
            "horizon": int(rolling_origin_cfg.get("horizon", 3)),
            "n_cycles": int(rolling_origin_cfg.get("n_cycles", 5)),
            "window": str(rolling_origin_cfg.get("window", "expanding")),
            "step_months": int(rolling_origin_cfg.get("step_months", 1)),
        },
        "sampler": {
            "name": str(sampler_cfg.get("name", "tpe")).lower(),
            "seed": _safe_int(sampler_cfg.get("seed")),
        },
        "max_trials": max_trials,
        "top_n_prechampions": top_n,
        "completed_trials": len(completed),
        "pruned_trials": len(pruned),
        "failed_trials": len(failed),
        "best_trial_number": best_trial_number,
        "best_candidate_id": best_candidate_id,
        "best_value": best_value,
        "fixed_params": dict(fixed_params),
        "trials": [
            serialize_optuna_trial(trial, fixed_params=fixed_params)
            for trial in study.trials
        ],
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
