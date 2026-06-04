"""Monthly CatBoost training nodes: Optuna TPE search and artifact emission.

Implements an Optuna TPE study over the configured CatBoost hyperparameter
search space. Each trial is fit on the training split, evaluated on the
validation split, ranked by the configured primary metric (default: WAPE),
and the top-N pre-champions are persisted as catalog artifacts.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from shared.metrics import mase as _shared_mase
from shared.metrics import wape as _shared_wape

from hdf_pipelines.utils.optuna_helpers import (
    create_optuna_study,
    suggest_trial_params,
    validate_optuna_search_space,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Identity columns that must never appear in the feature matrix.
_IDENTITY_COLUMNS = {"month_start_date", "monthly_demand", "sku", "ds", "y"}


# ── Public node ───────────────────────────────────────────────────────────────


def train_monthly_catboost_candidates(  # noqa: PLR0915
    monthly_catboost_train: pd.DataFrame,
    monthly_catboost_validation: pd.DataFrame,
    monthly_catboost_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Train CatBoost candidates via Optuna TPE and evaluate on validation split.

    Runs an Optuna TPE study over the configured hyperparameter search space.
    Each trial fits a CatBoostRegressor on the training split and evaluates it
    on the validation split. Successful trials are ranked by the primary metric
    (WAPE by default); the top-N are persisted as pre-champion artifacts for the
    downstream model-selection stage.

    Args:
        monthly_catboost_train: CatBoost-ready training DataFrame with all feature
            columns, the target column, the date column, and the SKU column.
        monthly_catboost_validation: CatBoost-ready validation DataFrame with the
            same schema.
        monthly_catboost_split_metadata: Metadata dict produced by the CatBoost
            adapter (``adapt_monthly_data_for_catboost``). Used to read
            ``all_feature_columns``; feature columns are inferred from the
            DataFrame columns when the key is absent.
        params: Contents of ``train_monthly.catboost`` from the parameter file.
            Expected keys: ``primary_metric``, ``selection_direction``,
            ``random_seed``, ``early_stopping_rounds``, ``prechampion_count``,
            ``mase_seasonal_period``, ``epsilon``, ``max_trials``,
            ``max_failed_trials``, ``sampler``, ``search_space``, and
            ``fixed_params``.

    Returns:
        Five-element tuple:

        1. ``tuning_results`` — DataFrame, one row per trial (ranked).
        2. ``validation_metrics`` — DataFrame, detailed per-trial metrics.
        3. ``prechampion_configs`` — Dict with top-N pre-champion configurations.
        4. ``candidate_models`` — Dict mapping candidate_id → fitted
           ``CatBoostRegressor`` (top-N only).
        5. ``training_metadata`` — Dict summarising the Optuna study run.

    Raises:
        ValueError: When inputs are invalid or required columns are missing.
        RuntimeError: When every trial fails to train.
    """
    # ── extract configuration ─────────────────────────────────────────────────
    primary_metric: str = str(params.get("primary_metric", "wape")).lower()
    selection_direction: str = str(params.get("selection_direction", "minimize")).lower()
    random_seed: int = int(params.get("random_seed", 42))
    early_stopping_rounds: int = int(params.get("early_stopping_rounds", 50))
    prechampion_count: int = int(params.get("prechampion_count", 3))
    mase_seasonal_period: int = int(params.get("mase_seasonal_period", 12))
    epsilon: float = float(params.get("epsilon", 1.0))
    max_trials: int = int(params.get("max_trials", 30))
    max_failed_trials: int | None = (
        int(params["max_failed_trials"])
        if params.get("max_failed_trials") is not None
        else None
    )
    sampler_config: dict = dict(params.get("sampler", {"name": "tpe", "seed": random_seed}))
    search_space: dict = validate_optuna_search_space(dict(params.get("search_space", {})))
    fixed_params: dict = dict(params.get("fixed_params", {}))

    date_col: str = str(
        monthly_catboost_split_metadata.get("date_column", "month_start_date")
    )
    target_col: str = str(
        monthly_catboost_split_metadata.get("target_column", "monthly_demand")
    )
    sku_col: str = str(monthly_catboost_split_metadata.get("sku_column", "sku"))

    # ── resolve feature columns ───────────────────────────────────────────────
    feature_columns = _resolve_feature_columns(
        monthly_catboost_train,
        monthly_catboost_split_metadata,
        date_col,
        target_col,
        sku_col,
    )

    # ── validate inputs ───────────────────────────────────────────────────────
    _validate_catboost_inputs(
        monthly_catboost_train,
        monthly_catboost_validation,
        feature_columns,
        date_col,
        target_col,
    )

    # ── prepare arrays ────────────────────────────────────────────────────────
    train_df = monthly_catboost_train.copy()
    val_df = monthly_catboost_validation.copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    val_df[date_col] = pd.to_datetime(val_df[date_col])
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    val_df = val_df.sort_values(date_col).reset_index(drop=True)

    X_train = train_df[feature_columns].values.astype(float)
    y_train = train_df[target_col].values.astype(float)
    X_val = val_df[feature_columns].values.astype(float)
    y_val = val_df[target_col].values.astype(float)

    logger.info(
        "CatBoost monthly training — train=%d rows  val=%d rows  features=%d  "
        "objective=%s  direction=%s  max_trials=%d",
        len(y_train),
        len(y_val),
        len(feature_columns),
        primary_metric,
        selection_direction,
        max_trials,
    )

    # ── Optuna study setup ────────────────────────────────────────────────────
    run_ts = datetime.now(tz=UTC).isoformat()
    study = create_optuna_study(selection_direction, sampler_config)
    trial_results: list[dict] = []
    fitted_models: dict[str, CatBoostRegressor] = {}
    n_failed_count: list[int] = [0]

    def _stop_on_max_failures(
        _study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            n_failed_count[0] += 1
            if max_failed_trials is not None and n_failed_count[0] >= max_failed_trials:
                logger.warning(
                    "CatBoost Optuna search: max_failed_trials=%d reached after %d "
                    "total trial(s). Stopping search early.",
                    max_failed_trials,
                    trial.number + 1,
                )
                _study.stop()

    # ── Optuna objective closure ──────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        candidate_id = f"catboost_trial_{trial.number + 1:03d}"
        t0 = time.perf_counter()

        config = suggest_trial_params(trial, search_space, fixed_params)
        config["random_seed"] = random_seed

        logger.info("[%d] %s — config=%s", trial.number + 1, candidate_id, config)

        try:
            model = _fit_candidate(
                config,
                X_train,
                y_train,
                X_val,
                y_val,
                early_stopping_rounds=early_stopping_rounds,
            )
            y_pred = model.predict(X_val)
            elapsed = time.perf_counter() - t0

            metrics = _compute_validation_metrics(
                y_val, y_pred, y_train, mase_seasonal_period, epsilon
            )
            objective_value = metrics[primary_metric]

            logger.info(
                "%s ✓ (%.2fs) wape=%.4f  mase=%s  rmse=%.2f  bias=%.4f",
                candidate_id,
                elapsed,
                metrics["wape"],
                f"{metrics['mase']:.4f}" if metrics["mase"] is not None else "n/a",
                metrics["rmse"],
                metrics["bias"],
            )

            fitted_models[candidate_id] = model
            trial_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "model_family": "catboost",
                    "granularity": "monthly",
                    "objective_metric": primary_metric,
                    "objective_value": objective_value,
                    **metrics,
                    **{f"param_{k}": v for k, v in config.items() if k != "random_seed"},
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "train_start": str(train_df[date_col].min().date()),
                    "train_end": str(train_df[date_col].max().date()),
                    "validation_start": str(val_df[date_col].min().date()),
                    "validation_end": str(val_df[date_col].max().date()),
                    "fit_seconds": round(elapsed, 3),
                    "error_message": None,
                }
            )
            return float(objective_value)

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.warning(
                "%s FAILED (%.2fs): %s | config=%s", candidate_id, elapsed, exc, config
            )
            trial_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "model_family": "catboost",
                    "granularity": "monthly",
                    "objective_metric": primary_metric,
                    "objective_value": None,
                    "wape": None,
                    "mase": None,
                    "rmse": None,
                    "bias": None,
                    "mae": None,
                    **{f"param_{k}": v for k, v in config.items() if k != "random_seed"},
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "train_start": str(train_df[date_col].min().date()),
                    "train_end": str(train_df[date_col].max().date()),
                    "validation_start": str(val_df[date_col].min().date()),
                    "validation_end": str(val_df[date_col].max().date()),
                    "fit_seconds": round(elapsed, 3),
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            raise  # Let Optuna record this as TrialState.FAIL

    # ── run Optuna study ──────────────────────────────────────────────────────
    study.optimize(
        objective,
        n_trials=max_trials,
        catch=(Exception,),
        callbacks=[_stop_on_max_failures],
    )

    # ── guard: at least one trial must succeed ────────────────────────────────
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]
    if not successful:
        summaries = [f"{r['candidate_id']}: {r['error_message']}" for r in failed]
        raise RuntimeError(
            "All CatBoost monthly trials failed during training.\n" + "\n".join(summaries)
        )

    # ── rank and select pre-champions ─────────────────────────────────────────
    tuning_df = _rank_candidates(
        pd.DataFrame(trial_results), primary_metric, selection_direction
    )

    ranked_mask = tuning_df["rank"].notna()
    actual_prechampion_count = min(prechampion_count, int(ranked_mask.sum()))
    prechampion_ids: list[str] = (
        tuning_df[ranked_mask]
        .assign(_rank_float=tuning_df.loc[ranked_mask, "rank"].astype(float))
        .nsmallest(actual_prechampion_count, "_rank_float")["candidate_id"]
        .tolist()
    )
    logger.info(
        "Pre-champion CatBoost candidates (top-%d): %s", actual_prechampion_count, prechampion_ids
    )

    # ── build validation metrics table (successful candidates only) ───────────
    metric_cols = [
        "candidate_id", "rank", "model_family", "granularity",
        "wape", "mase", "rmse", "bias", "mae",
        "n_train", "n_validation", "validation_start", "validation_end",
    ]
    param_cols = [c for c in tuning_df.columns if c.startswith("param_")]
    val_metrics_df = (
        tuning_df[tuning_df["status"] == "success"][metric_cols + param_cols]
        .reset_index(drop=True)
    )

    # ── build prechampion_configs artifact ────────────────────────────────────
    prechampion_configs = _build_prechampion_configs(
        tuning_df, prechampion_ids, primary_metric, prechampion_count, feature_columns
    )

    # ── persist top-N fitted models ───────────────────────────────────────────
    candidate_models: dict[str, Any] = {}
    for cid in prechampion_ids:
        if cid not in fitted_models:
            continue
        row = tuning_df[tuning_df["candidate_id"] == cid].iloc[0]
        candidate_models[cid] = {
            "rank": _safe_int(row["rank"]),
            "model_family": "catboost",
            "granularity": "monthly",
            "feature_columns": feature_columns,
            "config": {
                k.removeprefix("param_"): v
                for k, v in row.items()
                if str(k).startswith("param_")
            },
            "model": fitted_models[cid],
            "validation_metrics": {
                "wape": _safe_float(row["wape"]),
                "mase": _safe_float(row["mase"]),
                "rmse": _safe_float(row["rmse"]),
                "bias": _safe_float(row["bias"]),
                "mae": _safe_float(row["mae"]),
            },
        }

    # ── build training metadata ───────────────────────────────────────────────
    training_metadata = _build_training_metadata(
        trial_results=trial_results,
        tuning_df=tuning_df,
        prechampion_ids=prechampion_ids,
        primary_metric=primary_metric,
        selection_direction=selection_direction,
        prechampion_count=prechampion_count,
        mase_seasonal_period=mase_seasonal_period,
        feature_columns=feature_columns,
        date_col=date_col,
        target_col=target_col,
        train_df=train_df,
        val_df=val_df,
        run_ts=run_ts,
        search_space=search_space,
        fixed_params=fixed_params,
        max_trials=max_trials,
        random_seed=random_seed,
    )

    logger.info(
        "CatBoost training done — trials=%d  successful=%d  failed=%d  "
        "best=%s  wape=%.4f",
        len(trial_results),
        len(successful),
        len(failed),
        prechampion_ids[0] if prechampion_ids else "none",
        _safe_float(tuning_df[tuning_df["rank"] == 1]["wape"].iloc[0]) or float("nan"),
    )

    return (
        tuning_df,
        val_metrics_df,
        prechampion_configs,
        candidate_models,
        training_metadata,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _resolve_feature_columns(
    train_df: pd.DataFrame,
    split_metadata: dict,
    date_col: str,
    target_col: str,
    sku_col: str,
) -> list[str]:
    """Return the ordered list of feature columns to use for training.

    Uses ``split_metadata["all_feature_columns"]`` when present; otherwise infers
    feature columns by excluding the identity columns from the training DataFrame.

    Args:
        train_df: CatBoost-ready training DataFrame.
        split_metadata: Metadata dict from the CatBoost adapter.
        date_col: Date column name.
        target_col: Target column name.
        sku_col: SKU column name.

    Returns:
        Ordered list of feature column names (excludes date, target, and SKU).
    """
    if "all_feature_columns" in split_metadata:
        from_metadata = list(split_metadata["all_feature_columns"])
        available = [c for c in from_metadata if c in train_df.columns]
        missing_in_df = sorted(set(from_metadata) - set(available))
        if missing_in_df:
            logger.warning(
                "Feature columns listed in split_metadata but absent from "
                "training DataFrame: %s. Proceeding with available columns.",
                missing_in_df,
            )
        return available

    exclude = {date_col, target_col, sku_col} | _IDENTITY_COLUMNS
    return [c for c in train_df.columns if c not in exclude]


def _validate_catboost_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    date_col: str,
    target_col: str,
) -> None:
    """Raise a descriptive error for any invalid training or validation input.

    Args:
        train_df: Training DataFrame to validate.
        val_df: Validation DataFrame to validate.
        feature_columns: Feature columns that must be present in both DataFrames.
        date_col: Name of the date column.
        target_col: Name of the target column.
    """
    if train_df.empty:
        raise ValueError("CatBoost training DataFrame is empty.")
    if val_df.empty:
        raise ValueError("CatBoost validation DataFrame is empty.")
    if not feature_columns:
        raise ValueError(
            "No feature columns resolved. Check that the training DataFrame contains "
            "columns beyond the identity columns (date, target, SKU)."
        )

    for col in (date_col, target_col):
        if col not in train_df.columns:
            raise ValueError(f"Required column {col!r} missing from training data.")
        if col not in val_df.columns:
            raise ValueError(f"Required column {col!r} missing from validation data.")

    missing_train = sorted(set(feature_columns) - set(train_df.columns))
    if missing_train:
        raise ValueError(f"Feature columns missing from training data: {missing_train}")
    missing_val = sorted(set(feature_columns) - set(val_df.columns))
    if missing_val:
        raise ValueError(f"Feature columns missing from validation data: {missing_val}")

    if not pd.api.types.is_numeric_dtype(train_df[target_col]):
        raise ValueError(
            f"Target column {target_col!r} must be numeric; "
            f"found dtype {train_df[target_col].dtype}."
        )


def _fit_candidate(  # noqa: PLR0913
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int,
) -> CatBoostRegressor:
    """Instantiate and fit a CatBoostRegressor for the given trial configuration.

    Args:
        config: Hyperparameter dict (keys must be valid CatBoostRegressor kwargs).
            ``verbose`` and ``allow_writing_files`` are enforced internally and
            must not be present in ``config``.
        X_train: Training feature matrix.
        y_train: Training target array.
        X_val: Validation feature matrix (used for early stopping).
        y_val: Validation target array (used for early stopping).
        early_stopping_rounds: Number of rounds without improvement before stopping.

    Returns:
        Fitted CatBoostRegressor.
    """
    model = CatBoostRegressor(
        **config,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model


def _compute_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    mase_seasonal_period: int,
    epsilon: float,
) -> dict:
    """Compute WAPE, MASE, RMSE, bias, and MAE for a single CatBoost trial.

    Args:
        y_true: Observed validation values.
        y_pred: Forecasted validation values, same shape as y_true.
        y_train: Training series used for the MASE naive denominator.
        mase_seasonal_period: Seasonal period for the MASE naive benchmark.
        epsilon: Small constant for bias denominator stability.

    Returns:
        Dict with wape, mase, rmse, bias, and mae.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    wape_val = _shared_wape(y_true, y_pred)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum(np.abs(y_true)))
    bias_val = float(np.sum(y_pred - y_true)) / (denom + epsilon)

    mase_val: float | None = None
    if y_train is not None and len(y_train) > mase_seasonal_period:
        raw = _shared_mase(y_true, y_pred, y_train, mase_seasonal_period)
        mase_val = raw if (raw is not None and np.isfinite(raw)) else None

    return {
        "wape": wape_val,
        "mase": mase_val,
        "rmse": rmse_val,
        "bias": bias_val,
        "mae": mae_val,
    }


def _rank_candidates(
    df: pd.DataFrame, objective_metric: str, objective_direction: str
) -> pd.DataFrame:
    """Rank successful trials by objective metric; append failed rows at the end.

    Args:
        df: DataFrame with one row per trial, including a ``status`` column
            and an ``objective_value`` column for successful trials.
        objective_metric: Metric column used to sort successful trials.
        objective_direction: ``"minimize"`` or ``"maximize"``.

    Returns:
        DataFrame with a ``rank`` column (int for successful rows, None for failed).
    """
    success_mask = df["status"] == "success"
    success_df = df[success_mask].copy()
    failed_df = df[~success_mask].copy()

    ascending = objective_direction == "minimize"
    success_df = success_df.sort_values(
        "objective_value", ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    success_df["rank"] = list(range(1, len(success_df) + 1))

    failed_df["rank"] = None

    return pd.concat([success_df, failed_df], ignore_index=True)


def _build_prechampion_configs(
    ranked_df: pd.DataFrame,
    prechampion_ids: list[str],
    objective_metric: str,
    prechampion_count: int,
    feature_columns: list[str],
) -> dict:
    """Build the prechampion_configs JSON artifact consumed by the model-selection stage.

    Args:
        ranked_df: Ranked tuning results DataFrame.
        prechampion_ids: Ordered list of candidate IDs (rank-1 first).
        objective_metric: Metric used for ranking.
        prechampion_count: Configured maximum number of pre-champions.
        feature_columns: Feature columns used for training.

    Returns:
        Nested dict ready for JSON serialisation.
    """
    candidates = []
    for rank_pos, cid in enumerate(prechampion_ids, start=1):
        row = ranked_df[ranked_df["candidate_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        param_cols = {
            k.removeprefix("param_"): _to_native(v)
            for k, v in r.items()
            if str(k).startswith("param_")
        }
        candidates.append(
            {
                "candidate_id": cid,
                "rank": rank_pos,
                "model_params": param_cols,
                "feature_columns": feature_columns,
                "metrics": {
                    "wape": _safe_float(r.get("wape")),
                    "mase": _safe_float(r.get("mase")),
                    "rmse": _safe_float(r.get("rmse")),
                    "bias": _safe_float(r.get("bias")),
                    "mae": _safe_float(r.get("mae")),
                },
            }
        )

    return {
        "model_family": "catboost",
        "granularity": "monthly",
        "selection_stage": "validation",
        "objective_metric": objective_metric,
        "top_n": prechampion_count,
        "candidates": candidates,
    }


def _build_training_metadata(  # noqa: PLR0913
    trial_results: list[dict],
    tuning_df: pd.DataFrame,
    prechampion_ids: list[str],
    primary_metric: str,
    selection_direction: str,
    prechampion_count: int,
    mase_seasonal_period: int,
    feature_columns: list[str],
    date_col: str,
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_ts: str,
    search_space: dict,
    fixed_params: dict,
    max_trials: int,
    random_seed: int,
) -> dict:
    """Build the training_metadata JSON artifact.

    Args:
        trial_results: Raw list of per-trial result dicts.
        tuning_df: Ranked tuning results DataFrame.
        prechampion_ids: Ordered list of selected pre-champion candidate IDs.
        primary_metric: Primary selection metric name.
        selection_direction: ``"minimize"`` or ``"maximize"``.
        prechampion_count: Configured maximum number of pre-champions.
        mase_seasonal_period: MASE seasonal period used.
        feature_columns: Feature columns used for training.
        date_col: Date column name.
        target_col: Target column name.
        train_df: Training DataFrame (used for date range metadata).
        val_df: Validation DataFrame (used for date range metadata).
        run_ts: ISO-format run timestamp.
        search_space: Validated Optuna search space definition.
        fixed_params: Fixed hyperparameters not included in the search.
        max_trials: Configured number of Optuna trials.
        random_seed: Random seed used for reproducibility.

    Returns:
        Dict ready for JSON serialisation.
    """
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]

    best_row = (
        tuning_df[tuning_df["rank"] == 1].iloc[0]
        if (tuning_df["rank"] == 1).any()
        else None
    )
    best_candidate_id = str(best_row["candidate_id"]) if best_row is not None else None
    best_metric_value = (
        _safe_float(best_row[primary_metric]) if best_row is not None else None
    )

    warnings_list: list[str] = []
    if len(successful) < prechampion_count:
        warnings_list.append(
            f"Only {len(successful)} successful trial(s); "
            f"requested prechampion_count={prechampion_count}."
        )

    return {
        "model_family": "catboost",
        "model_class": "catboost.CatBoostRegressor",
        "granularity": "monthly",
        "training_stage": "validation",
        "run_timestamp": run_ts,
        "optimizer": "optuna_tpe",
        "primary_metric": primary_metric,
        "selection_direction": selection_direction,
        "prechampion_count": prechampion_count,
        "n_trials_configured": max_trials,
        "n_trials_run": len(trial_results),
        "n_candidates_successful": len(successful),
        "n_candidates_failed": len(failed),
        "best_candidate_id": best_candidate_id,
        "best_validation_metric": best_metric_value,
        "mase_seasonal_period": mase_seasonal_period,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "target_column": target_col,
        "date_column": date_col,
        "train_start": str(train_df[date_col].min().date()),
        "train_end": str(train_df[date_col].max().date()),
        "validation_start": str(val_df[date_col].min().date()),
        "validation_end": str(val_df[date_col].max().date()),
        "search_space": {k: dict(v) for k, v in search_space.items()},
        "fixed_params": dict(fixed_params),
        "random_seed": random_seed,
        "selected_prechampion_ids": list(prechampion_ids),
        "warnings": warnings_list,
        "failed_candidates": [
            {"candidate_id": r["candidate_id"], "error": r["error_message"]}
            for r in failed
        ],
    }


def _to_native(value: Any) -> Any:
    """Convert numpy scalar types to native Python types for JSON serialisation.

    Hyperparameter values read back from a ranked pandas row are often numpy
    scalars (e.g. ``int64``, ``float64``) because pandas coerces numeric columns.
    ``JSONDataset`` cannot serialise numpy scalars, so they are converted to
    native Python types here. Non-numpy values are returned unchanged.

    Args:
        value: Any value, possibly a numpy scalar.

    Returns:
        A native Python scalar when ``value`` is a numpy scalar, else ``value``.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


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
