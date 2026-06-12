"""Monthly CatBoost training nodes: rolling-origin Optuna search, direct multi-horizon.

Implements an Optuna TPE study where each trial is evaluated by a **rolling-origin
backtest** using the **direct multi-horizon strategy**:
- One independent CatBoostRegressor per forecast horizon h ∈ {1, 2, 3}.
- model_h trains on (features_at_origin_t, demand(t+h)) pairs (shifted-target formulation)
  and predicts demand(origin+h) from the origin row's features at inference time.
- No recursion — predictions are never reused as inputs for later steps.

The Optuna objective is pooled ``WMAPE_M3`` across rolling-origin cycles.
Top-N pre-champions are refit on full history (3 models each) for the downstream
model-selection stage, which selects champions directly from rolling-origin metrics
— no separate held-out test stage.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from shared.rolling_origin import RollingOriginCycle, run_rolling_origin

from hdf_pipelines.pipelines.train_monthly.nodes import (
    build_monthly_rolling_origin_cycles,
    build_rolling_origin_predictions_df,
    extract_rolling_origin_metric_set,
    log_trial_predictions,
    make_pruning_callback,
)
from hdf_pipelines.utils.optuna_helpers import (
    build_rolling_origin_pruner,
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
    monthly_catboost_full_train: pd.DataFrame,
    monthly_catboost_split_metadata: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, pd.DataFrame]:
    """Tune CatBoost with a rolling-origin backtest (direct multi-horizon) and persist pre-champions.

    Each Optuna trial is evaluated by a **rolling-origin backtest** using the
    **direct multi-horizon strategy**: for each horizon h ∈ {1, 2, 3} an
    independent CatBoostRegressor is trained on ``(features_at_origin_t, demand(t+h))``
    pairs (shifted-target formulation) and predicts ``demand(origin+h)`` from the
    origin row's features. No recursion is used. The Optuna objective is pooled
    ``WMAPE_M3``.

    Top-N pre-champions are refit on full history (3 models each) and persisted for
    the model-selection stage, which selects champions directly from rolling-origin
    metrics — no separate held-out test stage.

    Args:
        monthly_catboost_full_train: CatBoost-ready full-history DataFrame
            produced by ``adapt_monthly_data_for_catboost``.
        monthly_catboost_split_metadata: Metadata dict from the CatBoost adapter.
            Provides ``date_column``, ``target_column``, ``sku_column``, and
            ``all_feature_columns``.
        params: Contents of ``train_monthly.catboost`` from the parameter file.

    Returns:
        Five-element tuple:

        1. ``tuning_results`` — DataFrame, one row per trial (ranked).
        2. ``rolling_origin_metrics`` — DataFrame, per-candidate rolling-origin metrics.
        3. ``prechampion_configs`` — Dict with top-N pre-champion configurations and
           their rolling-origin metric sets.
        4. ``candidate_models`` — Dict mapping candidate_id → dict with three fitted
           CatBoostRegressors (``model_h1``, ``model_h2``, ``model_h3``) and shared
           ``feature_columns``.
        5. ``training_metadata`` — Dict summarising the Optuna study run.
        6. ``rolling_origin_predictions`` — Long-form DataFrame with one row per
           (candidate, cycle, horizon-step) prediction for all successful trials.

    Raises:
        ValueError: When inputs are invalid or required columns are missing.
        RuntimeError: When every trial fails to evaluate.
    """
    # ── extract configuration ─────────────────────────────────────────────────
    primary_metric: str = str(params.get("primary_metric", "wmape_m3")).lower()
    selection_direction: str = str(
        params.get("selection_direction", "minimize")
    ).lower()
    random_seed: int = int(params.get("random_seed", 42))
    prechampion_count: int = int(params.get("prechampion_count", 10))
    mase_seasonal_period: int = int(params.get("mase_seasonal_period", 12))
    epsilon: float = float(params.get("epsilon", 1.0))
    max_trials: int = int(params.get("max_trials", 100))
    max_failed_trials: int | None = (
        int(params["max_failed_trials"])
        if params.get("max_failed_trials") is not None
        else None
    )
    sampler_cfg: dict = dict(
        params.get("sampler", {"name": "tpe", "seed": random_seed})
    )
    pruning_cfg: dict = dict(params.get("pruning", {}))
    rolling_origin_cfg: dict = dict(params.get("rolling_origin", {}))
    horizon: int = int(rolling_origin_cfg.get("horizon", 3))
    search_space: dict = validate_optuna_search_space(
        dict(params.get("search_space", {}))
    )
    fixed_params: dict = dict(params.get("fixed_params", {}))

    date_col: str = str(
        monthly_catboost_split_metadata.get("date_column", "month_start_date")
    )
    target_col: str = str(
        monthly_catboost_split_metadata.get("target_column", "monthly_demand")
    )
    sku_col: str = str(monthly_catboost_split_metadata.get("sku_column", "sku"))

    if monthly_catboost_full_train.empty:
        raise ValueError("monthly_catboost_full_train is empty.")
    for col in (date_col, target_col, sku_col):
        if col not in monthly_catboost_full_train.columns:
            raise ValueError(
                f"Required column {col!r} missing from monthly_catboost_full_train."
            )

    # ── resolve feature columns ───────────────────────────────────────────────
    feature_columns = _resolve_feature_columns(
        monthly_catboost_full_train,
        monthly_catboost_split_metadata,
        date_col,
        target_col,
        sku_col,
    )
    if not feature_columns:
        raise ValueError(
            "No feature columns resolved. Check that all_feature_columns is set in "
            "monthly_catboost_split_metadata."
        )

    # ── prepare full-history frame ────────────────────────────────────────────
    full_df = monthly_catboost_full_train.copy()
    full_df[date_col] = pd.to_datetime(full_df[date_col])
    full_df = full_df.sort_values(date_col).reset_index(drop=True)

    # ── build rolling-origin cycles ───────────────────────────────────────────
    cycles = build_monthly_rolling_origin_cycles(full_df, date_col, rolling_origin_cfg)
    logger.info(
        "CatBoost direct multi-horizon rolling-origin training — %d rows | %s → %s | "
        "%d cycles (H=%d) | n_features=%d | objective=%s | max_trials=%d",
        len(full_df),
        full_df[date_col].min().date(),
        full_df[date_col].max().date(),
        len(cycles),
        horizon,
        len(feature_columns),
        primary_metric,
        max_trials,
    )

    # ── Optuna study setup ────────────────────────────────────────────────────
    run_ts = datetime.now(tz=UTC).isoformat()
    study = create_optuna_study(
        selection_direction,
        sampler_cfg,
        pruner=build_rolling_origin_pruner(pruning_cfg),
    )
    trial_results: list[dict] = []
    all_candidate_preds: dict[str, list[dict]] = {}
    n_failed_count: list[int] = [0]

    def _stop_on_max_failures(
        _study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            n_failed_count[0] += 1
            if max_failed_trials is not None and n_failed_count[0] >= max_failed_trials:
                logger.warning(
                    "CatBoost Optuna: max_failed_trials=%d reached after %d trial(s). Stopping.",
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

        _trial_preds: list[dict] = []
        _base_fit_fn = _make_direct_fit_forecast_fn(
            config=config,
            feature_cols=feature_columns,
            target_col=target_col,
            sku_col=sku_col,
            horizons=list(range(1, horizon + 1)),
        )

        def _fit_and_capture(
            train_df: pd.DataFrame, cycle: RollingOriginCycle
        ) -> np.ndarray:
            y_pred = _base_fit_fn(train_df, cycle)
            _trial_preds.append({
                "cycle_index": cycle.cycle_index,
                "origin_date": cycle.origin_date.strftime("%Y-%m-%d"),
                "target_start": cycle.target_dates[0].strftime("%Y-%m"),
                "target_end": cycle.target_dates[-1].strftime("%Y-%m"),
                "target_dates": [d.strftime("%Y-%m-%d") for d in cycle.target_dates],
                "y_pred": y_pred.tolist(),
            })
            return y_pred

        on_cycle_end = make_pruning_callback(
            trial, pruning_cfg, metric_key=primary_metric
        )
        try:
            _, aggregated = run_rolling_origin(
                full_df,
                date_col,
                target_col,
                cycles,
                _fit_and_capture,
                season=mase_seasonal_period,
                epsilon=epsilon,
                on_cycle_end=on_cycle_end,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.warning("%s FAILED (%.2fs): %s", candidate_id, elapsed, exc)
            trial_results.append(
                _failed_trial_row(
                    candidate_id,
                    config,
                    primary_metric,
                    len(full_df),
                    elapsed,
                    str(exc),
                )
            )
            raise

        metric_set = extract_rolling_origin_metric_set(aggregated, horizon)
        objective_value = metric_set.get(primary_metric)
        elapsed = time.perf_counter() - t0

        if objective_value is None:
            msg = f"{candidate_id}: all rolling-origin cycles failed"
            trial_results.append(
                _failed_trial_row(
                    candidate_id, config, primary_metric, len(full_df), elapsed, msg
                )
            )
            raise RuntimeError(msg)

        logger.info(
            "%s \u2713 (%.2fs) wmape=%.4f  wmape_m1=%.4f  wmape_m2=%.4f  wmape_m3=%.4f  mase=%s  bias=%.4f",
            candidate_id,
            elapsed,
            metric_set.get("wmape") or float("nan"),
            metric_set.get("wmape_m1") or float("nan"),
            metric_set.get("wmape_m2") or float("nan"),
            metric_set.get("wmape_m3") or float("nan"),
            (
                f"{metric_set['mase']:.4f}"
                if metric_set.get("mase") is not None
                else "n/a"
            ),
            metric_set.get("bias") or float("nan"),
        )
        log_trial_predictions(candidate_id, _trial_preds)
        all_candidate_preds[candidate_id] = list(_trial_preds)

        trial_results.append(
            {
                "candidate_id": candidate_id,
                "status": "success",
                "model_family": "catboost",
                "granularity": "monthly",
                "strategy": "direct_multi_horizon",
                "objective_metric": primary_metric,
                "objective_value": _safe_float(objective_value),
                **{f"ro_{k}": v for k, v in metric_set.items()},
                **{f"param_{k}": v for k, v in config.items() if k != "random_seed"},
                "n_obs": len(full_df),
                "fit_seconds": round(elapsed, 3),
                "error_message": None,
            }
        )
        return float(objective_value)

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
        summaries = [f"{r['candidate_id']}: {r.get('error_message')}" for r in failed]
        raise RuntimeError(
            "All CatBoost monthly trials failed during rolling-origin evaluation.\n"
            + "\n".join(summaries)
        )

    # ── rank trials ───────────────────────────────────────────────────────────
    tuning_df = _rank_candidates(
        pd.DataFrame(trial_results), f"ro_{primary_metric}", selection_direction
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
        "CatBoost pre-champions (top-%d): %s", actual_prechampion_count, prechampion_ids
    )

    # ── refit top-N on full history (3 models each) ───────────────────────────
    candidate_models: dict[str, Any] = {}
    for cid in prechampion_ids:
        row = tuning_df[tuning_df["candidate_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        cfg = {
            k.removeprefix("param_"): _to_native(v)
            for k, v in r.items()
            if str(k).startswith("param_")
        }
        cfg["random_seed"] = random_seed
        try:
            multi_models = _refit_direct_models_on_df(
                config=cfg,
                df=full_df,
                feature_cols=feature_columns,
                target_col=target_col,
                sku_col=sku_col,
                horizons=list(range(1, horizon + 1)),
            )
            metric_set = extract_rolling_origin_metric_set(
                {
                    k.removeprefix("ro_"): v
                    for k, v in r.items()
                    if str(k).startswith("ro_")
                },
                horizon,
            )
            candidate_models[cid] = {
                "rank": _safe_int(r["rank"]),
                "model_family": "catboost",
                "granularity": "monthly",
                "strategy": "direct_multi_horizon",
                "config": {k: v for k, v in cfg.items() if k != "random_seed"},
                "feature_columns": feature_columns,
                "rolling_origin_metrics": metric_set,
                **multi_models,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Full-history refit for %s failed: %s", cid, exc)

    # ── rolling-origin metrics table ──────────────────────────────────────────
    base_cols = [
        "candidate_id",
        "rank",
        "model_family",
        "granularity",
        "strategy",
        "objective_metric",
        "objective_value",
    ]
    param_cols_list = [c for c in tuning_df.columns if c.startswith("param_")]
    ro_cols_list = [c for c in tuning_df.columns if c.startswith("ro_")]
    rolling_origin_metrics_df = tuning_df[tuning_df["status"] == "success"][
        base_cols + ro_cols_list + param_cols_list
    ].reset_index(drop=True)

    # ── prechampion_configs (consumed by model-selection Node 2) ──────────────
    prechampion_configs = _build_direct_prechampion_configs(
        tuning_df=tuning_df,
        prechampion_ids=prechampion_ids,
        candidate_models=candidate_models,
        primary_metric=primary_metric,
        prechampion_count=prechampion_count,
        feature_columns=feature_columns,
        horizon=horizon,
    )

    # ── training metadata ─────────────────────────────────────────────────────
    training_metadata = _build_catboost_training_metadata(
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
        full_df=full_df,
        run_ts=run_ts,
        search_space=search_space,
        fixed_params=fixed_params,
        max_trials=max_trials,
        random_seed=random_seed,
        n_cycles=len(cycles),
        horizon=horizon,
    )

    best_val = _safe_float(
        tuning_df[tuning_df["rank"] == 1][f"ro_{primary_metric}"].iloc[0]
        if (tuning_df["rank"] == 1).any()
        else None
    )
    rolling_origin_predictions = build_rolling_origin_predictions_df(
        all_candidate_preds, full_df, date_col, target_col, epsilon=epsilon
    )
    logger.info(
        "CatBoost training done — trials=%d  successful=%d  failed=%d  "
        "best=%s  %s=%.4f  prediction_rows=%d",
        len(trial_results),
        len(successful),
        len(failed),
        prechampion_ids[0] if prechampion_ids else "none",
        primary_metric,
        best_val or float("nan"),
        len(rolling_origin_predictions),
    )
    return (
        tuning_df,
        rolling_origin_metrics_df,
        prechampion_configs,
        candidate_models,
        training_metadata,
        rolling_origin_predictions,
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


def _make_direct_fit_forecast_fn(
    config: dict,
    feature_cols: list[str],
    target_col: str,
    sku_col: str,
    horizons: list[int],
):
    """Build a direct multi-horizon fit_forecast_fn for the rolling-origin engine.

    Returns a callable ``(train_df, cycle) -> np.ndarray`` that:
    - For each horizon h trains a CatBoostRegressor on (features_at_t, demand(t+h))
      pairs (shifted-target formulation).
    - At cycle inference, applies each model_h to the last row of train_df
      (the origin row) to predict demand(origin+h).

    Features are evaluated at the origin row so all configured lag/rolling
    features are valid at inference time regardless of horizon.
    """

    def fit_forecast(
        train_df: "pd.DataFrame", cycle: "RollingOriginCycle"
    ) -> "np.ndarray":
        df = train_df.copy().reset_index(drop=True)
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            raise ValueError(
                "No feature columns available in train_df for CatBoost direct multi-horizon."
            )

        predictions: list[float] = []
        for h in horizons:
            # Shifted target: target for row t = demand(t+h)
            y_h = df.groupby(sku_col, sort=False)[target_col].transform(
                lambda s, shift=h: s.shift(-shift)
            )
            valid_mask = y_h.notna()
            if valid_mask.sum() < 3:  # too few pairs to train
                raise ValueError(
                    f"CatBoost direct multi-horizon: horizon h={h} has fewer than 3 training pairs "
                    f"in this cycle (train_df has {len(df)} rows, shift=-{h})."
                )
            X_train = df.loc[valid_mask, available].to_numpy(dtype=float)
            y_train = y_h[valid_mask].to_numpy(dtype=float)

            fit_cfg = {k: v for k, v in config.items() if k != "random_seed"}
            fit_cfg["random_seed"] = config.get("random_seed", 42)
            model_h = CatBoostRegressor(
                **fit_cfg, verbose=False, allow_writing_files=False
            )
            model_h.fit(X_train, y_train, verbose=False)

            # Predict from the origin row (last row of train_df)
            X_origin = df[available].iloc[[-1]].to_numpy(dtype=float)
            predictions.append(float(model_h.predict(X_origin)[0]))

        return np.array(predictions, dtype=float)

    return fit_forecast


def _refit_direct_models_on_df(
    config: dict,
    df: "pd.DataFrame",
    feature_cols: list[str],
    target_col: str,
    sku_col: str,
    horizons: list[int],
) -> dict:
    """Refit direct multi-horizon CatBoost models on an arbitrary DataFrame.

    Returns a dict with keys ``model_h1``, ``model_h2``, ``model_h3``
    (or model_h{k} for each k in horizons).

    Args:
        config: CatBoost hyperparameter dict.
        df: Full DataFrame to train on (must contain target_col and all feature_cols).
        feature_cols: Feature columns to use.
        target_col: Demand target column.
        sku_col: SKU identifier column.
        horizons: Horizon integers to produce models for (e.g. [1, 2, 3]).

    Returns:
        Dict ``{"model_h1": ..., "model_h2": ..., "model_h3": ...}``.
    """
    df = df.copy().reset_index(drop=True)
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(
            "No feature columns available in df for CatBoost full-history refit."
        )

    result: dict = {}
    for h in horizons:
        y_h = df.groupby(sku_col, sort=False)[target_col].transform(
            lambda s, shift=h: s.shift(-shift)
        )
        valid_mask = y_h.notna()
        X_full = df.loc[valid_mask, available].to_numpy(dtype=float)
        y_full = y_h[valid_mask].to_numpy(dtype=float)

        fit_cfg = {k: v for k, v in config.items() if k != "random_seed"}
        fit_cfg["random_seed"] = config.get("random_seed", 42)
        model_h = CatBoostRegressor(**fit_cfg, verbose=False, allow_writing_files=False)
        model_h.fit(X_full, y_full, verbose=False)
        result[f"model_h{h}"] = model_h

    return result


def _failed_trial_row(
    candidate_id: str,
    config: dict,
    objective_metric: str,
    n_obs: int,
    elapsed: float,
    error_msg: str,
) -> dict:
    """Build a failed-trial row for the trial_results list."""
    return {
        "candidate_id": candidate_id,
        "status": "failed",
        "model_family": "catboost",
        "granularity": "monthly",
        "strategy": "direct_multi_horizon",
        "objective_metric": objective_metric,
        "objective_value": None,
        **{f"param_{k}": v for k, v in config.items() if k != "random_seed"},
        "n_obs": n_obs,
        "fit_seconds": round(elapsed, 3),
        "error_message": error_msg,
    }


def _rank_candidates(
    df: pd.DataFrame, objective_metric: str, objective_direction: str
) -> pd.DataFrame:
    """Rank successful trials by objective metric; append failed rows at the end."""
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


def _build_direct_prechampion_configs(
    tuning_df: pd.DataFrame,
    prechampion_ids: list[str],
    candidate_models: dict,
    primary_metric: str,
    prechampion_count: int,
    feature_columns: list[str],
    horizon: int,
) -> dict:
    """Build prechampion_configs for the model-selection stage (rolling-origin metrics).

    Each candidate entry carries:
    - ``candidate_id``, ``rank``
    - ``config``: hyperparameter dict
    - ``feature_columns``
    - ``rolling_origin_metrics``: dict with wmape, wmape_m1/m2/m3, mase, bias, rmse
      (consumed by ``assemble_monthly_candidate_metrics`` Node 2).
    """
    candidates = []
    for rank_pos, cid in enumerate(prechampion_ids, start=1):
        row = tuning_df[tuning_df["candidate_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        config = {
            k.removeprefix("param_"): _to_native(v)
            for k, v in r.items()
            if str(k).startswith("param_")
        }
        # Rolling-origin metrics are stored with "ro_" prefix in tuning_df.
        ro_metrics: dict = {
            k.removeprefix("ro_"): _safe_float(v)
            for k, v in r.items()
            if str(k).startswith("ro_") and not k.endswith("_std")
        }
        candidates.append(
            {
                "candidate_id": cid,
                "rank": rank_pos,
                "config": config,
                "feature_columns": feature_columns,
                "rolling_origin_metrics": ro_metrics,
            }
        )

    return {
        "model_family": "catboost",
        "granularity": "monthly",
        "selection_stage": "rolling_origin",
        "primary_metric": primary_metric,
        "top_n": prechampion_count,
        "horizon": horizon,
        "candidates": candidates,
    }


def _build_catboost_training_metadata(  # noqa: PLR0913
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
    full_df: pd.DataFrame,
    run_ts: str,
    search_space: dict,
    fixed_params: dict,
    max_trials: int,
    random_seed: int,
    n_cycles: int,
    horizon: int,
) -> dict:
    """Build the training_metadata JSON artifact for rolling-origin CatBoost."""
    successful = [r for r in trial_results if r["status"] == "success"]
    failed = [r for r in trial_results if r["status"] == "failed"]

    best_row = (
        tuning_df[tuning_df["rank"] == 1].iloc[0]
        if (tuning_df["rank"] == 1).any()
        else None
    )
    best_candidate_id = str(best_row["candidate_id"]) if best_row is not None else None
    best_metric_col = f"ro_{primary_metric}"
    best_metric_value = (
        _safe_float(best_row[best_metric_col])
        if best_row is not None and best_metric_col in best_row.index
        else None
    )

    return {
        "model_family": "catboost",
        "granularity": "monthly",
        "strategy": "direct_multi_horizon",
        "evaluation_mode": "rolling_origin",
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
        f"best_{primary_metric}": best_metric_value,
        "mase_seasonal_period": mase_seasonal_period,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "target_column": target_col,
        "date_column": date_col,
        "full_train_start": str(full_df[date_col].min().date()),
        "full_train_end": str(full_df[date_col].max().date()),
        "n_obs": len(full_df),
        "n_cycles": n_cycles,
        "horizon": horizon,
        "search_space": {k: dict(v) for k, v in search_space.items()},
        "fixed_params": dict(fixed_params),
        "random_seed": random_seed,
        "selected_prechampion_ids": list(prechampion_ids),
        "note": (
            "Champions selected directly on rolling-origin metrics; "
            "no reserved out-of-sample window was used."
        ),
        "failed_candidates": [
            {"candidate_id": r["candidate_id"], "error": r.get("error_message")}
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
