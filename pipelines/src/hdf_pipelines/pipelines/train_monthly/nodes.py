"""Shared utilities for the monthly training pipeline.

Rolling-origin glue reused by the Prophet, SARIMAX, and CatBoost family tuners so
that cycle generation, the persisted metric set, and Optuna pruning are defined
once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from shared.metrics import mape
from shared.rolling_origin import RollingOriginCycle, generate_rolling_origin_cycles

logger = logging.getLogger(__name__)

# Metric keys persisted per candidate from the rolling-origin summary.
ROLLING_ORIGIN_BASE_METRICS: tuple[str, ...] = ("wmape", "mase", "bias", "rmse")


def compute_validation_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE between actual and predicted values on a validation fold.

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of model predictions, same shape as y_true.

    Returns:
        MAPE as a float. Lower is better.
    """
    return mape(y_true, y_pred)


def build_monthly_rolling_origin_cycles(
    full_df: pd.DataFrame,
    date_col: str,
    rolling_origin_cfg: dict[str, Any],
) -> list[RollingOriginCycle]:
    """Build rolling-origin cycles from a family's full-history frame.

    Uses the shared engine with the ``tuning.rolling_origin`` configuration so the
    cycles match the audit artifact emitted by ``model_input_preparation`` exactly
    (same pure function, same inputs).

    Args:
        full_df: Full-history modeling frame (full history).
        date_col: Month-start date column name.
        rolling_origin_cfg: ``tuning.rolling_origin`` block (horizon, n_cycles,
            window, step_months, min_train_periods).

    Returns:
        Ordered list of ``RollingOriginCycle``.
    """
    dates = (
        pd.to_datetime(full_df[date_col]).drop_duplicates().sort_values().tolist()
    )
    raw_min_train = rolling_origin_cfg.get("min_train_periods")
    return generate_rolling_origin_cycles(
        dates=dates,
        n_cycles=int(rolling_origin_cfg.get("n_cycles", 5)),
        horizon=int(rolling_origin_cfg.get("horizon", 3)),
        step_months=int(rolling_origin_cfg.get("step_months", 1)),
        window=str(rolling_origin_cfg.get("window", "expanding")),
        min_train_periods=int(raw_min_train) if raw_min_train is not None else None,
    )


def supported_rolling_origin_metrics(horizon: int) -> set[str]:
    """Return the set of metric names a rolling-origin tuner may optimise."""
    return set(ROLLING_ORIGIN_BASE_METRICS) | {
        f"wmape_m{h}" for h in range(1, horizon + 1)
    }


def extract_rolling_origin_metric_set(
    aggregated: dict[str, float],
    horizon: int,
) -> dict[str, Any]:
    """Flatten the rolling-origin metrics into the persisted candidate metric set.

    Args:
        aggregated: Output of ``run_rolling_origin`` (means + ``{key}_std``).
        horizon: Forecast horizon (drives the per-horizon keys).

    Returns:
        Dict with ``wmape``, ``mase``, ``bias``, ``rmse``, ``wmape_m{h}`` and their
        ``{key}_std`` dispersion, plus the cycle counts.
    """
    keys = list(ROLLING_ORIGIN_BASE_METRICS) + [
        f"wmape_m{h}" for h in range(1, horizon + 1)
    ]
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = _safe_float(aggregated.get(key))
        out[f"{key}_std"] = _safe_float(aggregated.get(f"{key}_std"))
    out["n_cycles"] = _safe_int(aggregated.get("n_cycles"))
    out["n_cycles_evaluated"] = _safe_int(aggregated.get("n_cycles_evaluated"))
    return out


def log_trial_predictions(
    candidate_id: str,
    trial_preds: list[dict],
) -> None:
    """Log trial predictions deduplicated by target month at INFO level.

    Rolling-origin cycles overlap by design, so the raw cycle-step prediction
    count is ``n_cycles * horizon``. For human-facing training/tuning logs we
    show one prediction per unique target month, keeping the value from the
    latest cycle that predicted that month.

    Args:
        candidate_id: Trial identifier string (e.g. catboost_trial_003).
        trial_preds: List of dicts with keys: target_dates (list[str]) and
            y_pred (list[float]). ``target_start``/``target_end`` are accepted
            for legacy callers and range display.
    """
    if not trial_preds:
        return

    flat_preds: list[float] = []
    unique_by_month: dict[str, float] = {}
    for p in trial_preds:
        y_pred = [float(v) for v in p.get("y_pred", [])]
        flat_preds.extend(y_pred)

        target_dates = p.get("target_dates") or []
        if len(target_dates) != len(y_pred):
            continue

        for raw_date, value in zip(target_dates, y_pred, strict=True):
            month = pd.to_datetime(raw_date).strftime("%Y-%m")
            unique_by_month[month] = value

    date_from = trial_preds[0].get("target_start", "?")
    date_to = trial_preds[-1].get("target_end", "?")

    if unique_by_month:
        preds_str = ", ".join(
            f"{month}={value:.1f}" for month, value in sorted(unique_by_month.items())
        )
        logger.info(
            "  %s · %d unique-month preds (%d cycle-step preds) [%s → %s]: [%s]",
            candidate_id,
            len(unique_by_month),
            len(flat_preds),
            date_from,
            date_to,
            preds_str,
        )
        return

    preds_str = ", ".join(f"{v:.1f}" for v in flat_preds)
    logger.info(
        "  %s · %d preds [%s → %s]: [%s]",
        candidate_id, len(flat_preds), date_from, date_to, preds_str,
    )


def make_pruning_callback(
    trial: optuna.Trial,
    pruning_cfg: dict[str, Any],
    metric_key: str = "wmape_m3",
):
    """Build a per-cycle ``on_cycle_end`` callback for Optuna pruning, or ``None``.

    The callback reports the running cross-cycle objective after each cycle and
    raises ``optuna.TrialPruned`` when the pruner flags the trial. Disabled when
    ``pruning.enabled`` is false.

    Args:
        trial: The active Optuna trial.
        pruning_cfg: ``tuning.pruning`` block (``enabled``, ``n_warmup_cycles``).
        metric_key: Aggregated metric used as the running objective.

    Returns:
        A callable ``(cycle_index, running_metrics) -> None`` or ``None``.
    """
    if not bool(pruning_cfg.get("enabled", False)):
        return None
    n_warmup = int(pruning_cfg.get("n_warmup_cycles", 1))

    def _on_cycle_end(cycle_index: int, running: dict[str, float]) -> None:
        value = running.get(metric_key)
        if value is None or not np.isfinite(value):
            return
        trial.report(float(value), step=cycle_index)
        if cycle_index >= n_warmup and trial.should_prune():
            raise optuna.TrialPruned()

    return _on_cycle_end


def build_rolling_origin_predictions_df(
    all_candidate_preds: dict[str, list[dict]],
    full_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    """Build the long-form rolling-origin predictions artifact for all candidates.

    Each row represents one horizon-step prediction from one rolling-origin cycle of
    one candidate. Storing these allows direct inspection of per-period errors (e.g.
    to identify a single month where a model systematically diverges).

    Args:
        all_candidate_preds: Mapping of ``candidate_id`` → list of per-cycle dicts.
            Each cycle dict must have ``cycle_index`` (int), ``origin_date`` (str),
            ``target_dates`` (list[str] in ``YYYY-MM-DD`` format), and ``y_pred``
            (list[float]).
        full_df: Full-history modeling frame; used as the ground-truth lookup.
        date_col: Month-start date column in ``full_df``.
        target_col: Demand target column in ``full_df``.
        epsilon: Guard added to the APE denominator to avoid division by zero.

    Returns:
        DataFrame with columns: ``candidate_id``, ``cycle_index``, ``origin_date``,
        ``horizon_step``, ``target_date``, ``y_true``, ``y_pred``, ``error``,
        ``abs_error``, ``ape``.  Empty DataFrame (same columns) when no predictions
        were captured.
    """
    _COLUMNS = [
        "candidate_id", "cycle_index", "origin_date", "horizon_step",
        "target_date", "y_true", "y_pred", "error", "abs_error", "ape",
    ]
    if not all_candidate_preds:
        return pd.DataFrame(columns=_COLUMNS)

    ts_map: dict[pd.Timestamp, float] = (
        full_df.assign(**{date_col: pd.to_datetime(full_df[date_col])})
        .set_index(date_col)[target_col]
        .to_dict()
    )

    rows: list[dict] = []
    for candidate_id, cycle_preds in all_candidate_preds.items():
        for pred_dict in cycle_preds:
            cycle_index = pred_dict.get("cycle_index")
            origin_date = pred_dict.get("origin_date")
            target_dates: list[str] = pred_dict.get("target_dates", [])
            y_preds: list[float] = pred_dict.get("y_pred", [])
            for h, (td, yp) in enumerate(zip(target_dates, y_preds), start=1):
                ts = pd.to_datetime(td)
                yt = ts_map.get(ts, float("nan"))
                err = float(yp) - float(yt) if not np.isnan(yt) else float("nan")
                abs_err = abs(err) if not np.isnan(err) else float("nan")
                ape = abs_err / (abs(float(yt)) + epsilon) if not np.isnan(yt) else float("nan")
                rows.append({
                    "candidate_id": candidate_id,
                    "cycle_index": cycle_index,
                    "origin_date": origin_date,
                    "horizon_step": h,
                    "target_date": ts.strftime("%Y-%m-%d"),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "error": err,
                    "abs_error": abs_err,
                    "ape": ape,
                })

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    """Convert to a finite Python float, else ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert to a Python int, else ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
