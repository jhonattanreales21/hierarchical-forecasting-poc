"""Shared utilities for the monthly training pipeline.

Rolling-origin glue reused by the Prophet, SARIMAX, and CatBoost family tuners so
that cycle generation, the persisted metric set, and Optuna pruning are defined
once (protocol §4, §5).
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

# Metric keys persisted per candidate from the macro-averaged rolling-origin run.
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
        full_df: Full-history modeling frame (through ``L``).
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
    """Flatten the macro-averaged metrics into the persisted candidate metric set.

    Args:
        aggregated: Output of ``run_rolling_origin`` (means + ``{key}_std``).
        horizon: Forecast horizon ``H`` (drives the per-horizon keys).

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
