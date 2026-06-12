"""Rolling-origin backtesting engine for the monthly forecasting layer.

This module is the single source of truth for the rolling-origin protocol
(see ``rolling_origin_evaluation_protocol_en.md``). It is reused by:

- ``model_input_preparation`` — to emit the auditable window specification
  (``monthly_rolling_origin_windows``), and
- ``train_monthly`` — to drive each Optuna trial's per-cycle backtest so that
  tuning and champion selection happen in a single step on the same metrics.

Design invariants (protocol §3, §5, §6):

- **Expanding window, step = 1.** Each cycle trains on all history up to and
  including its origin and forecasts the next ``horizon`` months.
- **Last cycle predicts ``[L-2, L-1, L]``** — the origin of the last cycle is
  ``L - horizon`` so that all observed history through ``L`` participates in
  tuning and selection; no months are reserved out of sample.
- **Pooled aggregation where applicable.** WMAPE, per-horizon WMAPE, and BIAS
  are pooled from cycle-level numerator/denominator totals. MASE and RMSE remain
  cycle-level summary averages.
- **MASE** uses a seasonal-naive (``s = season``) denominator computed on
  *each cycle's own train* window.

All functions are pure and operate on plain ``pandas``/``numpy`` objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.metrics import mase as _mase
from shared.metrics import rmse as _rmse
from shared.metrics import wape as _wape

logger = logging.getLogger(__name__)

# Default epsilon guard for per-horizon WMAPE and BIAS denominators (protocol §5).
DEFAULT_EPSILON: float = 1.0
_POOL_PREFIX: str = "_pool_"


@dataclass(frozen=True)
class RollingOriginCycle:
    """A single rolling-origin cycle.

    Attributes:
        cycle_index: 1-based position of the cycle (cycle 1 is the earliest origin).
        origin_date: Last observed month used for training in this cycle.
        train_end_date: Alias of ``origin_date`` (train includes months ``<= origin``).
        target_dates: The ``horizon`` consecutive month-start dates predicted by
            this cycle, in chronological order (M-1 … M-H).
    """

    cycle_index: int
    origin_date: pd.Timestamp
    train_end_date: pd.Timestamp
    target_dates: list[pd.Timestamp]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the cycle."""
        return {
            "cycle_index": self.cycle_index,
            "origin_date": self.origin_date.strftime("%Y-%m-%d"),
            "train_end_date": self.train_end_date.strftime("%Y-%m-%d"),
            "target_dates": [d.strftime("%Y-%m-%d") for d in self.target_dates],
        }


def generate_rolling_origin_cycles(
    dates: Sequence[pd.Timestamp] | pd.Series,
    n_cycles: int,
    horizon: int,
    step_months: int = 1,
    window: str = "expanding",
    min_train_periods: int | None = None,
) -> list[RollingOriginCycle]:
    """Generate the rolling-origin cycles for a monthly series through ``L``.

    With an expanding window and ``step = 1`` the last cycle's origin is
    ``L - horizon`` so it predicts exactly the most recent ``horizon`` months
    ``[L-H+1, …, L]`` (protocol §2, §4). Target months are taken positionally as
    the ``horizon`` dates immediately following each origin, which assumes a
    contiguous monthly series (enforced upstream by the data contract).

    Args:
        dates: Sorted, unique month-start dates of the observed series (through ``L``).
        n_cycles: Number of rolling-origin cycles (tuning **and** selection).
        horizon: Forecast horizon per cycle, in months (``H``; e.g. 3).
        step_months: Origin advance between consecutive cycles (protocol fixes 1).
        window: Training window type. Only ``"expanding"`` is supported.
        min_train_periods: Minimum number of training months required for the
            earliest cycle. When set and the first cycle's train is shorter, a
            ``ValueError`` is raised with a clear message (protocol §9.3, §10).

    Returns:
        List of ``RollingOriginCycle`` ordered from earliest to latest origin.

    Raises:
        ValueError: If inputs are invalid, the series is too short to form
            ``n_cycles`` cycles, or the first cycle's train is below
            ``min_train_periods``.
    """
    if window != "expanding":
        raise ValueError(
            f"Only the 'expanding' window is supported, got window={window!r}."
        )
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if step_months < 1:
        raise ValueError(f"step_months must be >= 1, got {step_months}.")

    ordered = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    if ordered.has_duplicates:
        raise ValueError("Duplicate month-start dates are not allowed in the series.")
    n_periods = len(ordered)

    # Origin index of the last cycle: predicts the final `horizon` months.
    last_origin_idx = n_periods - 1 - horizon
    # Origin index of the earliest cycle (expanding, fixed step).
    first_origin_idx = last_origin_idx - step_months * (n_cycles - 1)

    if first_origin_idx < 0:
        raise ValueError(
            "Series is too short for the requested rolling-origin configuration: "
            f"{n_periods} months cannot form {n_cycles} cycles with horizon={horizon} "
            f"and step_months={step_months}. The earliest origin would require at "
            f"least {horizon + step_months * (n_cycles - 1) + 1} months."
        )

    first_train_len = first_origin_idx + 1
    if min_train_periods is not None and first_train_len < min_train_periods:
        raise ValueError(
            "First rolling-origin cycle has insufficient training history: "
            f"{first_train_len} months < required minimum {min_train_periods}. "
            "Reduce n_cycles/horizon, extend the series, or lower min_train_periods."
        )

    cycles: list[RollingOriginCycle] = []
    for k in range(n_cycles):
        origin_idx = first_origin_idx + step_months * k
        origin_date = ordered[origin_idx]
        target_dates = [ordered[origin_idx + 1 + j] for j in range(horizon)]
        cycles.append(
            RollingOriginCycle(
                cycle_index=k + 1,
                origin_date=origin_date,
                train_end_date=origin_date,
                target_dates=target_dates,
            )
        )
    return cycles


def compute_cycle_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 12,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """Compute one cycle's metrics over its ``[M-1, …, M-H]`` block (protocol §5).

    Args:
        y_true: Actual demand for the cycle's target months, in horizon order.
        y_pred: Forecast for the same months, same shape as ``y_true``.
        y_train: The cycle's training demand (for the seasonal-naive MASE scale).
        season: Seasonal period for the naive MASE benchmark (12 for monthly).
        epsilon: Denominator guard for per-horizon WMAPE and BIAS.

    Returns:
        Dict with ``wmape`` (global over the block), ``wmape_m{h}`` for each
        horizon ``h`` (1-based), ``mase``, ``bias``, and ``rmse``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    error = y_pred - y_true
    abs_error = np.abs(error)
    abs_true = np.abs(y_true)

    metrics: dict[str, float] = {
        "wmape": _wape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mase": _mase(y_true, y_pred, y_train, season),
        f"{_POOL_PREFIX}wmape_num": float(np.sum(abs_error)),
        f"{_POOL_PREFIX}wmape_den": float(np.sum(abs_true)),
    }

    # Per-horizon WMAPE with epsilon guard (single point per horizon).
    for h in range(1, len(y_true) + 1):
        num = float(abs_error[h - 1])
        den = float(abs_true[h - 1]) + epsilon
        metrics[f"wmape_m{h}"] = num / den if den > 0 else float("nan")
        metrics[f"{_POOL_PREFIX}wmape_m{h}_num"] = num
        metrics[f"{_POOL_PREFIX}wmape_m{h}_den"] = den

    # Normalised directional BIAS over the block (protocol §5).
    denom_bias = float(np.sum(abs_true)) + epsilon
    metrics[f"{_POOL_PREFIX}bias_num"] = float(np.sum(error))
    metrics[f"{_POOL_PREFIX}bias_den"] = denom_bias
    metrics["bias"] = metrics[f"{_POOL_PREFIX}bias_num"] / denom_bias

    return metrics


def aggregate_rolling_origin_metrics(
    per_cycle_records: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Aggregate rolling-origin metrics across successfully evaluated cycles.

    WMAPE, per-horizon WMAPE, and BIAS are pooled from their cycle-level
    numerator/denominator totals. Other public metrics keep arithmetic means.
    For every public metric, ``{key}_std`` remains the dispersion of the per-cycle
    metric values and is diagnostic only.

    Args:
        per_cycle_records: One metric dict per cycle (output of
            ``compute_cycle_metrics``).

    Returns:
        Dict of pooled/averaged metrics plus ``{key}_std`` dispersion and
        ``n_cycles`` / ``n_cycles_evaluated`` counts.
    """
    if not per_cycle_records:
        return {"n_cycles": 0, "n_cycles_evaluated": 0}

    keys: list[str] = []
    for record in per_cycle_records:
        for key in record:
            if key.startswith(_POOL_PREFIX):
                continue
            if key not in keys:
                keys.append(key)

    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.array(
            [rec.get(key, np.nan) for rec in per_cycle_records], dtype=float
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            aggregated[key] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
        else:
            aggregated[key] = float(np.mean(finite))
            aggregated[f"{key}_std"] = float(np.std(finite))

    for key in _pooled_metric_keys(keys):
        num = _sum_component(per_cycle_records, f"{_POOL_PREFIX}{key}_num")
        den = _sum_component(per_cycle_records, f"{_POOL_PREFIX}{key}_den")
        aggregated[key] = float(num / den) if den and den > 0 else float("nan")

    aggregated["n_cycles"] = len(per_cycle_records)
    # A cycle counts as evaluated when its primary block metric is finite.
    aggregated["n_cycles_evaluated"] = int(
        np.sum(
            [np.isfinite(rec.get("wmape", np.nan)) for rec in per_cycle_records]
        )
    )
    return aggregated


def _pooled_metric_keys(public_keys: Sequence[str]) -> list[str]:
    """Return public metric keys that should be pooled across cycles."""
    keys = ["wmape", "bias"]
    keys.extend(
        sorted(
            [k for k in public_keys if k.startswith("wmape_m")],
            key=lambda item: int(item.replace("wmape_m", "")),
        )
    )
    return [k for k in keys if k in public_keys]


def _sum_component(records: Sequence[dict[str, float]], key: str) -> float:
    """Sum a private pooling component, ignoring missing or non-finite values."""
    values = np.array([rec.get(key, np.nan) for rec in records], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.sum(finite)) if finite.size else float("nan")


def _public_metrics(record: dict[str, float]) -> dict[str, float]:
    """Return only artifact-facing metrics from a cycle metric record."""
    return {k: v for k, v in record.items() if not k.startswith(_POOL_PREFIX)}


def run_rolling_origin(
    full_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    cycles: Sequence[RollingOriginCycle],
    fit_forecast_fn: Callable[[pd.DataFrame, RollingOriginCycle], np.ndarray],
    season: int = 12,
    epsilon: float = DEFAULT_EPSILON,
    on_cycle_end: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a rolling-origin backtest and return per-cycle and aggregated metrics.

    For each cycle the driver slices the leakage-safe training window
    (``full_df`` rows with ``date_col <= origin``), delegates fitting and
    ``horizon``-step forecasting to ``fit_forecast_fn``, then scores the forecast
    against the observed target months. A cycle that raises is logged and recorded
    with NaN metrics so the remaining cycles still contribute (protocol §6, §9.1).

    Args:
        full_df: Full-history modeling frame (through ``L``) with ``date_col``,
            ``target_col``, and any future-known exogenous columns. Must be sorted
            or sortable by ``date_col``.
        date_col: Name of the month-start date column.
        target_col: Name of the demand target column.
        cycles: Cycles produced by ``generate_rolling_origin_cycles``.
        fit_forecast_fn: Callable ``(train_df, cycle) -> np.ndarray`` returning the
            ``horizon`` forecasts for ``cycle.target_dates`` in order. It owns all
            family-specific logic (model fit, exogenous/future-regressor handling).
        season: Seasonal period for the per-cycle MASE denominator.
        epsilon: Denominator guard passed to ``compute_cycle_metrics``.
        on_cycle_end: Optional hook called after each *successfully evaluated*
            cycle with ``(cycle_index, running_aggregated_metrics)``. Intended for
            Optuna pruning — the callback may report the running objective and
            raise ``optuna.TrialPruned``, which propagates out of the driver
            (it is not swallowed as a cycle failure).

    Returns:
        Tuple ``(per_cycle_df, aggregated_metrics)``. ``per_cycle_df`` has one row
        per cycle with its origin, targets, status, and public metrics;
        ``aggregated_metrics`` uses pooled WMAPE/BIAS where applicable.
    """
    frame = full_df.sort_values(date_col).reset_index(drop=True)
    frame[date_col] = pd.to_datetime(frame[date_col])

    per_cycle_records: list[dict[str, float]] = []
    rows: list[dict] = []

    for cycle in cycles:
        train_df = frame[frame[date_col] <= cycle.origin_date].copy()
        y_train = train_df[target_col].to_numpy(dtype=float)

        target_set = pd.DatetimeIndex(cycle.target_dates)
        target_df = (
            frame[frame[date_col].isin(target_set)]
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        y_true = target_df[target_col].to_numpy(dtype=float)

        row: dict = {
            "cycle_index": cycle.cycle_index,
            "origin_date": cycle.origin_date.strftime("%Y-%m-%d"),
            "target_start": cycle.target_dates[0].strftime("%Y-%m-%d"),
            "target_end": cycle.target_dates[-1].strftime("%Y-%m-%d"),
            "n_train": int(len(y_train)),
            "status": "success",
        }

        if len(y_true) != len(cycle.target_dates):
            row["status"] = "missing_targets"
            rows.append(row)
            logger.warning(
                "Rolling-origin cycle %d: expected %d target months but found %d "
                "in the series — skipping cycle.",
                cycle.cycle_index, len(cycle.target_dates), len(y_true),
            )
            continue

        try:
            y_pred = np.asarray(fit_forecast_fn(train_df, cycle), dtype=float)
            if y_pred.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"fit_forecast_fn returned {y_pred.shape[0]} forecasts, "
                    f"expected {y_true.shape[0]}."
                )
            cycle_metrics = compute_cycle_metrics(
                y_true, y_pred, y_train, season=season, epsilon=epsilon
            )
        except Exception as exc:  # noqa: BLE001 — one bad cycle must not void the trial
            row["status"] = f"failed: {exc}"
            rows.append(row)
            logger.warning(
                "Rolling-origin cycle %d (origin=%s) failed: %s",
                cycle.cycle_index, row["origin_date"], exc,
            )
            continue

        per_cycle_records.append(cycle_metrics)
        row.update(_public_metrics(cycle_metrics))
        rows.append(row)

        # Pruning hook runs outside the try/except so optuna.TrialPruned propagates.
        if on_cycle_end is not None:
            running = aggregate_rolling_origin_metrics(per_cycle_records)
            on_cycle_end(cycle.cycle_index, running)

    aggregated = aggregate_rolling_origin_metrics(per_cycle_records)
    # Driver semantics: total cycles attempted vs. successfully evaluated.
    aggregated["n_cycles"] = len(cycles)
    aggregated["n_cycles_evaluated"] = len(per_cycle_records)
    per_cycle_df = pd.DataFrame(rows)
    return per_cycle_df, aggregated
