"""Shared helpers for Optuna-based hyperparameter optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial

_VALID_DIRECTIONS: frozenset[str] = frozenset({"minimize", "maximize"})


def create_optuna_study(
    direction: str,
    sampler_config: dict[str, Any] | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
) -> Study:
    """Create an ephemeral Optuna study with the configured sampler.

    Args:
        direction: ``"minimize"`` or ``"maximize"``.
        sampler_config: TPE sampler configuration (name, seed, n_startup_trials,
            multivariate, gamma).
        pruner: Optional Optuna pruner. When provided (e.g. for rolling-origin
            cycle pruning), trials may be pruned via ``trial.report`` /
            ``trial.should_prune`` inside the objective. Defaults to no pruning.

    Returns:
        A configured Optuna ``Study``.
    """
    sampler_config = sampler_config or {}
    sampler_name = str(sampler_config.get("name", "tpe")).lower()
    if sampler_name != "tpe":
        raise ValueError(
            f"Unsupported Optuna sampler {sampler_name!r}. Only 'tpe' is supported."
        )

    seed = sampler_config.get("seed")
    n_startup_trials = sampler_config.get("n_startup_trials", 10)
    multivariate = bool(sampler_config.get("multivariate", False))
    gamma_value = float(sampler_config.get("gamma", 0.25))
    # gamma expects Callable[[int], int]: fraction of trials to treat as "good"
    gamma_fn = lambda n: min(int(np.ceil(gamma_value * n)), 25)  # noqa: E731

    sampler = TPESampler(
        seed=int(seed) if seed is not None else None,
        n_startup_trials=int(n_startup_trials),
        multivariate=multivariate,
        gamma=gamma_fn,
    )
    return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)


def build_rolling_origin_pruner(
    pruning_config: dict[str, Any] | None,
) -> optuna.pruners.BasePruner | None:
    """Build an Optuna pruner for rolling-origin cycle pruning, or ``None``.

    The objective reports the running cross-cycle objective after each cycle; the
    ``MedianPruner`` then prunes trials that are clearly worse than the running
    median at the same cycle (protocol §9.2). ``n_warmup_cycles`` cycles run before
    pruning is allowed so the first noisy cycles never trigger a prune.

    Args:
        pruning_config: ``tuning.pruning`` block with ``enabled``, ``n_startup_trials``,
            and ``n_warmup_cycles``.

    Returns:
        A ``MedianPruner`` when enabled, else ``None`` (no pruning).
    """
    cfg = pruning_config or {}
    if not bool(cfg.get("enabled", False)):
        return None
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.get("n_startup_trials", 10)),
        n_warmup_steps=int(cfg.get("n_warmup_cycles", 1)),
    )


def validate_objective_metric_direction(
    metric: str,
    direction: str,
    supported_metrics: set[str] | frozenset[str],
) -> tuple[str, str]:
    """Validate that the objective metric exists and matches the direction."""
    metric = str(metric)
    direction = str(direction).lower()

    if metric not in supported_metrics:
        supported_display = ", ".join(sorted(supported_metrics))
        raise ValueError(
            f"Unsupported Optuna objective metric {metric!r}. "
            f"Supported metrics: {supported_display}."
        )

    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Unsupported Optuna objective direction {direction!r}. "
            "Use 'minimize' or 'maximize'."
        )

    expected_direction = (
        "maximize"
        if metric == "forecast_precision" or metric.endswith("_forecast_precision")
        else "minimize"
    )
    if direction != expected_direction:
        raise ValueError(
            f"Objective direction {direction!r} is incompatible with metric "
            f"{metric!r}. Expected {expected_direction!r}."
        )

    return metric, direction


def validate_optuna_search_space(  # noqa: PLR0912
    search_space: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and normalize an Optuna-native search-space definition."""
    if not search_space:
        raise ValueError(
            "Optuna search_space is empty — check "
            "train_monthly.prophet.tuning.search_space in the parameter file."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for param_name, spec in search_space.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Search-space definition for {param_name!r} must be a mapping."
            )

        param_type = str(spec.get("type", "")).lower()
        if param_type == "float":
            low = spec.get("low")
            high = spec.get("high")
            if low is None or high is None:
                raise ValueError(
                    f"Float search-space {param_name!r} must define 'low' and 'high'."
                )
            low = float(low)
            high = float(high)
            if low > high:
                raise ValueError(
                    f"Float search-space {param_name!r} has low > high ({low} > {high})."
                )
            log = bool(spec.get("log", False))
            if log and low <= 0:
                raise ValueError(
                    f"Float search-space {param_name!r} cannot use log scale with low <= 0."
                )
            normalized[param_name] = {
                "type": "float",
                "low": low,
                "high": high,
                "log": log,
            }
            continue

        if param_type == "int":
            low = spec.get("low")
            high = spec.get("high")
            if low is None or high is None:
                raise ValueError(
                    f"Int search-space {param_name!r} must define 'low' and 'high'."
                )
            low = int(low)
            high = int(high)
            if low > high:
                raise ValueError(
                    f"Int search-space {param_name!r} has low > high ({low} > {high})."
                )
            log = bool(spec.get("log", False))
            if log and low <= 0:
                raise ValueError(
                    f"Int search-space {param_name!r} cannot use log scale with low <= 0."
                )
            step = int(spec.get("step", 1))
            if step <= 0:
                raise ValueError(
                    f"Int search-space {param_name!r} must define a positive 'step'."
                )
            normalized[param_name] = {
                "type": "int",
                "low": low,
                "high": high,
                "log": log,
                "step": step,
            }
            continue

        if param_type == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                raise ValueError(
                    f"Categorical search-space {param_name!r} must define non-empty 'choices'."
                )
            normalized[param_name] = {"type": "categorical", "choices": choices}
            continue

        raise ValueError(
            f"Unsupported search-space type {param_type!r} for parameter {param_name!r}."
        )

    return normalized


def suggest_trial_params(
    trial: Trial,
    search_space: dict[str, dict[str, Any]],
    fixed_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample one trial configuration from a validated search space."""
    fixed_params = dict(fixed_params or {})
    params: dict[str, Any] = {}

    for param_name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        else:
            params[param_name] = trial.suggest_categorical(
                param_name,
                list(spec["choices"]),
            )

    params.update(fixed_params)
    return params


def serialize_optuna_trial(
    trial: FrozenTrial,
    fixed_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an Optuna trial into plain Python data for Kedro JSON artifacts."""
    params = dict(trial.params)
    params.update(dict(fixed_params or {}))

    return {
        "trial_number": int(trial.number),
        "state": trial.state.name.lower(),
        "value": _safe_float(trial.value),
        "params": _to_builtin(params),
        "user_attrs": _to_builtin(dict(trial.user_attrs)),
    }


def _safe_float(value: Any) -> float | None:
    """Convert to float, returning None for missing or non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy/Pandas-like values into builtin Python types."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
