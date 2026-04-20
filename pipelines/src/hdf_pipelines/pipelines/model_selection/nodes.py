"""Model selection nodes: evaluate candidates on test sets and elect champion models."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_candidates_on_test(
    candidate_monthly_prophet: Any,
    candidate_monthly_catboost: Any,
    candidate_monthly_sarimax: Any,
    candidate_weekly_prophet: Any,
    candidate_weekly_catboost: Any,
    candidate_weekly_sarimax: Any,
    model_input_monthly_prophet_test: pd.DataFrame,
    model_input_monthly_catboost_test: pd.DataFrame,
    model_input_monthly_sarimax_test: pd.DataFrame,
    model_input_weekly_prophet_test: pd.DataFrame,
    model_input_weekly_catboost_test: pd.DataFrame,
    model_input_weekly_sarimax_test: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Score all candidate models on their respective held-out test sets.

    Args:
        candidate_monthly_prophet: Trained Prophet model for monthly data.
        candidate_monthly_catboost: Trained CatBoost model for monthly data.
        candidate_monthly_sarimax: Fitted SARIMAX model for monthly data.
        candidate_weekly_prophet: Trained Prophet model for weekly data.
        candidate_weekly_catboost: Trained CatBoost model for weekly data.
        candidate_weekly_sarimax: Fitted SARIMAX model for weekly data.
        model_input_monthly_prophet_test: Monthly Prophet test split.
        model_input_monthly_catboost_test: Monthly CatBoost test split.
        model_input_monthly_sarimax_test: Monthly SARIMAX test split.
        model_input_weekly_prophet_test: Weekly Prophet test split.
        model_input_weekly_catboost_test: Weekly CatBoost test split.
        model_input_weekly_sarimax_test: Weekly SARIMAX test split.
        parameters: Config from model_selection key with primary_metric and
            secondary_metrics.

    Returns:
        DataFrame with columns [model_name, granularity, MAPE, RMSE, MASE, horizon]
        containing one row per (model, granularity) combination.
    """
    raise NotImplementedError(
        "For each candidate model, generate predictions on the corresponding test split, "
        "compute MAPE/RMSE/MASE using shared.metrics, and collect results into a DataFrame."
    )


def select_champion_models(
    evaluation_report: pd.DataFrame,
    candidate_monthly_prophet: Any,
    candidate_monthly_catboost: Any,
    candidate_monthly_sarimax: Any,
    candidate_weekly_prophet: Any,
    candidate_weekly_catboost: Any,
    candidate_weekly_sarimax: Any,
    parameters: dict,
) -> tuple[Any, Any]:
    """Select the best model per granularity based on the evaluation report.

    Args:
        evaluation_report: Output of evaluate_candidates_on_test.
        candidate_monthly_prophet: Trained Prophet (monthly).
        candidate_monthly_catboost: Trained CatBoost (monthly).
        candidate_monthly_sarimax: Fitted SARIMAX (monthly).
        candidate_weekly_prophet: Trained Prophet (weekly).
        candidate_weekly_catboost: Trained CatBoost (weekly).
        candidate_weekly_sarimax: Fitted SARIMAX (weekly).
        parameters: Config from model_selection key with primary_metric and
            selection_strategy.

    Returns:
        Tuple of (champion_monthly_model, champion_weekly_model) — the best model
        objects for each granularity, ready to be serialised to data/06_models/champions/.
    """
    raise NotImplementedError(
        "Filter evaluation_report by granularity, rank by primary_metric, use "
        "tie_breaker if needed, and return the corresponding model object for each "
        "granularity."
    )


def persist_champion_registry(
    champion_monthly: Any,
    champion_weekly: Any,
    evaluation_report: pd.DataFrame,
) -> dict:
    """Build a JSON-serialisable registry mapping granularity to champion model info.

    Args:
        champion_monthly: The elected monthly champion model.
        champion_weekly: The elected weekly champion model.
        evaluation_report: Full evaluation scores DataFrame.

    Returns:
        Dict with keys 'monthly' and 'weekly', each containing model_name, metrics,
        and timestamp. Written to data/06_models/champions/champion_registry.json.
    """
    raise NotImplementedError(
        "Extract model class name and best metric scores from evaluation_report, "
        "add an ISO-format timestamp, and return a dict keyed by granularity."
    )
