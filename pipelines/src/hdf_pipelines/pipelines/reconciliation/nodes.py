"""Reconciliation nodes: enforce monthly–weekly coherence on raw forecasts."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def reconcile_forecasts(
    forecast_monthly_raw: pd.DataFrame,
    forecast_weekly_raw: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hierarchical reconciliation to make monthly and weekly forecasts coherent.

    Args:
        forecast_monthly_raw: Raw monthly forecast output from the inference pipeline.
            Expected columns: ds, y_forecast, y_lower, y_upper, model_name.
        forecast_weekly_raw: Raw weekly forecast output from the inference pipeline.
            Same schema as forecast_monthly_raw.
        parameters: Config from reconciliation key. Expected keys: enabled,
            methods, default_method ('bottom_up' or 'mint_shrink').

    Returns:
        Tuple of (reconciled_monthly, reconciled_weekly) DataFrames where
        sum(weekly) == monthly for each overlapping month.
    """
    raise NotImplementedError(
        "If parameters['enabled'] is False, return the raw forecasts unchanged. "
        "Otherwise, implement the method specified in parameters['default_method']: "
        "for 'bottom_up', sum weekly forecasts to monthly; for 'mint_shrink', apply "
        "the MinT(Shrink) estimator using the covariance of in-sample residuals. "
        "Return (reconciled_monthly, reconciled_weekly)."
    )


def compute_reconciliation_diagnostics(
    forecast_monthly_raw: pd.DataFrame,
    forecast_weekly_raw: pd.DataFrame,
    forecast_monthly_reconciled: pd.DataFrame,
    forecast_weekly_reconciled: pd.DataFrame,
) -> pd.DataFrame:
    """Compute diagnostics measuring the coherence gap before and after reconciliation.

    Args:
        forecast_monthly_raw: Raw monthly forecast.
        forecast_weekly_raw: Raw weekly forecast.
        forecast_monthly_reconciled: Reconciled monthly forecast.
        forecast_weekly_reconciled: Reconciled weekly forecast.

    Returns:
        DataFrame with columns [month, pre_reconciliation_gap, post_reconciliation_gap,
        revision_monthly, revision_weekly] — one row per forecast month.
    """
    raise NotImplementedError(
        "For each calendar month, compute the gap between sum(weekly) and monthly "
        "before and after reconciliation. Calculate the revision magnitude for each "
        "granularity. Return a diagnostics DataFrame."
    )
