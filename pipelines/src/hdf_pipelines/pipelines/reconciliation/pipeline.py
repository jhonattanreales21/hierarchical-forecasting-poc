"""Reconciliation pipeline: raw forecasts → coherent hierarchical forecasts."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_reconciliation_diagnostics, reconcile_forecasts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=reconcile_forecasts,
                inputs=[
                    "forecast_monthly_raw",
                    "forecast_weekly_raw",
                    "params:reconciliation",
                ],
                outputs=["forecast_monthly_reconciled", "forecast_weekly_reconciled"],
                name="reconcile_forecasts",
            ),
            node(
                func=compute_reconciliation_diagnostics,
                inputs=[
                    "forecast_monthly_raw",
                    "forecast_weekly_raw",
                    "forecast_monthly_reconciled",
                    "forecast_weekly_reconciled",
                ],
                outputs="reconciliation_diagnostics",
                name="compute_reconciliation_diagnostics",
            ),
        ]
    )
