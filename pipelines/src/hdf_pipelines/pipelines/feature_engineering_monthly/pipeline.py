"""Kedro pipeline for monthly feature engineering."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_monthly_calendar_features,
    build_monthly_exogenous_features,
    build_monthly_prophet_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly feature engineering pipeline."""
    return pipeline(
        [
            node(
                func=build_monthly_calendar_features,
                inputs=["demand_monthly", "params:feature_engineering_monthly"],
                outputs="monthly_calendar_features",
                name="build_monthly_calendar_features",
            ),
            node(
                func=build_monthly_exogenous_features,
                inputs=["exogenous_monthly", "params:feature_engineering_monthly"],
                outputs="monthly_exogenous_features",
                name="build_monthly_exogenous_features",
            ),
            node(
                func=build_monthly_prophet_features,
                inputs=[
                    "demand_monthly",
                    "monthly_calendar_features",
                    "monthly_exogenous_features",
                    "params:feature_engineering_monthly",
                ],
                outputs="monthly_prophet_features",
                name="build_monthly_prophet_features",
            ),
        ]
    )
