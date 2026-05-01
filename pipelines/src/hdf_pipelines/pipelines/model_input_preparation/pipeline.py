"""Kedro pipeline for Monthly Prophet model-input preparation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_monthly_prophet_future_regressors,
    build_monthly_prophet_split_metadata,
    prepare_monthly_prophet_modeling_data,
    split_monthly_prophet_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Monthly Prophet model-input preparation pipeline."""
    return pipeline(
        [
            node(
                func=prepare_monthly_prophet_modeling_data,
                inputs=["monthly_prophet_features", "params:model_input_preparation"],
                outputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_prophet_preparation_metadata",
                ],
                name="prepare_monthly_prophet_modeling_data",
            ),
            node(
                func=split_monthly_prophet_data,
                inputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_prophet_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_test",
                    "monthly_prophet_full_train",
                    "monthly_prophet_split_preparation_metadata",
                ],
                name="split_monthly_prophet_data",
            ),
            node(
                func=build_monthly_prophet_future_regressors,
                inputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_calendar_features",
                    "monthly_exogenous_features",
                    "params:model_input_preparation",
                    "params:feature_engineering_monthly",
                ],
                outputs=[
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                ],
                name="build_monthly_prophet_future_regressors",
            ),
            node(
                func=build_monthly_prophet_split_metadata,
                inputs=[
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_test",
                    "monthly_prophet_full_train",
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                    "monthly_prophet_split_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs="monthly_prophet_split_metadata",
                name="build_monthly_prophet_split_metadata",
            ),
        ]
    )
