"""Feature engineering pipeline for monthly granularity."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import aggregate_to_monthly, build_monthly_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_to_monthly,
                inputs="intermediate_demand_data",
                outputs="primary_monthly_demand",
                name="aggregate_to_monthly",
            ),
            node(
                func=build_monthly_features,
                inputs=[
                    "primary_monthly_demand",
                    "intermediate_exogenous_data",
                    "params:feature_engineering",
                ],
                outputs="feature_monthly_data",
                name="build_monthly_features",
            ),
        ]
    )
