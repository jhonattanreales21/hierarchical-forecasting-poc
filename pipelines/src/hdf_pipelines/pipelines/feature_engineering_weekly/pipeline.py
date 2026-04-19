"""Feature engineering pipeline for weekly granularity."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import aggregate_to_weekly, build_weekly_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aggregate_to_weekly,
                inputs="intermediate_demand_data",
                outputs="primary_weekly_demand",
                name="aggregate_to_weekly",
            ),
            node(
                func=build_weekly_features,
                inputs=[
                    "primary_weekly_demand",
                    "intermediate_exogenous_data",
                    "params:feature_engineering",
                ],
                outputs="feature_weekly_data",
                name="build_weekly_features",
            ),
        ]
    )
