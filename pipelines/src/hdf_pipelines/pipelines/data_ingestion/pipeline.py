"""Data ingestion pipeline: raw CSV → cleaned intermediate and primary datasets."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_demand_daily,
    build_demand_monthly,
    build_demand_weekly,
    build_exogenous_monthly,
    load_and_clean_demand,
    load_and_clean_exogenous,
    mask_raw_demand,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=mask_raw_demand,
                inputs="raw_daily_demand",
                outputs="raw_daily_demand_masked",
                name="mask_raw_demand",
            ),
            node(
                func=load_and_clean_demand,
                inputs="raw_daily_demand_masked",
                outputs="demand_cleaned",
                name="load_and_clean_demand",
            ),
            node(
                func=load_and_clean_exogenous,
                inputs="raw_exogenous_variables",
                outputs="exogenous_cleaned",
                name="load_and_clean_exogenous",
            ),
            node(
                func=build_demand_daily,
                inputs="demand_cleaned",
                outputs="demand_daily",
                name="build_demand_daily",
            ),
            node(
                func=build_demand_weekly,
                inputs="demand_cleaned",
                outputs="demand_weekly",
                name="build_demand_weekly",
            ),
            node(
                func=build_demand_monthly,
                inputs="demand_cleaned",
                outputs="demand_monthly",
                name="build_demand_monthly",
            ),
            node(
                func=build_exogenous_monthly,
                inputs="exogenous_cleaned",
                outputs="exogenous_monthly",
                name="build_exogenous_monthly",
            ),
        ]
    )
