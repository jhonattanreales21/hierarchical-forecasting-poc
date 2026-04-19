"""Data ingestion pipeline: raw CSV → cleaned intermediate parquet files."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_and_clean_demand, load_and_clean_exogenous


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_clean_demand,
                inputs="raw_demand_data",
                outputs="intermediate_demand_data",
                name="load_and_clean_demand",
            ),
            node(
                func=load_and_clean_exogenous,
                inputs="raw_exogenous_data",
                outputs="intermediate_exogenous_data",
                name="load_and_clean_exogenous",
            ),
        ]
    )
