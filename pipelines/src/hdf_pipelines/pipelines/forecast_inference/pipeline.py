"""Forecast inference pipeline: metadata-driven monthly champion → forecast outputs."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_monthly_champion_forecasts


def create_pipeline(**kwargs) -> Pipeline:
    """Create the metadata-driven monthly forecast inference pipeline.

    Loads the generic production champion (Prophet, SARIMAX, or CatBoost) and
    dispatches inference by ``champion_monthly_metadata["model_family"]``, producing
    a single standardized monthly forecast schema for every supported family.

    CatBoost inference requires the historical demand buffer (``monthly_catboost_full_train``)
    and split metadata (``monthly_catboost_split_metadata``) to seed recursive lag and
    rolling features.  These inputs are loaded for every run but are silently ignored
    when the champion family is Prophet or SARIMAX.

    Inputs (from catalog):
        champion_monthly_model
        champion_monthly_metadata
        monthly_future_3m
        monthly_future_6m
        monthly_future_12m
        params:forecast_inference.monthly
        monthly_catboost_full_train
        monthly_catboost_split_metadata

    Outputs (to catalog):
        monthly_forecast_3m
        monthly_forecast_6m
        monthly_forecast_12m
        monthly_forecast_latest
        monthly_inference_metadata
    """
    return pipeline(
        [
            node(
                func=generate_monthly_champion_forecasts,
                inputs=[
                    "champion_monthly_model",
                    "champion_monthly_metadata",
                    "monthly_future_3m",
                    "monthly_future_6m",
                    "monthly_future_12m",
                    "params:forecast_inference.monthly",
                    "monthly_catboost_full_train",
                    "monthly_catboost_split_metadata",
                ],
                outputs=[
                    "monthly_forecast_3m",
                    "monthly_forecast_6m",
                    "monthly_forecast_12m",
                    "monthly_forecast_latest",
                    "monthly_inference_metadata",
                ],
                name="generate_monthly_champion_forecasts",
            ),
        ]
    )
