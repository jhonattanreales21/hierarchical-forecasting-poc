"""Forecast inference pipeline: metadata-driven monthly champion → forecast outputs."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_monthly_champion_forecasts


def create_pipeline(**kwargs) -> Pipeline:
    """Create the metadata-driven monthly forecast inference pipeline.

    Loads the generic production champion (Prophet or SARIMAX) and dispatches
    inference by ``champion_monthly_metadata["model_family"]``, producing a single
    standardized monthly forecast schema for every supported family.

    Inputs (from catalog):
        champion_monthly_model
        champion_monthly_metadata
        monthly_prophet_future_3m
        monthly_prophet_future_6m
        monthly_prophet_future_12m
        params:forecast_inference.monthly

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
                    # Temporary compatibility source until generic future frames exist.
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                    "params:forecast_inference.monthly",
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
