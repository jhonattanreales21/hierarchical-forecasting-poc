"""Forecast inference pipeline: Monthly Prophet champion → official forecast outputs."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_monthly_prophet_forecasts


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Monthly Prophet forecast inference pipeline.

    Inputs (from catalog):
        monthly_prophet_champion_model
        monthly_prophet_champion_metadata
        monthly_prophet_future_3m
        monthly_prophet_future_6m
        monthly_prophet_future_12m
        params:forecast_inference.monthly_prophet

    Outputs (to catalog):
        monthly_prophet_forecast_3m
        monthly_prophet_forecast_6m
        monthly_prophet_forecast_12m
        monthly_prophet_forecast_latest
        monthly_prophet_inference_metadata
    """
    return pipeline(
        [
            node(
                func=generate_monthly_prophet_forecasts,
                inputs=[
                    "monthly_prophet_champion_model",
                    "monthly_prophet_champion_metadata",
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                    "params:forecast_inference.monthly_prophet",
                ],
                outputs=[
                    "monthly_prophet_forecast_3m",
                    "monthly_prophet_forecast_6m",
                    "monthly_prophet_forecast_12m",
                    "monthly_prophet_forecast_latest",
                    "monthly_prophet_inference_metadata",
                ],
                name="generate_monthly_prophet_forecasts",
            ),
        ]
    )
