"""Forecast inference pipeline: champion models → raw forward-looking predictions."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    allocate_daily_forecast,
    generate_monthly_forecast,
    generate_weekly_forecast,
    load_inference_inputs,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_inference_inputs,
                inputs=[
                    "monthly_prophet_features",
                    "feature_weekly_data",
                    "params:forecast_inference",
                ],
                outputs=["monthly_inference_df", "weekly_inference_df"],
                name="load_inference_inputs",
            ),
            node(
                func=generate_monthly_forecast,
                inputs=[
                    "champion_monthly_model",
                    "monthly_inference_df",
                    "params:forecast_inference",
                ],
                outputs="forecast_monthly_raw",
                name="generate_monthly_forecast",
            ),
            node(
                func=generate_weekly_forecast,
                inputs=[
                    "champion_weekly_model",
                    "weekly_inference_df",
                    "params:forecast_inference",
                ],
                outputs="forecast_weekly_raw",
                name="generate_weekly_forecast",
            ),
            node(
                func=allocate_daily_forecast,
                inputs=[
                    "forecast_weekly_reconciled",
                    "feature_weekly_data",
                    "params:forecast_inference",
                ],
                outputs="forecast_daily_output",
                name="allocate_daily_forecast",
            ),
        ]
    )
