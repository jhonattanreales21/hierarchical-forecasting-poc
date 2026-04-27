"""Model input preparation pipeline: feature data → per-model splits and backtest folds."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_monthly_splits_catboost,
    build_monthly_splits_prophet,
    build_monthly_splits_sarimax,
    build_weekly_splits_catboost,
    build_weekly_splits_prophet,
    build_weekly_splits_sarimax,
    generate_backtest_folds_monthly,
    generate_backtest_folds_weekly,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_monthly_splits_prophet,
                inputs=["monthly_prophet_features", "params:model_input"],
                outputs=[
                    "model_input_monthly_prophet_train",
                    "model_input_monthly_prophet_validation",
                    "model_input_monthly_prophet_test",
                ],
                name="build_monthly_splits_prophet",
            ),
            node(
                func=build_monthly_splits_catboost,
                inputs=["monthly_prophet_features", "params:model_input"],
                outputs=[
                    "model_input_monthly_catboost_train",
                    "model_input_monthly_catboost_validation",
                    "model_input_monthly_catboost_test",
                ],
                name="build_monthly_splits_catboost",
            ),
            node(
                func=build_monthly_splits_sarimax,
                inputs=["monthly_prophet_features", "params:model_input"],
                outputs=[
                    "model_input_monthly_sarimax_train",
                    "model_input_monthly_sarimax_validation",
                    "model_input_monthly_sarimax_test",
                ],
                name="build_monthly_splits_sarimax",
            ),
            node(
                func=build_weekly_splits_prophet,
                inputs=["feature_weekly_data", "params:model_input"],
                outputs=[
                    "model_input_weekly_prophet_train",
                    "model_input_weekly_prophet_validation",
                    "model_input_weekly_prophet_test",
                ],
                name="build_weekly_splits_prophet",
            ),
            node(
                func=build_weekly_splits_catboost,
                inputs=["feature_weekly_data", "params:model_input"],
                outputs=[
                    "model_input_weekly_catboost_train",
                    "model_input_weekly_catboost_validation",
                    "model_input_weekly_catboost_test",
                ],
                name="build_weekly_splits_catboost",
            ),
            node(
                func=build_weekly_splits_sarimax,
                inputs=["feature_weekly_data", "params:model_input"],
                outputs=[
                    "model_input_weekly_sarimax_train",
                    "model_input_weekly_sarimax_validation",
                    "model_input_weekly_sarimax_test",
                ],
                name="build_weekly_splits_sarimax",
            ),
            node(
                func=generate_backtest_folds_monthly,
                inputs=["monthly_prophet_features", "params:model_input"],
                outputs="backtest_folds_monthly",
                name="generate_backtest_folds_monthly",
            ),
            node(
                func=generate_backtest_folds_weekly,
                inputs=["feature_weekly_data", "params:model_input"],
                outputs="backtest_folds_weekly",
                name="generate_backtest_folds_weekly",
            ),
        ]
    )
