"""Model selection pipeline: score candidates and elect champions per granularity."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_candidates_on_test,
    persist_champion_registry,
    select_champion_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_candidates_on_test,
                inputs=[
                    "candidate_monthly_prophet",
                    "candidate_monthly_catboost",
                    "candidate_monthly_sarimax",
                    "candidate_weekly_prophet",
                    "candidate_weekly_catboost",
                    "candidate_weekly_sarimax",
                    "model_input_monthly_prophet_test",
                    "model_input_monthly_catboost_test",
                    "model_input_monthly_sarimax_test",
                    "model_input_weekly_prophet_test",
                    "model_input_weekly_catboost_test",
                    "model_input_weekly_sarimax_test",
                    "params:model_selection",
                ],
                outputs="model_selection_report",
                name="evaluate_candidates_on_test",
            ),
            node(
                func=select_champion_models,
                inputs=[
                    "model_selection_report",
                    "candidate_monthly_prophet",
                    "candidate_monthly_catboost",
                    "candidate_monthly_sarimax",
                    "candidate_weekly_prophet",
                    "candidate_weekly_catboost",
                    "candidate_weekly_sarimax",
                    "params:model_selection",
                ],
                outputs=["champion_monthly_model", "champion_weekly_model"],
                name="select_champion_models",
            ),
            node(
                func=persist_champion_registry,
                inputs=[
                    "champion_monthly_model",
                    "champion_weekly_model",
                    "model_selection_report",
                ],
                outputs="champion_registry",
                name="persist_champion_registry",
            ),
        ]
    )
