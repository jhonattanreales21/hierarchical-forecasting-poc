"""Monthly Prophet model-selection pipeline.

Evaluates pre-champion candidates on the held-out test set, selects the final
champion, optionally refits on all historical data, and persists the champion
model and metadata for Stage 6 forecast inference.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_monthly_prophet_champion_model,
    evaluate_monthly_prophet_prechampions_on_test,
    evaluate_monthly_prophet_rolling_origin_metrics,
    select_monthly_prophet_champion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly Prophet model-selection pipeline.

    Inputs (from catalog):
        monthly_prophet_test
        monthly_prophet_train
        monthly_prophet_full_train
        monthly_prophet_split_metadata
        monthly_prophet_tuning_results
        monthly_prophet_prechampion_configs
        monthly_prophet_candidate_models
        params:model_selection.monthly_prophet

    Outputs (to catalog):
        monthly_prophet_test_metrics
        monthly_prophet_champion_test_forecast
        monthly_operational_test_forecasts
        monthly_operational_lead_time_metrics
        monthly_prophet_model_selection_summary
        monthly_prophet_champion_metadata
        monthly_model_selection_audit
        monthly_prophet_champion_model
    """
    return pipeline(
        [
            node(
                func=evaluate_monthly_prophet_prechampions_on_test,
                inputs=[
                    "monthly_prophet_test",
                    "monthly_prophet_candidate_models",
                    "monthly_prophet_prechampion_configs",
                    "monthly_prophet_tuning_results",
                    "monthly_prophet_train",
                    "params:model_selection.monthly_prophet",
                ],
                outputs=[
                    "monthly_prophet_test_metrics",
                    "monthly_prophet_champion_test_forecast",
                ],
                name="evaluate_monthly_prophet_prechampions_on_test",
            ),
            node(
                func=evaluate_monthly_prophet_rolling_origin_metrics,
                inputs=[
                    "monthly_prophet_full_train",
                    "monthly_prophet_test",
                    "monthly_prophet_prechampion_configs",
                    "monthly_prophet_split_metadata",
                    "params:model_selection.monthly_prophet",
                ],
                outputs=[
                    "monthly_operational_test_forecasts",
                    "monthly_operational_lead_time_metrics",
                ],
                name="evaluate_monthly_prophet_rolling_origin_metrics",
            ),
            node(
                func=select_monthly_prophet_champion,
                inputs=[
                    "monthly_prophet_test_metrics",
                    "monthly_prophet_tuning_results",
                    "monthly_prophet_prechampion_configs",
                    "monthly_prophet_split_metadata",
                    "monthly_operational_lead_time_metrics",
                    "params:model_selection.monthly_prophet",
                ],
                outputs=[
                    "monthly_prophet_model_selection_summary",
                    "monthly_prophet_champion_metadata",
                    "monthly_model_selection_audit",
                ],
                name="select_monthly_prophet_champion",
            ),
            node(
                func=build_monthly_prophet_champion_model,
                inputs=[
                    "monthly_prophet_full_train",
                    "monthly_prophet_candidate_models",
                    "monthly_prophet_champion_metadata",
                    "params:model_selection.monthly_prophet",
                ],
                outputs="monthly_prophet_champion_model",
                name="build_monthly_prophet_champion_model",
            ),
        ]
    )
