"""Monthly multi-family model selection pipeline.

Compares Prophet and SARIMAX prechampion candidates on the held-out test set,
selects one family champion per family, elects one monthly production champion,
and persists the generic champion artifacts.

Inputs (from catalog):
    monthly_prophet_candidate_models
    monthly_prophet_prechampion_configs
    monthly_prophet_train
    monthly_prophet_validation
    monthly_prophet_test
    monthly_sarimax_candidate_models
    monthly_sarimax_prechampion_configs
    monthly_sarimax_training_metadata
    monthly_sarimax_train
    monthly_sarimax_validation
    monthly_sarimax_test
    monthly_sarimax_full_train
    monthly_prophet_full_train
    params:model_selection.monthly
    params:model_selection.monthly_prophet
    params:model_selection.monthly_sarimax

Outputs (to catalog):
    monthly_candidate_test_metrics
    monthly_family_champion_summary
    monthly_model_selection_summary
    champion_monthly_model
    champion_monthly_metadata
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_monthly_champion_artifacts,
    evaluate_monthly_family_candidates_on_test,
    select_monthly_family_champions,
    select_monthly_production_champion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly multi-family model selection pipeline."""
    return pipeline(
        [
            node(
                func=evaluate_monthly_family_candidates_on_test,
                inputs=[
                    "monthly_prophet_candidate_models",
                    "monthly_prophet_prechampion_configs",
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_test",
                    "monthly_sarimax_candidate_models",
                    "monthly_sarimax_prechampion_configs",
                    "monthly_sarimax_training_metadata",
                    "monthly_sarimax_train",
                    "monthly_sarimax_validation",
                    "monthly_sarimax_test",
                    "params:model_selection.monthly",
                    "params:model_selection.monthly_prophet",
                    "params:model_selection.monthly_sarimax",
                ],
                outputs="monthly_candidate_test_metrics",
                name="evaluate_monthly_family_candidates_on_test",
            ),
            node(
                func=select_monthly_family_champions,
                inputs=[
                    "monthly_candidate_test_metrics",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_family_champion_summary",
                name="select_monthly_family_champions",
            ),
            node(
                func=select_monthly_production_champion,
                inputs=[
                    "monthly_family_champion_summary",
                    "monthly_candidate_test_metrics",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_model_selection_summary",
                name="select_monthly_production_champion",
            ),
            node(
                func=build_monthly_champion_artifacts,
                inputs=[
                    "monthly_model_selection_summary",
                    "monthly_family_champion_summary",
                    "monthly_candidate_test_metrics",
                    "monthly_prophet_candidate_models",
                    "monthly_sarimax_candidate_models",
                    "monthly_prophet_full_train",
                    "monthly_sarimax_full_train",
                    "monthly_sarimax_training_metadata",
                    "params:model_selection.monthly",
                ],
                outputs=["champion_monthly_model", "champion_monthly_metadata"],
                name="build_monthly_champion_artifacts",
            ),
        ]
    )
