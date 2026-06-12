"""Monthly multi-family model selection pipeline.

Compares Prophet, SARIMAX, and CatBoost prechampion candidates on the held-out test
set, selects one family champion per family, elects one monthly production champion,
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
    monthly_catboost_candidate_models
    monthly_catboost_prechampion_configs
    monthly_catboost_split_metadata
    monthly_catboost_train
    monthly_catboost_validation
    monthly_catboost_test
    monthly_catboost_full_train
    params:model_selection.monthly
    params:model_selection.monthly_prophet
    params:model_selection.monthly_sarimax
    params:model_selection.monthly_catboost

Outputs (to catalog):
    monthly_candidate_test_metrics
    monthly_family_champion_summary
    monthly_model_selection_summary
    champion_monthly_model
    champion_monthly_metadata
    monthly_family_champion_importance
    monthly_catboost_shap_explainer
    monthly_catboost_shap_values
    monthly_family_champion_explainability_metadata
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    annotate_monthly_candidate_champion_flags,
    build_monthly_champion_artifacts,
    evaluate_monthly_family_candidates_on_test,
    generate_monthly_family_champion_explanations,
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
                    "monthly_catboost_candidate_models",
                    "monthly_catboost_prechampion_configs",
                    "monthly_catboost_split_metadata",
                    "monthly_catboost_train",
                    "monthly_catboost_validation",
                    "monthly_catboost_test",
                    "params:model_selection.monthly_catboost",
                ],
                outputs="monthly_candidate_test_metrics_unflagged",
                name="evaluate_monthly_family_candidates_on_test",
            ),
            node(
                func=select_monthly_family_champions,
                inputs=[
                    "monthly_candidate_test_metrics_unflagged",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_family_champion_summary",
                name="select_monthly_family_champions",
            ),
            node(
                func=select_monthly_production_champion,
                inputs=[
                    "monthly_family_champion_summary",
                    "monthly_candidate_test_metrics_unflagged",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_model_selection_summary",
                name="select_monthly_production_champion",
            ),
            node(
                func=annotate_monthly_candidate_champion_flags,
                inputs=[
                    "monthly_candidate_test_metrics_unflagged",
                    "monthly_family_champion_summary",
                    "monthly_model_selection_summary",
                ],
                outputs="monthly_candidate_test_metrics",
                name="annotate_monthly_candidate_champion_flags",
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
                    "monthly_catboost_candidate_models",
                    "monthly_catboost_full_train",
                    "monthly_catboost_split_metadata",
                ],
                outputs=["champion_monthly_model", "champion_monthly_metadata"],
                name="build_monthly_champion_artifacts",
            ),
            node(
                func=generate_monthly_family_champion_explanations,
                inputs=[
                    "monthly_family_champion_summary",
                    "monthly_prophet_candidate_models",
                    "monthly_sarimax_candidate_models",
                    "monthly_catboost_candidate_models",
                    "monthly_prophet_full_train",
                    "monthly_sarimax_full_train",
                    "monthly_catboost_full_train",
                    "monthly_sarimax_training_metadata",
                    "monthly_catboost_split_metadata",
                    "params:model_selection.monthly",
                ],
                outputs=[
                    "monthly_family_champion_importance",
                    "monthly_catboost_shap_explainer",
                    "monthly_catboost_shap_values",
                    "monthly_family_champion_explainability_metadata",
                ],
                name="generate_monthly_family_champion_explanations",
            ),
        ]
    )
