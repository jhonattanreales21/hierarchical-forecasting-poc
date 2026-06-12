"""Monthly multi-family model selection pipeline (rolling-origin protocol).

Champions are selected directly from the macro-averaged rolling-origin metrics
produced by training — there is no separate held-out test stage (protocol §4, §11).
Phase 1 compares Prophet and SARIMAX; CatBoost is reintroduced in Phase 2.

Inputs (from catalog):
    monthly_prophet_prechampion_configs
    monthly_sarimax_prechampion_configs
    monthly_prophet_candidate_models
    monthly_sarimax_candidate_models
    monthly_prophet_full_train
    monthly_sarimax_full_train
    monthly_sarimax_training_metadata
    params:model_selection.monthly

Outputs (to catalog):
    monthly_candidate_metrics
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
    assemble_monthly_candidate_metrics,
    build_monthly_champion_artifacts,
    generate_monthly_family_champion_explanations,
    select_monthly_family_champions,
    select_monthly_production_champion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly multi-family model selection pipeline."""
    return pipeline(
        [
            node(
                func=assemble_monthly_candidate_metrics,
                inputs=[
                    "monthly_prophet_prechampion_configs",
                    "monthly_sarimax_prechampion_configs",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_candidate_metrics_unflagged",
                name="assemble_monthly_candidate_metrics",
            ),
            node(
                func=select_monthly_family_champions,
                inputs=[
                    "monthly_candidate_metrics_unflagged",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_family_champion_summary",
                name="select_monthly_family_champions",
            ),
            node(
                func=select_monthly_production_champion,
                inputs=[
                    "monthly_family_champion_summary",
                    "monthly_candidate_metrics_unflagged",
                    "params:model_selection.monthly",
                ],
                outputs="monthly_model_selection_summary",
                name="select_monthly_production_champion",
            ),
            node(
                func=annotate_monthly_candidate_champion_flags,
                inputs=[
                    "monthly_candidate_metrics_unflagged",
                    "monthly_family_champion_summary",
                    "monthly_model_selection_summary",
                ],
                outputs="monthly_candidate_metrics",
                name="annotate_monthly_candidate_champion_flags",
            ),
            node(
                func=build_monthly_champion_artifacts,
                inputs=[
                    "monthly_model_selection_summary",
                    "monthly_family_champion_summary",
                    "monthly_candidate_metrics",
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
            node(
                func=generate_monthly_family_champion_explanations,
                inputs=[
                    "monthly_family_champion_summary",
                    "monthly_prophet_candidate_models",
                    "monthly_sarimax_candidate_models",
                    "monthly_prophet_full_train",
                    "monthly_sarimax_full_train",
                    "monthly_sarimax_training_metadata",
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
