"""Monthly CatBoost training and tuning pipeline.

Produces all CatBoost training artifacts: tuning results, validation metrics,
pre-champion configurations, candidate model artifacts, and training metadata.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_monthly_catboost_candidates


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly CatBoost training and tuning pipeline.

    Inputs (from catalog):
        monthly_catboost_train
        monthly_catboost_validation
        monthly_catboost_split_metadata
        params:train_monthly.catboost

    Outputs (to catalog):
        monthly_catboost_tuning_results
        monthly_catboost_validation_metrics
        monthly_catboost_prechampion_configs
        monthly_catboost_candidate_models
        monthly_catboost_training_metadata
    """
    return pipeline(
        [
            node(
                func=train_monthly_catboost_candidates,
                inputs=[
                    "monthly_catboost_train",
                    "monthly_catboost_validation",
                    "monthly_catboost_split_metadata",
                    "params:train_monthly.catboost",
                ],
                outputs=[
                    "monthly_catboost_tuning_results",
                    "monthly_catboost_validation_metrics",
                    "monthly_catboost_prechampion_configs",
                    "monthly_catboost_candidate_models",
                    "monthly_catboost_training_metadata",
                ],
                name="train_monthly_catboost_candidates",
            ),
        ]
    )
