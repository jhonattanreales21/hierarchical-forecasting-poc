"""Monthly Prophet training and tuning pipeline.

Produces all Stage-4 artifacts: tuning results, validation metrics, pre-champion
configurations, candidate model artifacts, and the best model for model-selection.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_evaluate_monthly_prophet_candidates


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly Prophet training and tuning pipeline.

    Inputs (from catalog):
        monthly_prophet_train
        monthly_prophet_validation
        monthly_prophet_split_metadata
        params:train_monthly.prophet

    Outputs (to catalog):
        monthly_prophet_tuning_results
        monthly_prophet_validation_metrics
        monthly_prophet_prechampion_configs
        monthly_prophet_candidate_models
        monthly_prophet_training_metadata
        candidate_monthly_prophet          ← rank-1 model for model-selection stage
    """
    return pipeline(
        [
            node(
                func=train_and_evaluate_monthly_prophet_candidates,
                inputs=[
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_split_metadata",
                    "params:train_monthly.prophet",
                ],
                outputs=[
                    "monthly_prophet_tuning_results",
                    "monthly_prophet_validation_metrics",
                    "monthly_prophet_prechampion_configs",
                    "monthly_prophet_candidate_models",
                    "monthly_prophet_training_metadata",
                    "candidate_monthly_prophet",
                ],
                name="train_and_evaluate_monthly_prophet_candidates",
            ),
        ]
    )
