"""Monthly SARIMAX training and tuning pipeline.

Produces all SARIMAX artifacts: tuning results, validation metrics,
pre-champion configurations, candidate model artifacts, training metadata,
and the rank-1 candidate for model-selection.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_evaluate_monthly_sarimax_candidates


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly SARIMAX training and tuning pipeline.

    Inputs (from catalog):
        monthly_sarimax_full_train
        monthly_sarimax_split_metadata
        params:train_monthly.sarimax

    Outputs (to catalog):
        monthly_sarimax_tuning_results
        monthly_sarimax_rolling_origin_metrics
        monthly_sarimax_prechampion_configs
        monthly_sarimax_candidate_models
        monthly_sarimax_training_metadata
        candidate_monthly_sarimax          ← rank-1 full-history model for model selection
        monthly_sarimax_rolling_origin_predictions
    """
    return pipeline(
        [
            node(
                func=train_and_evaluate_monthly_sarimax_candidates,
                inputs=[
                    "monthly_sarimax_full_train",
                    "monthly_sarimax_split_metadata",
                    "params:train_monthly.sarimax",
                ],
                outputs=[
                    "monthly_sarimax_tuning_results",
                    "monthly_sarimax_rolling_origin_metrics",
                    "monthly_sarimax_prechampion_configs",
                    "monthly_sarimax_candidate_models",
                    "monthly_sarimax_training_metadata",
                    "candidate_monthly_sarimax",
                    "monthly_sarimax_rolling_origin_predictions",
                ],
                name="train_and_evaluate_monthly_sarimax_candidates",
            ),
        ]
    )
