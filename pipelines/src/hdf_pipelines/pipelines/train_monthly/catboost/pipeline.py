"""Monthly CatBoost training and tuning pipeline (rolling-origin, direct multi-horizon).

Produces all CatBoost training artifacts using a rolling-origin backtest with the
direct multi-horizon strategy: one independent model per horizon h ∈ {1,2,3},
no recursion. The Optuna objective is pooled WMAPE_M3. Champions are selected
directly from rolling-origin metrics — no separate held-out test stage.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_monthly_catboost_candidates


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly CatBoost training pipeline (rolling-origin, direct multi-horizon).

    Inputs (from catalog):
        monthly_catboost_full_train
        monthly_catboost_split_metadata
        params:train_monthly.catboost

    Outputs (to catalog):
        monthly_catboost_tuning_results
        monthly_catboost_rolling_origin_metrics
        monthly_catboost_prechampion_configs
        monthly_catboost_candidate_models
        monthly_catboost_training_metadata
        monthly_catboost_rolling_origin_predictions
    """
    return pipeline(
        [
            node(
                func=train_monthly_catboost_candidates,
                inputs=[
                    "monthly_catboost_full_train",
                    "monthly_catboost_split_metadata",
                    "params:train_monthly.catboost",
                ],
                outputs=[
                    "monthly_catboost_tuning_results",
                    "monthly_catboost_rolling_origin_metrics",
                    "monthly_catboost_prechampion_configs",
                    "monthly_catboost_candidate_models",
                    "monthly_catboost_training_metadata",
                    "monthly_catboost_rolling_origin_predictions",
                ],
                name="train_monthly_catboost_candidates",
            ),
        ]
    )
