"""Weekly SARIMAX sub-pipeline: tune then train."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_best_candidate, tune_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=tune_hyperparameters,
                inputs=["train", "validation", "parameters"],
                outputs="tuning_result",
                name="tune_hyperparameters",
            ),
            node(
                func=train_best_candidate,
                inputs=["train", "validation", "tuning_result"],
                outputs="candidate_model",
                name="train_best_candidate",
            ),
        ]
    )
