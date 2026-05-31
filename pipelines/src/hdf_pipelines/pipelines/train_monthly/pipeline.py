"""Monthly training pipeline: composes prophet, catboost, and sarimax sub-pipelines."""

from kedro.pipeline import Pipeline, pipeline

from .catboost.pipeline import create_pipeline as create_catboost_pipeline
from .prophet.pipeline import create_pipeline as create_prophet_pipeline
from .sarimax.pipeline import create_pipeline as create_sarimax_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    prophet_pipe = pipeline(
        create_prophet_pipeline(),
        namespace="train_monthly.prophet",
        inputs={
            "monthly_prophet_train": "monthly_prophet_train",
            "monthly_prophet_validation": "monthly_prophet_validation",
            "monthly_prophet_split_metadata": "monthly_prophet_split_metadata",
        },
        outputs={
            "monthly_prophet_tuning_results": "monthly_prophet_tuning_results",
            "monthly_prophet_validation_metrics": "monthly_prophet_validation_metrics",
            "monthly_prophet_prechampion_configs": "monthly_prophet_prechampion_configs",
            "monthly_prophet_candidate_models": "monthly_prophet_candidate_models",
            "monthly_prophet_training_metadata": "monthly_prophet_training_metadata",
            "candidate_monthly_prophet": "candidate_monthly_prophet",
        },
        parameters={"train_monthly.prophet": "train_monthly.prophet"},
    )

    catboost_pipe = pipeline(
        create_catboost_pipeline(),
        namespace="train_monthly.catboost",
        inputs={
            "train": "model_input_monthly_catboost_train",
            "validation": "model_input_monthly_catboost_validation",
        },
        parameters={"parameters": "params:train_monthly"},
        outputs={"candidate_model": "candidate_monthly_catboost"},
    )
    sarimax_pipe = pipeline(
        create_sarimax_pipeline(),
        namespace="train_monthly.sarimax",
        inputs={
            "monthly_sarimax_train": "monthly_sarimax_train",
            "monthly_sarimax_validation": "monthly_sarimax_validation",
            "monthly_sarimax_split_metadata": "monthly_sarimax_split_metadata",
        },
        outputs={
            "monthly_sarimax_tuning_results": "monthly_sarimax_tuning_results",
            "monthly_sarimax_validation_metrics": "monthly_sarimax_validation_metrics",
            "monthly_sarimax_prechampion_configs": "monthly_sarimax_prechampion_configs",
            "monthly_sarimax_candidate_models": "monthly_sarimax_candidate_models",
            "monthly_sarimax_training_metadata": "monthly_sarimax_training_metadata",
            "candidate_monthly_sarimax": "candidate_monthly_sarimax",
        },
        parameters={"train_monthly.sarimax": "train_monthly.sarimax"},
    )
    return prophet_pipe + catboost_pipe + sarimax_pipe
