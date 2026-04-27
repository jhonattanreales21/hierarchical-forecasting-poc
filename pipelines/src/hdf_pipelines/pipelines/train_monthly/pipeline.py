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
            "train": "model_input_monthly_prophet_train",
            "validation": "model_input_monthly_prophet_validation",
        },
        parameters={"parameters": "params:train_monthly"},
        outputs={"candidate_model": "candidate_monthly_prophet"},
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
            "train": "model_input_monthly_sarimax_train",
            "validation": "model_input_monthly_sarimax_validation",
        },
        parameters={"parameters": "params:train_monthly"},
        outputs={"candidate_model": "candidate_monthly_sarimax"},
    )
    return prophet_pipe + catboost_pipe + sarimax_pipe
