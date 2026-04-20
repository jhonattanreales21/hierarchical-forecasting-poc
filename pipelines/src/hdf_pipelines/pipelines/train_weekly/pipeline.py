"""Weekly training pipeline: composes prophet, catboost, and sarimax sub-pipelines."""

from kedro.pipeline import Pipeline, pipeline

from .catboost.pipeline import create_pipeline as create_catboost_pipeline
from .prophet.pipeline import create_pipeline as create_prophet_pipeline
from .sarimax.pipeline import create_pipeline as create_sarimax_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    prophet_pipe = pipeline(
        create_prophet_pipeline(),
        namespace="train_weekly.prophet",
        inputs={
            "train": "model_input_weekly_prophet_train",
            "validation": "model_input_weekly_prophet_validation",
        },
        parameters={"parameters": "params:train_weekly"},
        outputs={"candidate_model": "candidate_weekly_prophet"},
    )
    catboost_pipe = pipeline(
        create_catboost_pipeline(),
        namespace="train_weekly.catboost",
        inputs={
            "train": "model_input_weekly_catboost_train",
            "validation": "model_input_weekly_catboost_validation",
        },
        parameters={"parameters": "params:train_weekly"},
        outputs={"candidate_model": "candidate_weekly_catboost"},
    )
    sarimax_pipe = pipeline(
        create_sarimax_pipeline(),
        namespace="train_weekly.sarimax",
        inputs={
            "train": "model_input_weekly_sarimax_train",
            "validation": "model_input_weekly_sarimax_validation",
        },
        parameters={"parameters": "params:train_weekly"},
        outputs={"candidate_model": "candidate_weekly_sarimax"},
    )
    return prophet_pipe + catboost_pipe + sarimax_pipe
