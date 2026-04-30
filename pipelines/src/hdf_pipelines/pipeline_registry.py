"""Project pipeline registry."""

from kedro.pipeline import Pipeline

from hdf_pipelines.pipelines import (
    data_ingestion,
    feature_engineering_monthly,
    feature_engineering_weekly,
    forecast_inference,
    model_input_preparation,
    model_selection,
    reconciliation,
    train_monthly,
    train_weekly,
)
from hdf_pipelines.pipelines.model_selection.prophet.pipeline import (
    create_pipeline as create_prophet_monthly_selection_pipeline,
)
from hdf_pipelines.pipelines.train_monthly.prophet.pipeline import (
    create_pipeline as create_prophet_monthly_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        Mapping from pipeline name to Pipeline object. Includes individual stages
        and composed shortcuts for common execution patterns.
    """
    ingestion = data_ingestion.create_pipeline()
    fe_monthly = feature_engineering_monthly.create_pipeline()
    fe_weekly = feature_engineering_weekly.create_pipeline()
    model_input = model_input_preparation.create_pipeline()
    monthly_training = train_monthly.create_pipeline()
    weekly_training = train_weekly.create_pipeline()
    selection = model_selection.create_pipeline()
    recon = reconciliation.create_pipeline()
    inference = forecast_inference.create_pipeline()

    # Standalone Prophet-only training pipeline
    prophet_monthly = create_prophet_monthly_pipeline()

    # Standalone Monthly Prophet model-selection pipeline
    prophet_monthly_selection = create_prophet_monthly_selection_pipeline()

    training = monthly_training + weekly_training
    full_experiment = (
        ingestion + fe_monthly + fe_weekly + model_input + training + selection
    )
    inference_flow = inference + recon

    default = full_experiment + recon + inference

    return {
        "__default__": default,
        "data_ingestion": ingestion,
        "feature_engineering_monthly": fe_monthly,
        "feature_engineering_weekly": fe_weekly,
        "model_input_preparation": model_input,
        "train_monthly": monthly_training,
        "train_monthly_prophet": prophet_monthly,
        "model_selection_monthly_prophet": prophet_monthly_selection,
        "train_weekly": weekly_training,
        "model_selection": selection,
        "reconciliation": recon,
        "forecast_inference": inference,
        "training": training,
        "inference": inference_flow,
        "full_experiment": full_experiment,
    }
