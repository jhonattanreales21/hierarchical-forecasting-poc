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
from hdf_pipelines.pipelines.model_selection.monthly.pipeline import (
    create_pipeline as create_monthly_model_selection_pipeline,
)
from hdf_pipelines.pipelines.model_selection.prophet.pipeline import (
    create_pipeline as create_prophet_monthly_selection_pipeline,
)
from hdf_pipelines.pipelines.train_monthly.prophet.pipeline import (
    create_pipeline as create_prophet_monthly_pipeline,
)
from hdf_pipelines.pipelines.train_monthly.sarimax.pipeline import (
    create_pipeline as create_sarimax_monthly_pipeline,
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
    prophet_monthly_training = create_prophet_monthly_pipeline()

    # Standalone SARIMAX-only training pipeline
    sarimax_monthly_training = create_sarimax_monthly_pipeline()

    # Standalone Monthly Prophet model-selection pipeline (Prophet-specific champion)
    prophet_monthly_selection = create_prophet_monthly_selection_pipeline()

    # Monthly multi-family model selection: Prophet vs SARIMAX (Phase 5)
    monthly_model_selection = create_monthly_model_selection_pipeline()

    # End-to-end Monthly Prophet MVP: ingestion → features → splits → train → select → infer
    prophet_monthly_e2e = (
        ingestion
        + fe_monthly
        + model_input
        + prophet_monthly_training
        + prophet_monthly_selection
        + inference
    )

    # End-to-end Prophet + SARIMAX comparison:
    # ingestion → features → splits → train both → compare → generic champion
    prophet_sarimax_comparison = (
        ingestion
        + fe_monthly
        + model_input
        + prophet_monthly_training
        + sarimax_monthly_training
        + monthly_model_selection
    )

    # Validated monthly MVP — the current stable default execution route
    monthly_mvp = prophet_monthly_e2e

    # Scaffolded composed shortcuts — include NotImplementedError stubs; not part of default
    experimental_training = monthly_training + weekly_training
    experimental_full_experiment = (
        ingestion + fe_monthly + fe_weekly + model_input + experimental_training + selection
    )
    experimental_inference = inference + recon

    return {
        # ── Validated routes ──────────────────────────────────────────────────
        "__default__": monthly_mvp,
        "monthly_mvp": monthly_mvp,
        "prophet_monthly_e2e": prophet_monthly_e2e,
        # ── Phase 5: multi-family monthly model selection ─────────────────────
        "monthly_model_selection": monthly_model_selection,
        "prophet_sarimax_comparison": prophet_sarimax_comparison,
        # ── Individual stage pipelines ────────────────────────────────────────
        "data_ingestion": ingestion,
        "feature_engineering_monthly": fe_monthly,
        "model_input_preparation": model_input,
        "train_monthly": monthly_training,
        "model_selection": selection,
        "forecast_inference": inference,
        # ── Scaffolded / experimental (include NotImplementedError stubs) ─────
        "feature_engineering_weekly": fe_weekly,
        "train_weekly": weekly_training,
        "reconciliation": recon,
        "experimental_training": experimental_training,
        "experimental_inference": experimental_inference,
        "experimental_full_experiment": experimental_full_experiment,
    }
