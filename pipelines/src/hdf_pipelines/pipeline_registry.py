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
from hdf_pipelines.pipelines.train_monthly.catboost.pipeline import (
    create_pipeline as create_catboost_monthly_pipeline,
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

    # Standalone CatBoost-only training pipeline
    catboost_monthly_training = create_catboost_monthly_pipeline()

    # Standalone Monthly Prophet model-selection pipeline (Prophet-specific champion)
    prophet_monthly_selection = create_prophet_monthly_selection_pipeline()

    # Monthly multi-family model selection: Prophet vs SARIMAX vs CatBoost
    monthly_model_selection = create_monthly_model_selection_pipeline()

    # Prophet-only training + Prophet-specific selection. Kept as an isolated
    # single-family reference route; it produces the Prophet-specific champion
    # artifacts and does not feed the generic metadata-driven inference.
    prophet_monthly_e2e = (
        ingestion + fe_monthly + model_input + prophet_monthly_training
    )

    # Single-family monthly routes for SARIMAX and CatBoost: raw → features →
    # splits → train that family's candidates. These mirror prophet_monthly_e2e
    # but stop after training, since only Prophet currently has a single-family
    # selection stage. Exposed as the per-model shortcuts in the Makefile.
    sarimax_monthly_e2e = (
        ingestion + fe_monthly + model_input + sarimax_monthly_training
    )
    catboost_monthly_e2e = (
        ingestion + fe_monthly + model_input + catboost_monthly_training
    )

    # Legacy two-family comparison kept for backward compatibility: trains only
    # Prophet + SARIMAX before selection. Superseded by monthly_training_comparison.
    prophet_sarimax_comparison = (
        ingestion
        + fe_monthly
        + model_input
        + prophet_monthly_training
        + sarimax_monthly_training
        + monthly_model_selection
    )

    # Active monthly training comparison: train all monthly families
    # (Prophet + SARIMAX + CatBoost) before selection.
    # ingestion → features → splits → train Prophet + SARIMAX + CatBoost → compare.
    # monthly_model_selection elects the production champion across all three
    # families; CatBoost competes on the same leakage-safe held-out test metrics.
    monthly_training_comparison = (
        ingestion
        + fe_monthly
        + model_input
        + prophet_monthly_training
        + sarimax_monthly_training
        + catboost_monthly_training
        + monthly_model_selection
    )

    # Canonical reproducible monthly route: multi-family comparison followed by
    # metadata-driven champion inference. This is the project default.
    monthly_forecast_e2e = monthly_training_comparison + inference

    # Prophet-only validated reference route (training + Prophet selection).
    monthly_mvp = prophet_monthly_e2e

    # Scaffolded composed shortcuts — include NotImplementedError stubs; not part of default
    experimental_training = monthly_training + weekly_training
    experimental_full_experiment = (
        ingestion
        + fe_monthly
        + fe_weekly
        + model_input
        + experimental_training
        + selection
    )
    experimental_inference = inference + recon

    return {
        # ── Default reproducible route ────────────────────────────────────────
        "__default__": monthly_forecast_e2e,
        "monthly_forecast_e2e": monthly_forecast_e2e,
        # ── Multi-family monthly comparison + selection ───────────────────────
        "monthly_model_selection": monthly_model_selection,
        "monthly_training_comparison": monthly_training_comparison,
        # Legacy alias: Prophet + SARIMAX only (no CatBoost training)
        "prophet_sarimax_comparison": prophet_sarimax_comparison,
        # ── Single-family monthly routes (per-model Makefile shortcuts) ───────
        "monthly_mvp": monthly_mvp,
        "prophet_monthly_e2e": prophet_monthly_e2e,
        "sarimax_monthly_e2e": sarimax_monthly_e2e,
        "catboost_monthly_e2e": catboost_monthly_e2e,
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
