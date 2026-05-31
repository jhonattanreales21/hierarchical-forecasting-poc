"""Monthly model-input preparation pipeline.

Flow:
  1. build_monthly_modeling_data       → monthly_modeling_data  (generic, month_start_date/monthly_demand)
  2. split_monthly_modeling_data       → monthly_train/validation/test/full_train  (generic)
  3. build_monthly_split_metadata      → monthly_split_metadata  (generic)
  4. adapt_monthly_data_for_prophet    → monthly_prophet_modeling_data/train/validation/test/full_train
  5. build_monthly_prophet_future_regressors → monthly_prophet_future_3m/6m/12m
  6. build_monthly_prophet_split_metadata   → monthly_prophet_split_metadata
  7. adapt_monthly_data_for_sarimax    → monthly_sarimax_train/validation/test/full_train/split_metadata
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    adapt_monthly_data_for_prophet,
    adapt_monthly_data_for_sarimax,
    build_monthly_modeling_data,
    build_monthly_prophet_future_regressors,
    build_monthly_prophet_split_metadata,
    build_monthly_split_metadata,
    split_monthly_modeling_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monthly model-input preparation pipeline."""
    return pipeline(
        [
            # ── Step 1: Generic monthly modeling data ─────────────────────────
            node(
                func=build_monthly_modeling_data,
                inputs=[
                    "monthly_prophet_features",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_modeling_data",
                    "monthly_preparation_metadata",
                ],
                name="build_monthly_modeling_data",
            ),
            # ── Step 2: Generic temporal splits ───────────────────────────────
            node(
                func=split_monthly_modeling_data,
                inputs=[
                    "monthly_modeling_data",
                    "monthly_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_train",
                    "monthly_validation",
                    "monthly_test",
                    "monthly_full_train",
                    "monthly_split_preparation_metadata",
                ],
                name="split_monthly_modeling_data",
            ),
            # ── Step 3: Generic split metadata ────────────────────────────────
            node(
                func=build_monthly_split_metadata,
                inputs=[
                    "monthly_train",
                    "monthly_validation",
                    "monthly_test",
                    "monthly_full_train",
                    "monthly_split_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs="monthly_split_metadata",
                name="build_monthly_split_metadata",
            ),
            # ── Step 4: Prophet compatibility adapter ─────────────────────────
            node(
                func=adapt_monthly_data_for_prophet,
                inputs=[
                    "monthly_modeling_data",
                    "monthly_train",
                    "monthly_validation",
                    "monthly_test",
                    "monthly_full_train",
                    "monthly_split_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_test",
                    "monthly_prophet_full_train",
                    "monthly_prophet_adapter_metadata",
                ],
                name="adapt_monthly_data_for_prophet",
            ),
            # ── Step 5: Prophet future regressors ─────────────────────────────
            node(
                func=build_monthly_prophet_future_regressors,
                inputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_calendar_features",
                    "monthly_exogenous_features",
                    "params:model_input_preparation",
                    "params:feature_engineering_monthly",
                ],
                outputs=[
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                ],
                name="build_monthly_prophet_future_regressors",
            ),
            # ── Step 6: Prophet split metadata ────────────────────────────────
            node(
                func=build_monthly_prophet_split_metadata,
                inputs=[
                    "monthly_prophet_train",
                    "monthly_prophet_validation",
                    "monthly_prophet_test",
                    "monthly_prophet_full_train",
                    "monthly_prophet_future_3m",
                    "monthly_prophet_future_6m",
                    "monthly_prophet_future_12m",
                    "monthly_prophet_adapter_metadata",
                    "params:model_input_preparation",
                ],
                outputs="monthly_prophet_split_metadata",
                name="build_monthly_prophet_split_metadata",
            ),
            # ── Step 7: SARIMAX adapter ───────────────────────────────────────
            node(
                func=adapt_monthly_data_for_sarimax,
                inputs=[
                    "monthly_train",
                    "monthly_validation",
                    "monthly_test",
                    "monthly_full_train",
                    "monthly_split_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_sarimax_train",
                    "monthly_sarimax_validation",
                    "monthly_sarimax_test",
                    "monthly_sarimax_full_train",
                    "monthly_sarimax_split_metadata",
                ],
                name="adapt_monthly_data_for_sarimax",
            ),
        ]
    )
