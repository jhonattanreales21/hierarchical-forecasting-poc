"""Monthly model-input preparation pipeline.

The fixed train/validation/test hold-out is removed. The series is kept whole
(full history) and the rolling-origin engine slices cycles internally at train
time.

Flow:
  1. build_monthly_modeling_data            → monthly_modeling_data  (generic, month_start_date/monthly_demand)
  2. prepare_monthly_full_history           → monthly_full_train  (generic full history)
  3. build_monthly_rolling_origin_windows   → monthly_rolling_origin_windows  (audit artifact)
  4. build_monthly_split_metadata           → monthly_split_metadata  (generic)
  5. adapt_monthly_data_for_prophet         → monthly_prophet_modeling_data/full_train
  6. build_monthly_prophet_future_regressors → monthly_prophet_future_3m/6m/12m
  7. build_monthly_prophet_split_metadata   → monthly_prophet_split_metadata
  8. adapt_monthly_data_for_sarimax         → monthly_sarimax_full_train/split_metadata
  9. build_monthly_generic_future_frames    → monthly_future_3m/6m/12m  (canonical inference inputs)
 10. adapt_monthly_data_for_catboost        → monthly_catboost_full_train/split_metadata  (direct multi-horizon)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    adapt_monthly_data_for_catboost,
    adapt_monthly_data_for_prophet,
    adapt_monthly_data_for_sarimax,
    build_monthly_generic_future_frames,
    build_monthly_modeling_data,
    build_monthly_prophet_future_regressors,
    build_monthly_prophet_split_metadata,
    build_monthly_rolling_origin_windows,
    build_monthly_split_metadata,
    prepare_monthly_full_history,
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
            # ── Step 2: Generic full history (full history) ──────────────────────
            node(
                func=prepare_monthly_full_history,
                inputs=[
                    "monthly_modeling_data",
                    "monthly_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_full_train",
                    "monthly_split_preparation_metadata",
                ],
                name="prepare_monthly_full_history",
            ),
            # ── Step 3: Rolling-origin window specification (audit artifact) ──
            node(
                func=build_monthly_rolling_origin_windows,
                inputs=[
                    "monthly_full_train",
                    "params:model_input_preparation",
                ],
                outputs="monthly_rolling_origin_windows",
                name="build_monthly_rolling_origin_windows",
            ),
            # ── Step 4: Generic full-history metadata ─────────────────────────
            node(
                func=build_monthly_split_metadata,
                inputs=[
                    "monthly_full_train",
                    "monthly_split_preparation_metadata",
                    "params:model_input_preparation",
                ],
                outputs="monthly_split_metadata",
                name="build_monthly_split_metadata",
            ),
            # ── Step 5: Prophet compatibility adapter ─────────────────────────
            node(
                func=adapt_monthly_data_for_prophet,
                inputs=[
                    "monthly_modeling_data",
                    "monthly_full_train",
                    "monthly_split_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_prophet_modeling_data",
                    "monthly_prophet_full_train",
                    "monthly_prophet_adapter_metadata",
                ],
                name="adapt_monthly_data_for_prophet",
            ),
            # ── Step 6: Prophet future regressors ─────────────────────────────
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
            # ── Step 7: Prophet split metadata ────────────────────────────────
            node(
                func=build_monthly_prophet_split_metadata,
                inputs=[
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
            # ── Step 8: SARIMAX adapter ───────────────────────────────────────
            node(
                func=adapt_monthly_data_for_sarimax,
                inputs=[
                    "monthly_full_train",
                    "monthly_split_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_sarimax_full_train",
                    "monthly_sarimax_split_metadata",
                ],
                name="adapt_monthly_data_for_sarimax",
            ),
            # ── Step 9: Generic future frames (canonical champion inference inputs) ──
            node(
                func=build_monthly_generic_future_frames,
                inputs=[
                    "monthly_modeling_data",
                    "monthly_calendar_features",
                    "monthly_exogenous_features",
                    "params:model_input_preparation",
                    "params:feature_engineering_monthly",
                ],
                outputs=[
                    "monthly_future_3m",
                    "monthly_future_6m",
                    "monthly_future_12m",
                ],
                name="build_monthly_generic_future_frames",
            ),
            # ── Step 10: CatBoost adapter (direct multi-horizon) ───────────
            node(
                func=adapt_monthly_data_for_catboost,
                inputs=[
                    "monthly_full_train",
                    "monthly_split_metadata",
                    "params:model_input_preparation",
                ],
                outputs=[
                    "monthly_catboost_full_train",
                    "monthly_catboost_split_metadata",
                ],
                name="adapt_monthly_data_for_catboost",
            ),
        ]
    )
