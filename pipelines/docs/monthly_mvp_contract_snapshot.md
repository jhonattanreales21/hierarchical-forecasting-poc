# Monthly MVP — Contract Snapshot

**Generated:** 2026-05-31  
**Pipeline:** `monthly_mvp`  
**Branch:** `refactor/phase-0-safety-baseline`

---

## Tabular Datasets

### `monthly_prophet_features` — *feature*

- **Shape:** 38 rows × 37 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `month_start_date`: datetime64[ns]
  - `sku`: object
  - `monthly_demand`: float64
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `market_share_stress`: float64
  - `market_share_uplift`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `expected_market_share_lag_2`: float64
  - `market_share_stress_lag_2`: float64
  - `market_share_uplift_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `expected_market_share_lag_3`: float64
  - `market_share_stress_lag_3`: float64
  - `market_share_uplift_lag_3`: float64
- **Null counts (non-zero only):**
  - pfizer_limited_lag_3: 1
  - surgifoam_limited_lag_3: 1
  - rebate_target_lag_3: 1
  - expected_market_share_lag_3: 1
  - market_share_stress_lag_3: 1
  - market_share_uplift_lag_3: 1

### `monthly_prophet_modeling_data` — *model_input*

- **Shape:** 37 rows × 31 columns
- **Date column:** `ds`
- **Date range:** 2023-04-01 → 2026-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `y`: float64
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_train` — *model_input*

- **Shape:** 31 rows × 31 columns
- **Date column:** `ds`
- **Date range:** 2023-04-01 → 2025-10-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `y`: float64
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_validation` — *model_input*

- **Shape:** 3 rows × 31 columns
- **Date column:** `ds`
- **Date range:** 2025-11-01 → 2026-01-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `y`: float64
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_test` — *model_input*

- **Shape:** 3 rows × 31 columns
- **Date column:** `ds`
- **Date range:** 2026-02-01 → 2026-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `y`: float64
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_full_train` — *model_input*

- **Shape:** 37 rows × 31 columns
- **Date column:** `ds`
- **Date range:** 2023-04-01 → 2026-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `y`: float64
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_future_3m` — *model_input*

- **Shape:** 3 rows × 30 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2026-07-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_future_6m` — *model_input*

- **Shape:** 6 rows × 30 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2026-10-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_future_12m` — *model_input*

- **Shape:** 12 rows × 30 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2027-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `business_days`: int64
  - `total_tuesdays`: int64
  - `total_thursdays`: int64
  - `working_tuesdays`: int64
  - `working_thursdays`: int64
  - `has_5_working_tuesdays`: int64
  - `has_5_working_thursdays`: int64
  - `tuesday_holidays`: int64
  - `thursday_holidays`: int64
  - `total_holidays`: int64
  - `pfizer_limited`: float64
  - `surgifoam_limited`: float64
  - `rebate_target`: float64
  - `expected_market_share`: float64
  - `pfizer_limited_lag_1`: float64
  - `surgifoam_limited_lag_1`: float64
  - `rebate_target_lag_1`: float64
  - `expected_market_share_lag_1`: float64
  - `pfizer_limited_lag_2`: float64
  - `surgifoam_limited_lag_2`: float64
  - `rebate_target_lag_2`: float64
  - `pfizer_limited_lag_3`: float64
  - `surgifoam_limited_lag_3`: float64
  - `rebate_target_lag_3`: float64
  - `market_share_stress`: float64
  - `market_share_stress_lag_1`: float64
  - `market_share_uplift`: float64
  - `market_share_uplift_lag_1`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_tuning_results` — *training*

- **Shape:** 30 rows × 36 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `trial_number`: int64
  - `status`: object
  - `error_message`: object
  - `optimizer`: object
  - `objective_direction`: object
  - `selection_metric`: object
  - `selection_metric_value`: float64
  - `wape`: float64
  - `mase`: float64
  - `mae`: float64
  - `rmse`: float64
  - `mape`: float64
  - `wmape`: float64
  - `forecast_precision`: float64
  - `business_success_flag`: bool
  - `horizon_2_mae`: float64
  - `horizon_2_mape`: float64
  - `horizon_2_wmape`: float64
  - `horizon_2_forecast_precision`: float64
  - `horizon_3_mae`: float64
  - `horizon_3_mape`: float64
  - `horizon_3_wmape`: float64
  - `horizon_3_forecast_precision`: float64
  - `changepoint_prior_scale`: float64
  - `seasonality_prior_scale`: float64
  - `holidays_prior_scale`: float64
  - `seasonality_mode`: object
  - `yearly_seasonality`: bool
  - `weekly_seasonality`: bool
  - `daily_seasonality`: bool
  - `interval_width`: float64
  - `active_regressors`: object
  - `trained_at`: object
  - `rank`: int64
  - `is_prechampion`: bool
- **Null counts (non-zero only):**
  - error_message: 30

### `monthly_prophet_validation_metrics` — *training*

- **Shape:** 30 rows × 21 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `trial_number`: int64
  - `wape`: float64
  - `mase`: float64
  - `mae`: float64
  - `rmse`: float64
  - `mape`: float64
  - `wmape`: float64
  - `forecast_precision`: float64
  - `business_success_flag`: bool
  - `horizon_2_mae`: float64
  - `horizon_2_mape`: float64
  - `horizon_2_wmape`: float64
  - `horizon_2_forecast_precision`: float64
  - `horizon_3_mae`: float64
  - `horizon_3_mape`: float64
  - `horizon_3_wmape`: float64
  - `horizon_3_forecast_precision`: float64
  - `validation_start_date`: object
  - `validation_end_date`: object
  - `validation_rows`: int64
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_test_metrics` — *selection*

- **Shape:** 3 rows × 22 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `status`: object
  - `error_message`: object
  - `wape`: float64
  - `mase`: float64
  - `mae`: float64
  - `rmse`: float64
  - `mape`: float64
  - `wmape`: float64
  - `forecast_precision`: float64
  - `business_success_flag`: bool
  - `horizon_2_mae`: float64
  - `horizon_2_mape`: float64
  - `horizon_2_wmape`: float64
  - `horizon_2_forecast_precision`: float64
  - `horizon_3_mae`: float64
  - `horizon_3_mape`: float64
  - `horizon_3_wmape`: float64
  - `horizon_3_forecast_precision`: float64
  - `test_start_date`: object
  - `test_end_date`: object
  - `test_rows`: int64
- **Null counts (non-zero only):**
  - error_message: 3

### `monthly_prophet_model_selection_summary` — *selection*

- **Shape:** 3 rows × 17 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `validation_rank`: int64
  - `test_rank`: int64
  - `is_champion`: bool
  - `primary_metric`: object
  - `primary_metric_value`: float64
  - `wape`: float64
  - `mase`: float64
  - `mae`: float64
  - `rmse`: float64
  - `mape`: float64
  - `wmape`: float64
  - `test_m2_wape`: float64
  - `test_m3_wape`: float64
  - `forecast_precision`: float64
  - `business_success_flag`: bool
  - `selection_reason`: object
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_champion_test_forecast` — *selection*

- **Shape:** 9 rows × 9 columns
- **Date column:** `ds`
- **Date range:** 2026-02-01 → 2026-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `y`: float64
  - `yhat`: float64
  - `yhat_lower`: float64
  - `yhat_upper`: float64
  - `candidate_id`: object
  - `is_champion`: bool
  - `dataset`: object
- **Null counts (non-zero only):**
  (none)

### `monthly_operational_test_forecasts` — *reporting*

- **Shape:** 9 rows × 9 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `model_family`: object
  - `candidate_id`: object
  - `split`: object
  - `origin_month`: datetime64[ns]
  - `target_month`: datetime64[ns]
  - `lead_time`: int64
  - `training_cutoff`: datetime64[ns]
  - `y_true`: float64
  - `y_pred`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_operational_lead_time_metrics` — *reporting*

- **Shape:** 3 rows × 5 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `n_m2_pairs`: int64
  - `test_m2_wape`: float64
  - `n_m3_pairs`: int64
  - `test_m3_wape`: float64
- **Null counts (non-zero only):**
  (none)

### `monthly_model_selection_audit` — *reporting*

- **Shape:** 3 rows × 16 columns
- **Date column:** `not detected`
- **Date range:** N/A → N/A
- **Columns and dtypes:**
  - `candidate_id`: object
  - `is_champion`: bool
  - `test_rank`: int64
  - `validation_rank`: int64
  - `primary_metric`: object
  - `wape`: float64
  - `mase`: float64
  - `rmse`: float64
  - `mape`: float64
  - `test_m2_wape`: float64
  - `test_m3_wape`: float64
  - `n_m2_pairs`: object
  - `n_m3_pairs`: object
  - `missing_operational_metrics`: object
  - `selection_reason`: object
  - `audited_at`: object
- **Null counts (non-zero only):**
  - n_m2_pairs: 3
  - n_m3_pairs: 3
  - missing_operational_metrics: 3

### `monthly_prophet_forecast_3m` — *inference*

- **Shape:** 3 rows × 16 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2026-07-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `horizon_month`: int64
  - `yhat`: float64
  - `yhat_lower`: float64
  - `yhat_upper`: float64
  - `model_family`: object
  - `model_granularity`: object
  - `champion_id`: object
  - `forecast_run_id`: object
  - `forecast_created_at`: object
  - `forecast_horizon_months`: int64
  - `selection_metric`: object
  - `selection_metric_value`: float64
  - `business_success_flag`: bool
  - `source_dataset`: object
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_forecast_6m` — *inference*

- **Shape:** 6 rows × 16 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2026-10-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `horizon_month`: int64
  - `yhat`: float64
  - `yhat_lower`: float64
  - `yhat_upper`: float64
  - `model_family`: object
  - `model_granularity`: object
  - `champion_id`: object
  - `forecast_run_id`: object
  - `forecast_created_at`: object
  - `forecast_horizon_months`: int64
  - `selection_metric`: object
  - `selection_metric_value`: float64
  - `business_success_flag`: bool
  - `source_dataset`: object
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_forecast_12m` — *inference*

- **Shape:** 12 rows × 16 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2027-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `horizon_month`: int64
  - `yhat`: float64
  - `yhat_lower`: float64
  - `yhat_upper`: float64
  - `model_family`: object
  - `model_granularity`: object
  - `champion_id`: object
  - `forecast_run_id`: object
  - `forecast_created_at`: object
  - `forecast_horizon_months`: int64
  - `selection_metric`: object
  - `selection_metric_value`: float64
  - `business_success_flag`: bool
  - `source_dataset`: object
- **Null counts (non-zero only):**
  (none)

### `monthly_prophet_forecast_latest` — *inference*

- **Shape:** 12 rows × 16 columns
- **Date column:** `ds`
- **Date range:** 2026-05-01 → 2027-04-01
- **Columns and dtypes:**
  - `ds`: datetime64[ns]
  - `sku`: object
  - `horizon_month`: int64
  - `yhat`: float64
  - `yhat_lower`: float64
  - `yhat_upper`: float64
  - `model_family`: object
  - `model_granularity`: object
  - `champion_id`: object
  - `forecast_run_id`: object
  - `forecast_created_at`: object
  - `forecast_horizon_months`: int64
  - `selection_metric`: object
  - `selection_metric_value`: float64
  - `business_success_flag`: bool
  - `source_dataset`: object
- **Null counts (non-zero only):**
  (none)

## JSON / Metadata Artifacts

### `monthly_prophet_split_metadata` — *model_input*

- **Type:** `dict`
- **Top-level keys:** ['model_family', 'granularity', 'split_mode', 'active_regressors', 'train', 'validation', 'test', 'full_train', 'future_horizons', 'dropped_rows']
- **Sample (truncated):**

```json
{
  "model_family": "prophet",
  "granularity": "monthly",
  "split_mode": "months",
  "active_regressors": [
    "business_days",
    "total_tuesdays",
    "total_thursdays",
    "working_tuesdays",
    "working_thursdays",
    "has_5_working_tuesdays",
    "has_5_working_thursdays",
    "tuesday_holidays",
    "thursday_holidays",
    "total_holidays",
    "pfizer_limited",
    "surgifoam_limited",
    "rebate_target",
    "expected_market_share",
    "pfizer_limited_lag_1",
    "surgifoam_limited_lag_1",
    "rebate_target_lag_1",
    "expected_market_share_lag_1",
    "pfizer_limited_lag_2",
    "surgifoam_limited_lag_2",
    "rebate_target_lag_2",
    "pfizer_limited_lag_3",
    "surgifoam_limited_lag_3",
    "rebate_target_lag_3",
    "market_share_stress",
    "market_share_stress_l
```

### `monthly_prophet_prechampion_configs` — *training*

- **Type:** `dict`
- **Top-level keys:** ['model_family', 'granularity', 'selection_metric', 'top_n_prechampions', 'prechampions']
- **Sample (truncated):**

```json
{
  "model_family": "prophet",
  "granularity": "monthly",
  "selection_metric": "wape",
  "top_n_prechampions": 3,
  "prechampions": [
    {
      "candidate_id": "prophet_candidate_004",
      "rank": 1,
      "validation_metrics": {
        "wape": 0.40700625849098565,
        "mase": 1.4361186986037149,
        "mae": 2.6129801795121277,
        "rmse": 2.6987810245331616,
        "mape": 0.3702100435891407,
        "wmape": 0.38691710456744244,
        "forecast_precision": 0.6297899564108593,
        "horizon_2_mape": 0.2532285254528911,
        "horizon_3_mape": 0.5728321324099105
      },
      "model_params": {
        "changepoint_prior_scale": 0.020492680115417352,
        "seasonality_prior_scale": 2.0148477884158655,
        "holidays_prior_scale": 3.347776308515933,
        "
```

### `monthly_prophet_training_metadata` — *training*

- **Type:** `dict`
- **Top-level keys:** ['model_family', 'granularity', 'optimizer', 'objective', 'sampler', 'max_trials', 'top_n_prechampions', 'completed_trials', 'failed_trials', 'best_trial_number', 'best_candidate_id', 'best_value', 'fixed_params', 'trials']
- **Sample (truncated):**

```json
{
  "model_family": "prophet",
  "granularity": "monthly",
  "optimizer": "optuna",
  "objective": {
    "metric": "wape",
    "direction": "minimize"
  },
  "sampler": {
    "name": "tpe",
    "seed": 42
  },
  "max_trials": 30,
  "top_n_prechampions": 3,
  "completed_trials": 30,
  "failed_trials": 0,
  "best_trial_number": 3,
  "best_candidate_id": "prophet_candidate_004",
  "best_value": 0.40700625849098565,
  "fixed_params": {
    "yearly_seasonality": true,
    "weekly_seasonality": false,
    "daily_seasonality": false,
    "interval_width": 0.8
  },
  "trials": [
    {
      "trial_number": 0,
      "state": "complete",
      "value": 0.9972558517872033,
      "params": {
        "changepoint_prior_scale": 0.043284502212938815,
        "seasonality_prior_scale": 8.927180304353628,

```

### `monthly_prophet_champion_metadata` — *selection*

- **Type:** `dict`
- **Top-level keys:** ['model_family', 'granularity', 'champion_id', 'selection_stage', 'selection_metric', 'selection_metric_value', 'business_success_precision_threshold', 'business_success_flag', 'refit_on_full_train', 'selected_at', 'active_regressors', 'model_params', 'validation_metrics', 'test_metrics', 'operational_test_metrics', 'optuna_best_trial', 'train_window', 'test_window']
- **Sample (truncated):**

```json
{
  "model_family": "prophet",
  "granularity": "monthly",
  "champion_id": "prophet_candidate_007",
  "selection_stage": "model_selection",
  "selection_metric": "wape",
  "selection_metric_value": 0.2230848985088411,
  "business_success_precision_threshold": 0.85,
  "business_success_flag": false,
  "refit_on_full_train": true,
  "selected_at": "2026-05-31T07:29:02.388898+00:00",
  "active_regressors": [
    "business_days",
    "total_tuesdays",
    "total_thursdays",
    "working_tuesdays",
    "working_thursdays",
    "has_5_working_tuesdays",
    "has_5_working_thursdays",
    "tuesday_holidays",
    "thursday_holidays",
    "total_holidays",
    "pfizer_limited",
    "surgifoam_limited",
    "rebate_target",
    "expected_market_share",
    "pfizer_limited_lag_1",
    "surgifoam_lim
```

### `monthly_prophet_inference_metadata` — *inference*

- **Type:** `dict`
- **Top-level keys:** ['model_family', 'model_granularity', 'champion_id', 'forecast_run_id', 'forecast_created_at', 'active_regressors', 'horizons', 'latest_output', 'selection']
- **Sample (truncated):**

```json
{
  "model_family": "prophet",
  "model_granularity": "monthly",
  "champion_id": "prophet_candidate_007",
  "forecast_run_id": "monthly_prophet_20260531_072916_30bc1d",
  "forecast_created_at": "2026-05-31T07:29:16.362534+00:00",
  "active_regressors": [
    "business_days",
    "total_tuesdays",
    "total_thursdays",
    "working_tuesdays",
    "working_thursdays",
    "has_5_working_tuesdays",
    "has_5_working_thursdays",
    "tuesday_holidays",
    "thursday_holidays",
    "total_holidays",
    "pfizer_limited",
    "surgifoam_limited",
    "rebate_target",
    "expected_market_share",
    "pfizer_limited_lag_1",
    "surgifoam_limited_lag_1",
    "rebate_target_lag_1",
    "expected_market_share_lag_1",
    "pfizer_limited_lag_2",
    "surgifoam_limited_lag_2",
    "rebate_target_l
```

## Model Artifacts (Pickle)

### `monthly_prophet_candidate_models` — *training*

- **Type:** Pickle artifact
- **Purpose:** See catalog.yml for physical path and expected object type.

### `candidate_monthly_prophet` — *training*

- **Type:** Pickle artifact
- **Purpose:** See catalog.yml for physical path and expected object type.

### `monthly_prophet_champion_model` — *selection*

- **Type:** Pickle artifact
- **Purpose:** See catalog.yml for physical path and expected object type.

