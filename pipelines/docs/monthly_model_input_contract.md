# Monthly Model Input Contract

## Overview

The `model_input_preparation` pipeline produces model-family-specific input datasets from the generic monthly modeling layer. Each adapter transforms the canonical generic splits into the format expected by a specific model family.

---

## Generic Monthly Layer (Phase 2)

All model adapters consume these upstream datasets:

| Dataset | Format | Description |
|---------|--------|-------------|
| `monthly_modeling_data` | Parquet | Full cleaned monthly table with canonical column names |
| `monthly_train` | Parquet | Training split |
| `monthly_validation` | Parquet | Validation split |
| `monthly_test` | Parquet | Test split |
| `monthly_full_train` | Parquet | All splits combined for final refit |
| `monthly_split_metadata` | JSON | Split boundaries, row counts, active features |

**Canonical column names:**
- Date: `month_start_date` (datetime, month-start)
- Target: `monthly_demand` (float)
- SKU: `sku` (string)
- Active regressors: configured via `model_input_preparation.monthly.active_regressors`

---

## Prophet Adapter (Phase 2)

Renames generic columns to Prophet's `ds`/`y` convention. All other columns are preserved.

| Dataset | Key columns |
|---------|-------------|
| `monthly_prophet_train` | `ds`, `y`, `sku`, `*active_regressors` |
| `monthly_prophet_validation` | same |
| `monthly_prophet_test` | same |
| `monthly_prophet_full_train` | same |
| `monthly_prophet_future_3m` | `ds`, `sku`, `*active_regressors` (no `y`) |
| `monthly_prophet_future_6m` | same |
| `monthly_prophet_future_12m` | same |
| `monthly_prophet_split_metadata` | JSON |

These datasets are **stable and must not be modified** by any downstream adapter.

---

## SARIMAX Adapter (Phase 3)

Transforms the generic monthly splits into a tabular structure suited for SARIMAX estimation. The adapter **does not rename** date or target columns — it retains canonical names and separates target from exogenous features.

### Output datasets

| Dataset | Format | Description |
|---------|--------|-------------|
| `monthly_sarimax_train` | Parquet | SARIMAX-ready train split |
| `monthly_sarimax_validation` | Parquet | SARIMAX-ready validation split |
| `monthly_sarimax_test` | Parquet | SARIMAX-ready test split |
| `monthly_sarimax_full_train` | Parquet | SARIMAX-ready full-train split |
| `monthly_sarimax_split_metadata` | JSON | SARIMAX metadata with split diagnostics |

### Output schema (tabular contract)

Each Parquet file contains:

```
month_start_date   datetime[ns]   — monthly date index (month start, MS frequency)
monthly_demand     float64        — target series
<exogenous_cols>   float64        — configured exogenous features (empty by default)
```

`sku` is excluded from the output DataFrame; it is preserved as metadata only. Future SARIMAX training nodes reconstruct `y = df["monthly_demand"]` and `exog = df[exogenous_columns]` directly.

### Configuration (`conf/base/parameters/model_input.yml`)

```yaml
model_input_preparation:
  monthly_sarimax:
    enabled: true
    date_column: "month_start_date"
    target_column: "monthly_demand"
    sku_column: "sku"
    frequency: "MS"
    exogenous_columns: []          # add column names here to include exogenous features
    allow_empty_exog: true
    require_regular_frequency: true
    sort_by_date: true
    drop_rows_with_null_target: true
    drop_rows_with_null_exog: false
    impute_exog: false
    output_format: "tabular"
```

### Validation behaviour

| Check | Behaviour |
|-------|-----------|
| Missing date column | `ValueError` with column name |
| Missing target column | `ValueError` with column name |
| Missing configured exogenous column | `ValueError` with column name |
| Duplicate dates | `ValueError` — expected single-SKU monthly series |
| Missing monthly periods | `ValueError` if `require_regular_frequency: true`; recorded in metadata otherwise |
| Null target rows | Dropped if `drop_rows_with_null_target: true`; `ValueError` otherwise |
| Null exogenous rows | `ValueError` unless `drop_rows_with_null_exog: true` or `impute_exog: true` |
| Target listed in exogenous_columns | De-duplicated silently; `monthly_demand` appears once |

### Metadata contract (`monthly_sarimax_split_metadata`)

```json
{
  "granularity": "monthly",
  "model_family": "sarimax",
  "date_column": "month_start_date",
  "target_column": "monthly_demand",
  "sku_column": "sku",
  "frequency": "MS",
  "exogenous_columns": [],
  "allow_empty_exog": true,
  "splits": {
    "train":      {"rows": N, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing_periods": [], "null_target_rows_dropped": 0},
    "validation": {"rows": N, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing_periods": [], "null_target_rows_dropped": 0},
    "test":       {"rows": N, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing_periods": [], "null_target_rows_dropped": 0},
    "full_train": {"rows": N, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing_periods": [], "null_target_rows_dropped": 0}
  },
  "source_metadata": {"from": "monthly_split_metadata"},
  "created_by": "model_input_preparation.sarimax_adapter"
}
```

### Deferred: SARIMAX training (Phase 4)

This adapter does **not** implement SARIMAX training, tuning, or inference. Future training nodes will consume `monthly_sarimax_train`, `monthly_sarimax_validation`, etc. and reconstruct `y` and `exog` as:

```python
y    = split_df.set_index("month_start_date")["monthly_demand"]
exog = split_df.set_index("month_start_date")[exogenous_columns] or None
```

---

## Pipeline Flow

```
monthly_prophet_features
  → build_monthly_modeling_data       → monthly_modeling_data
  → split_monthly_modeling_data       → monthly_train / validation / test / full_train
  → build_monthly_split_metadata      → monthly_split_metadata
  → adapt_monthly_data_for_prophet    → monthly_prophet_* datasets
  → build_monthly_prophet_future_regressors → monthly_prophet_future_*
  → build_monthly_prophet_split_metadata    → monthly_prophet_split_metadata
  → adapt_monthly_data_for_sarimax    → monthly_sarimax_* datasets
```

SARIMAX adapter consumes **generic** `monthly_*` datasets. It never reads `monthly_prophet_*` datasets.
