# hdf-pipelines

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

`hdf_pipelines` is the Kedro package for the **Hierarchical Demand Forecasting
PoC**. It owns the reproducible data and ML workflow for a single critical SKU:
ingestion, temporal aggregation, monthly feature engineering, model input
preparation, monthly model training, model selection, champion inference, and
the scaffolded weekly/reconciliation layers.

The project hierarchy is temporal, not product-, category-, or location-based:

1. Monthly planning layer - primary analytical and business-facing layer.
2. Weekly operational layer - secondary enhancement.
3. Daily layer - optional exploratory allocation/disaggregation only.

The pipeline layer is the source of truth for forecasting logic. Streamlit and
FastAPI consume curated outputs and may trigger workflows, but they must not
contain core validation, aggregation, training, model selection, reconciliation,
or inference logic.

## North Star

The goal is to compare **~4 model families** (Prophet, CatBoost, SARIMAX, and optionally N-HiTS) at **two granularities** (monthly and weekly) and select the best-performing champion per layer. The monthly layer is the primary decision surface; the weekly layer is an operational complement. Both layers feed into a temporal hierarchical reconciliation step.

## Current Decisions

- The canonical reproducible route is now `monthly_forecast_e2e`.
- `kedro run` is equivalent to `kedro run --pipeline monthly_forecast_e2e`.
- The monthly layer is the required champion layer; weekly remains scaffolded.
- Monthly model selection currently compares Prophet and SARIMAX.
- A generic monthly production champion is persisted as `champion_monthly_model`
  plus `champion_monthly_metadata`.
- Forecast inference is metadata-driven and dispatches by the selected champion
  family (`prophet`, `sarimax`, or `catboost`).
- The older Prophet-only route is retained as a reference route:
  `monthly_mvp` / `prophet_monthly_e2e`.
- CatBoost monthly training is now fully implemented: Optuna TPE tuning, validation
  metrics, prechampion configs, and candidate model artifacts are produced by
  `train_monthly.catboost`. The `model_input_preparation` stage generates
  CatBoost-ready splits with target-derived lag and rolling features via
  `adapt_monthly_data_for_catboost`. Recursive inference is also implemented and
  wired in `forecast_inference`. CatBoost is available via the `train_monthly`
  route but is not yet part of `monthly_forecast_e2e` or `monthly_model_selection`.
- User-upload routing must distinguish daily demand from monthly demand before
  training. Daily input can feed monthly and weekly workflows; monthly-only
  input must stay monthly-only unless a documented disaggregation method is
  added.

## Pipeline Status

| Pipeline | Status | Notes |
| --- | --- | --- |
| `data_ingestion` | Implemented and tested | Cleans demand and exogenous inputs; emits daily, weekly, monthly, and monthly exogenous primary datasets. |
| `feature_engineering_monthly` | Implemented and tested | Builds monthly calendar and exogenous features used by the monthly modeling path. |
| `model_input_preparation` | Implemented and tested | Builds generic monthly splits, Prophet-compatible splits/future frames, SARIMAX-compatible splits, and CatBoost-ready splits with target-derived lag and rolling features. |
| `train_monthly.prophet` | Implemented and tested | Optuna TPE tuning, validation metrics, top prechampions, and candidate artifacts. |
| `train_monthly.sarimax` | Implemented and tested | Optuna TPE-based SARIMAX tuning with exogenous variables, Ljung-Box residual filter, rolling-origin M-2/M-3 validation metrics, top prechampions, and candidate artifacts. |
| `train_monthly.catboost` | Implemented and tested | Optuna TPE tuning (50 trials, up to 10 prechampions), validation metrics, prechampion configs, candidate model artifacts, and training metadata. Recursive inference adapter wired in `forecast_inference`. |
| `monthly_model_selection` | Implemented and tested | Compares monthly Prophet and SARIMAX candidates on held-out test data and elects the monthly production champion. CatBoost not yet included. |
| `forecast_inference` | Implemented and tested | Generates standardized 3-, 6-, and 12-month forecasts from the generic monthly champion. Dispatches to Prophet, SARIMAX, or CatBoost (recursive) based on champion family. |
| `monthly_mvp` / `prophet_monthly_e2e` | Reference route | Prophet-only training and Prophet-specific champion selection. Preserved for comparison and audit continuity. |
| `feature_engineering_weekly` | Scaffolded | Node and pipeline structure exist; functions still raise `NotImplementedError`. |
| `train_weekly` | Scaffolded | Prophet, CatBoost, and SARIMAX structures exist; functions still raise `NotImplementedError`. |
| `reconciliation` | Scaffolded | Monthly-weekly reconciliation contract exists; implementation pending weekly completion. |
| `train_monthly` | Implemented (experimental route) | Composes Prophet, CatBoost, and SARIMAX sub-pipelines; all three families are now implemented. Not the default route; CatBoost is not yet included in `monthly_model_selection`. |
| `model_selection` | Legacy scaffold | Older multi-granularity selection path; use `monthly_model_selection` for the current implemented monthly path. |

## Canonical Monthly Flow

The current stable end-to-end route is:

```text
data_ingestion
  -> feature_engineering_monthly
    -> model_input_preparation
      -> train_monthly.prophet
      -> train_monthly.sarimax
        -> monthly_model_selection
          -> forecast_inference
```

This route performs the monthly multi-family comparison, refits the elected
production champion according to the champion protocol, and generates
application-ready monthly forecast outputs.

Run it from `pipelines/`:

```bash
uv run kedro run --pipeline monthly_forecast_e2e
```

Or simply:

```bash
uv run kedro run
```

## Registered Routes

List registered pipelines with:

```bash
uv run kedro registry list
```

Current public routes:

```bash
# Stable default monthly route
uv run kedro run --pipeline monthly_forecast_e2e
uv run kedro run

# Monthly comparison without final inference
uv run kedro run --pipeline prophet_sarimax_comparison

# Monthly generic model selection only
# Requires existing Prophet and SARIMAX candidate artifacts.
uv run kedro run --pipeline monthly_model_selection

# Prophet-only reference path
uv run kedro run --pipeline monthly_mvp
uv run kedro run --pipeline prophet_monthly_e2e

# Individual implemented stages
uv run kedro run --pipeline data_ingestion
uv run kedro run --pipeline feature_engineering_monthly
uv run kedro run --pipeline model_input_preparation
uv run kedro run --pipeline forecast_inference
```

Scaffolded or legacy routes are registered for architecture continuity, but are
not part of the stable path yet:

```bash
uv run kedro run --pipeline feature_engineering_weekly
uv run kedro run --pipeline train_weekly
uv run kedro run --pipeline reconciliation
uv run kedro run --pipeline train_monthly
uv run kedro run --pipeline model_selection
uv run kedro run --pipeline experimental_training
uv run kedro run --pipeline experimental_full_experiment
uv run kedro run --pipeline experimental_inference
```

These routes include placeholder nodes and may raise `NotImplementedError`.

## Data Layer Outputs

Key artifacts produced by the current monthly route:

```text
data/
├── 02_intermediate/
│   ├── demand_cleaned.parquet
│   └── exogenous_cleaned.parquet
├── 03_primary/
│   ├── demand_daily.parquet
│   ├── demand_weekly.parquet
│   ├── demand_monthly.parquet
│   └── exogenous_monthly.parquet
├── 04_feature/
│   ├── monthly_calendar_features.parquet
│   ├── monthly_exogenous_features.parquet
│   └── monthly_prophet_features.parquet
├── 05_model_input/
│   ├── monthly_modeling_data.parquet
│   ├── monthly_train.parquet
│   ├── monthly_validation.parquet
│   ├── monthly_test.parquet
│   ├── monthly_full_train.parquet
│   ├── monthly_split_metadata.json
│   ├── monthly_prophet_modeling_data.parquet
│   ├── monthly_prophet_train.parquet
│   ├── monthly_prophet_validation.parquet
│   ├── monthly_prophet_test.parquet
│   ├── monthly_prophet_full_train.parquet
│   ├── monthly_prophet_future_3m.parquet
│   ├── monthly_prophet_future_6m.parquet
│   ├── monthly_prophet_future_12m.parquet
│   ├── monthly_prophet_split_metadata.json
│   ├── monthly_sarimax_train.parquet
│   ├── monthly_sarimax_validation.parquet
│   ├── monthly_sarimax_test.parquet
│   ├── monthly_sarimax_full_train.parquet
│   ├── monthly_sarimax_split_metadata.json
│   ├── monthly_catboost_train.parquet
│   ├── monthly_catboost_validation.parquet
│   ├── monthly_catboost_test.parquet
│   ├── monthly_catboost_full_train.parquet
│   └── monthly_catboost_split_metadata.json
├── 06_models/
│   ├── tuning/
│   │   ├── monthly_prophet_tuning_results.parquet
│   │   ├── monthly_prophet_validation_metrics.parquet
│   │   ├── monthly_prophet_prechampion_configs.json
│   │   ├── monthly_prophet_training_metadata.json
│   │   ├── monthly_sarimax_tuning_results.parquet
│   │   ├── monthly_sarimax_validation_metrics.parquet
│   │   ├── monthly_sarimax_prechampion_configs.json
│   │   ├── monthly_sarimax_candidate_models.pkl
│   │   ├── monthly_sarimax_training_metadata.json
│   │   ├── monthly_catboost_tuning_results.parquet
│   │   ├── monthly_catboost_validation_metrics.parquet
│   │   ├── monthly_catboost_prechampion_configs.json
│   │   ├── monthly_catboost_candidate_models.pkl
│   │   └── monthly_catboost_training_metadata.json
│   ├── candidates/
│   │   ├── monthly_prophet_candidate_models.pkl
│   │   ├── monthly_prophet.pkl
│   │   ├── monthly_sarimax.pkl
│   │   └── monthly_catboost.pkl
│   ├── selection/
│   │   ├── monthly_candidate_test_metrics.parquet
│   │   ├── monthly_family_champion_summary.parquet
│   │   └── monthly_model_selection_summary.parquet
│   └── champions/
│       ├── monthly_champion.pkl
│       └── champion_monthly_metadata.json
└── 07_model_output/
    ├── monthly_forecast_3m.parquet
    ├── monthly_forecast_6m.parquet
    ├── monthly_forecast_12m.parquet
    ├── monthly_forecast_latest.parquet
    └── monthly_inference_metadata.json
```

Legacy and future-stage catalog entries also exist for raw/reconciled monthly
and weekly forecasts, CatBoost placeholders, weekly model inputs, reporting
tables, and reconciliation diagnostics.

## Champion Protocol

The implemented monthly path follows the staged model-selection decision:

1. Fit and tune candidates on the training split only.
2. Score tuned configurations on the validation split.
3. Preserve the top prechampions per model family.
4. Evaluate prechampions on the held-out temporal test split.
5. Select family champions and one monthly production champion.
6. Refit the selected production champion on all available historical data when
   configured.
7. Use the refit champion for the official 3-, 6-, and 12-month forecast outputs.

The primary selection metric is WAPE. MASE, RMSE, and bias-related tie breakers
are retained for auditability and academic reporting.

## Data Contracts

Demand and exogenous inputs must satisfy the project contracts before feature
generation:

- Dates are parsed as dates and sorted before feature generation.
- Duplicate timestamps at the same temporal level are invalid unless explicitly
  aggregated.
- Missing periods are handled explicitly and should not be silently inferred.
- Negative demand is invalid unless documented as returns or adjustments.
- Exogenous columns are configured explicitly and treated as core model inputs.
- Monthly input supports monthly training only.
- Daily input may be aggregated to monthly and weekly layers inside the pipeline
  layer.

## Configuration

All dataset I/O goes through `conf/base/catalog.yml`. Node code must not read or
write project data paths directly.

Configuration files:

```text
conf/base/
├── catalog.yml
├── parameters.yml
└── parameters/
    ├── evaluation.yml
    ├── feature_engineering.yml
    ├── forecast_inference.yml
    ├── model_input.yml
    ├── model_selection.yml
    ├── reconciliation.yml
    ├── train_monthly.yml
    └── train_weekly.yml
```

Local overrides and credentials belong in `conf/local/`, which is gitignored.

## Environment Setup

This repository uses `uv`. Do not use `pip` directly.

From the repository root:

```bash
uv sync --all-packages
uv tree
```

From `pipelines/`:

```bash
uv run kedro registry list
uv run kedro catalog describe-datasets --pipeline monthly_forecast_e2e
```

## Linting and Testing

From the repository root:

```bash
uv run --package hdf_pipelines ruff check pipelines/
uv run --package hdf_pipelines pytest pipelines/ --cov
```

From `pipelines/`:

```bash
uv run ruff check .
uv run pytest
uv run pytest tests/test_run.py
```

Current tests cover ingestion, monthly feature engineering, monthly model input
preparation, monthly Prophet training helpers, monthly SARIMAX training helpers,
monthly CatBoost training helpers, monthly model selection, forecast inference
(including CatBoost recursive dispatch), Optuna helpers, shared metrics, and
Kedro bootstrap/registry behavior.

## Project Structure

```text
pipelines/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters.yml
│   │   └── parameters/
│   └── local/
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
├── docs/
│   ├── demand_forecast_temporal_hierarchical_blueprint.md
│   ├── kedro_functional_logic_proposal.md
│   ├── monthly_model_input_contract.md
│   └── monthly_mvp_contract_snapshot.md
├── notebooks/
├── src/hdf_pipelines/
│   ├── settings.py
│   ├── pipeline_registry.py
│   └── pipelines/
│       ├── data_ingestion/
│       ├── feature_engineering_monthly/
│       ├── feature_engineering_weekly/
│       ├── model_input_preparation/
│       ├── train_monthly/
│       │   ├── prophet/
│       │   ├── catboost/
│       │   └── sarimax/
│       ├── train_weekly/
│       ├── model_selection/
│       │   ├── monthly/
│       │   └── prophet/
│       ├── reconciliation/
│       └── forecast_inference/
└── tests/
```

## Notebooks

Kedro-coupled notebooks live under `pipelines/notebooks/` and should be launched
from `pipelines/`:

```bash
uv run kedro jupyter lab
uv run kedro jupyter notebook
uv run kedro ipython
```

Production logic belongs in pipeline nodes under `src/hdf_pipelines/pipelines/`,
not in notebooks.

## Visualization

```bash
uv run kedro viz run
```

Kedro-Viz renders the DAG of nodes, datasets, and registered pipeline routes.

## Key References

- Blueprint: `docs/demand_forecast_temporal_hierarchical_blueprint.md`
- Monthly model input contract: `docs/monthly_model_input_contract.md`
- Monthly MVP contract snapshot: `docs/monthly_mvp_contract_snapshot.md`
- Kedro functional proposal: `docs/kedro_functional_logic_proposal.md`
- Root architecture overview: `../docs/architecture.md`
- Root contribution guide: `../CONTRIBUTING.md`

Consult the blueprint before changing architecture, hierarchy priorities,
validation methodology, or model-selection behavior.

## Conventions

- Nodes are pure functions: DataFrames and parameters in, DataFrames/artifacts
  out.
- All production data I/O goes through the Kedro catalog.
- Reusable metrics, schemas, loaders, and retrieval utilities belong in
  `shared/`.
- Monthly quality and reproducibility must not be compromised by weekly or daily
  extensions.
- Forecasting evaluation must be time-aware and leakage-safe.
- Do not commit raw data, generated heavy artifacts, credentials, or `.env`
  files.
- Use `uv` for dependency management.
