# hdf-pipelines

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Kedro-based pipeline layer for the **Hierarchical Demand Forecasting PoC** — a Master's final project in Applied AI. This package (`hdf_pipelines`) orchestrates the full forecasting lifecycle: data ingestion, feature engineering, model training, selection, reconciliation, and inference across monthly and weekly temporal granularities.

The pipeline is the backbone of a multi-layer system that also includes a FastAPI serving layer and a Streamlit viewer app. All data I/O follows Kedro's data engineering convention using a versioned catalog.

---

## North Star

The goal is to compare **~4 model families** (Prophet, CatBoost, SARIMAX, and optionally N-HiTS) at **two granularities** (monthly and weekly) and select the best-performing champion per layer. The monthly layer is the primary decision surface; the weekly layer is an operational complement. Both layers feed into a temporal hierarchical reconciliation step.

The current milestone completes the **Monthly Prophet MVP** — a fully runnable end-to-end path from raw data to official 3-month, 6-month and 12-month forecast outputs. CatBoost, SARIMAX, weekly, and reconciliation stages are scaffolded and will be validated in subsequent iterations.

---

## Pipeline Status

| Pipeline | Status | Tests | Notes |
|----------|--------|-------|-------|
| `data_ingestion` | ✅ Complete | ✅ | Cleaned demand + exogenous primary datasets |
| `feature_engineering_monthly` | ✅ Complete | ✅ | Calendar + exogenous features; Prophet-ready output |
| `model_input_preparation` | ✅ Complete | ✅ | Monthly Prophet train/val/test splits + future horizons (3m, 6m, 12m) |
| `train_monthly` | 🔄 Prophet done | — | Prophet: tuning + champion refit on full train ✅ · CatBoost, SARIMAX: scaffolded |
| `model_selection` | 🔄 Prophet done | — | Monthly Prophet: test-set evaluation + champion selection ✅ · CatBoost, SARIMAX, weekly: scaffolded |
| `forecast_inference` | 🔄 Prophet done | — | Monthly Prophet: 3m + 6m forecasts + latest + metadata ✅ · weekly, daily allocation: scaffolded |
| `feature_engineering_weekly` | 🔧 Wired | — | Nodes defined and wired; not yet end-to-end validated |
| `train_weekly` | 🔧 Wired | — | Prophet, CatBoost, SARIMAX sub-pipelines scaffolded |
| `reconciliation` | 🔧 Wired | — | MinT reconciliation + diagnostics; pending weekly completion |

**✅ Complete** — nodes implemented, pipeline runs end-to-end, outputs validated, unit tests in place.
**🔄 Prophet done** — Monthly Prophet path is fully operational; remaining model families (CatBoost, SARIMAX) and the weekly layer are scaffolded and pending validation.
**🔧 Wired** — `pipeline.py` and `nodes.py` in place and registered; not yet end-to-end tested.

---

## Monthly Prophet MVP — End-to-End Flow

The following pipeline sequence is fully validated and produces real, inspectable outputs:

```
data_ingestion
  → feature_engineering_monthly
    → model_input_preparation
      → train_monthly_prophet        # hyperparameter tuning + champion refit
        → model_selection_monthly_prophet   # test-set ranking + champion selection
          → forecast_inference       # 3m + 6m + 12m official forecasts
```

Run the complete MVP slice step by step:

```bash
kedro run --pipeline data_ingestion
kedro run --pipeline feature_engineering_monthly
kedro run --pipeline model_input_preparation
kedro run --pipeline train_monthly_prophet
kedro run --pipeline model_selection_monthly_prophet
kedro run --pipeline forecast_inference
```

Or run the entire flow with a single command using the composed pipeline:

```bash
kedro run --pipeline prophet_monthly_e2e
```

---

## Environment Setup

This project uses `uv` for dependency management. **Do not use `pip` directly.**

```bash
# From the pipelines/ directory
uv sync                    # Install all dependencies (including dev)
uv sync --no-dev           # Production dependencies only
```

---

## Running Pipelines

### Current default route

```bash
# Run the validated Monthly Prophet MVP — the current stable default
kedro run --pipeline monthly_mvp
kedro run                          # equivalent to monthly_mvp
```

This executes: `data_ingestion → feature_engineering_monthly → model_input_preparation → train_monthly_prophet → model_selection_monthly_prophet → forecast_inference`

### Individual validated stages

```bash
kedro run --pipeline data_ingestion
kedro run --pipeline feature_engineering_monthly
kedro run --pipeline model_input_preparation
kedro run --pipeline train_monthly             # Prophet only (CatBoost, SARIMAX scaffolded)
kedro run --pipeline model_selection           # Prophet path only
kedro run --pipeline forecast_inference        # Monthly Prophet outputs only
kedro run --pipeline prophet_monthly_e2e       # same as monthly_mvp
```

### Scaffolded stages — ⚠️ not yet end-to-end validated

```bash
kedro run --pipeline feature_engineering_weekly    # NotImplementedError
kedro run --pipeline train_weekly                  # NotImplementedError
kedro run --pipeline reconciliation                # NotImplementedError
```

### Experimental composed shortcuts — ⚠️ include NotImplementedError stubs, will fail

```bash
kedro run --pipeline experimental_training              # train_monthly + train_weekly
kedro run --pipeline experimental_full_experiment       # ingestion → all features → training → selection
kedro run --pipeline experimental_inference             # forecast_inference + reconciliation
```

### Full intended execution order (target architecture, not yet fully executable)

```
data_ingestion
    → feature_engineering_monthly
    → feature_engineering_weekly        [🔧 scaffolded]
        → model_input_preparation
            → train_monthly (prophet · catboost · sarimax)
            → train_weekly  (prophet · catboost · sarimax)  [🔧 scaffolded]
                → model_selection                            [🔧 scaffolded]
                    → reconciliation                         [🔧 scaffolded]
                        → forecast_inference
```

---

## Data Layer — Current Outputs

Artifacts produced by the Monthly Prophet MVP path:

```
data/
├── 02_intermediate/
│   ├── demand_cleaned.parquet
│   └── exogenous_cleaned.parquet
├── 03_primary/
│   ├── demand_daily.parquet
│   ├── demand_monthly.parquet
│   ├── demand_weekly.parquet
│   └── exogenous_monthly.parquet
├── 04_feature/
│   ├── monthly_calendar_features.parquet
│   ├── monthly_exogenous_features.parquet
│   └── monthly_prophet_features.parquet
├── 05_model_input/
│   ├── monthly_prophet_modeling_data.parquet
│   ├── monthly_prophet_train.parquet
│   ├── monthly_prophet_validation.parquet
│   ├── monthly_prophet_test.parquet
│   ├── monthly_prophet_full_train.parquet
│   ├── monthly_prophet_future_3m.parquet       # inference input (3-month horizon)
│   ├── monthly_prophet_future_6m.parquet       # inference input (6-month horizon)
│   ├── monthly_prophet_future_12m.parquet      # inference input (12-month horizon)
│   └── monthly_prophet_split_metadata.json
├── 06_models/
│   ├── candidates/
│   │   └── monthly_prophet_candidate_models.pkl  # all tuned Prophet candidates
│   ├── tuning/
│   │   ├── monthly_prophet_tuning_results.parquet
│   │   ├── monthly_prophet_validation_metrics.parquet
│   │   ├── monthly_prophet_prechampion_configs.json
│   │   └── monthly_prophet_training_metadata.json
│   ├── selection/
│   │   ├── monthly_prophet_test_metrics.parquet
│   │   ├── monthly_prophet_model_selection_summary.parquet
│   │   └── monthly_prophet_champion_test_forecast.parquet
│   └── champions/
│       ├── monthly_prophet_champion.pkl          # champion model (refit on full train)
│       └── monthly_prophet_champion_metadata.json
└── 07_model_output/
    ├── monthly_prophet_forecast_3m.parquet      # official 3-month forecast
    ├── monthly_prophet_forecast_6m.parquet      # official 6-month forecast
    ├── monthly_prophet_forecast_12m.parquet     # official 12-month forecast
    ├── monthly_prophet_forecast_latest.parquet  # alias of 12m; consumed by app + API
    └── monthly_prophet_inference_metadata.json  # run ID, regressors, horizon summaries
```

Layers `08_reporting/` and reconciled outputs in `07_model_output/` will be populated once the full multi-model selection and reconciliation stages are validated.

---

## Project Structure

```
pipelines/
├── conf/
│   ├── base/                          # Shared configuration (catalog, parameters)
│   │   ├── catalog.yml                # Dataset definitions (all I/O goes here)
│   │   ├── parameters.yml             # Global parameters
│   │   ├── parameters/                # Per-pipeline parameter files
│   │   │   ├── feature_engineering.yml
│   │   │   ├── model_input.yml
│   │   │   ├── train_monthly.yml
│   │   │   ├── train_weekly.yml
│   │   │   ├── model_selection.yml
│   │   │   ├── reconciliation.yml
│   │   │   ├── evaluation.yml
│   │   │   └── forecast_inference.yml
│   │   └── logging.yml
│   └── local/                         # Local overrides and credentials (gitignored)
│
├── data/
│   ├── 01_raw/                        # Raw input data (gitignored)
│   ├── 02_intermediate/               # ✅ Cleaned/preprocessed data
│   ├── 03_primary/                    # ✅ Domain-level aggregated data
│   ├── 04_feature/                    # ✅ Feature-engineered datasets
│   ├── 05_model_input/                # ✅ Model input tables and splits (Prophet)
│   ├── 06_models/                     # 🔄 Prophet training + selection artifacts
│   ├── 07_model_output/               # 🔄 Prophet forecast outputs (3m, 6m, latest)
│   └── 08_reporting/                  # 🔧 Evaluation reports (pending multi-model)
│
├── src/hdf_pipelines/
│   ├── __init__.py
│   ├── __main__.py
│   ├── settings.py                    # Kedro project settings (OmegaConfigLoader)
│   ├── pipeline_registry.py           # Registers all pipelines and composed shortcuts
│   └── pipelines/
│       ├── data_ingestion/            # ✅ Load, clean, and aggregate raw demand + exogenous data
│       ├── feature_engineering_monthly/  # ✅ Calendar and exogenous features at monthly resolution
│       ├── feature_engineering_weekly/   # 🔧 Aggregate demand and build weekly features
│       ├── model_input_preparation/   # ✅ Merge monthly features → Prophet train/val/test/future splits
│       ├── train_monthly/             # 🔄 Prophet: tuned + champion ✅ · CatBoost, SARIMAX: scaffolded
│       │   ├── prophet/               # ✅ Hyperparameter search + candidate training + champion refit
│       │   ├── catboost/              # 🔧 Scaffolded
│       │   └── sarimax/               # 🔧 Scaffolded
│       ├── train_weekly/              # 🔧 Prophet, CatBoost, SARIMAX sub-pipelines scaffolded
│       │   ├── prophet/
│       │   ├── catboost/
│       │   └── sarimax/
│       ├── model_selection/           # 🔄 Monthly Prophet: test evaluation + champion ✅ · others: scaffolded
│       │   └── prophet/               # ✅ Test-set evaluation, ranking, champion metadata, model refit
│       ├── reconciliation/            # 🔧 MinT reconciliation; pending monthly + weekly completion
│       └── forecast_inference/        # 🔄 Monthly Prophet: 3m + 6m + latest ✅ · weekly + allocation: scaffolded
│
├── notebooks/
│   └── kedro_demo.ipynb              # Demo notebook (catalog + pipeline exploration)
├── tests/
│   ├── test_data_ingestion_nodes.py
│   ├── test_feature_engineering_monthly_nodes.py
│   ├── test_model_input_preparation_nodes.py
│   ├── test_metrics.py
│   ├── test_pipeline_registry.py
│   └── test_run.py
└── pyproject.toml
```

---

## Catalog and Parameters

All datasets are defined in `conf/base/catalog.yml`. Never read or write data files directly from node code — always go through the catalog.

Per-pipeline parameters live in `conf/base/parameters/<pipeline_name>.yml`. Global parameters are in `conf/base/parameters.yml`.

```bash
kedro catalog list       # List all registered datasets
kedro pipeline list      # List all registered pipelines
```

---

## Linting and Testing

```bash
# Lint
ruff check src/
ruff format src/

# Tests
pytest                          # Run all tests with coverage
pytest tests/test_run.py        # Smoke test for Kedro bootstrap
```

Current test coverage includes: `data_ingestion`, `feature_engineering_monthly`, `model_input_preparation`, and shared `metrics`. Tests for the training and inference nodes are pending.

---

## Visualization

```bash
kedro viz run    # Launch Kedro-Viz at http://localhost:4141
```

Kedro-Viz renders the full DAG of nodes, datasets, and pipeline dependencies — useful for understanding execution flow and debugging catalog wiring.

---

## Notebooks

`notebooks/kedro_demo.ipynb` demonstrates how to load catalog datasets and explore pipeline outputs interactively. Production logic must live in pipeline nodes under `src/hdf_pipelines/pipelines/`.

```bash
kedro jupyter lab       # Launch JupyterLab with catalog/context pre-loaded
kedro jupyter notebook  # Launch classic Jupyter
kedro ipython           # Launch IPython session
```

> Use [`nbstripout`](https://github.com/kynan/nbstripout) to strip notebook outputs before committing: `nbstripout --install`.

---

## Key Conventions

- **Nodes must be pure functions**: inputs in, outputs out. No side effects, no direct file I/O.
- **All datasets go through the catalog** — no `pd.read_csv(...)` in node code.
- **Config is environment-aware**: `conf/base/` is shared, `conf/local/` is gitignored.
- **Do not commit data, credentials, or `.env` files.**
- **Use `uv`**, not `pip`, for all dependency management.
