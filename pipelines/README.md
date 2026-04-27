# hdf-pipelines

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Kedro-based pipeline layer for the **Hierarchical Demand Forecasting PoC** — a Master's final project in Applied AI. This package (`hdf_pipelines`) orchestrates the full forecasting lifecycle: data ingestion, feature engineering, model training, selection, reconciliation, and inference across monthly and weekly temporal granularities.

The pipeline is the backbone of a multi-layer system that also includes a FastAPI serving layer and a Streamlit viewer app. All data I/O follows Kedro's data engineering convention using a versioned catalog.

---

## Pipeline Status

| Pipeline | Status | Tests | Notes |
|----------|--------|-------|-------|
| `data_ingestion` | ✅ Implemented | ✅ | Produces cleaned demand and exogenous primary datasets |
| `feature_engineering_monthly` | ✅ Implemented | ✅ | Calendar + exogenous features; Prophet-ready output |
| `model_input_preparation` | ✅ Implemented | ✅ | Monthly Prophet train/val/test splits + future horizons |
| `feature_engineering_weekly` | 🔧 Wired | — | Nodes defined and wired; not yet end-to-end validated |
| `train_monthly` | 🔧 Wired | — | Prophet, CatBoost, SARIMAX sub-pipelines scaffolded |
| `train_weekly` | 🔧 Wired | — | Prophet, CatBoost, SARIMAX sub-pipelines scaffolded |
| `model_selection` | 🔧 Wired | — | Evaluate candidates + elect champion per granularity |
| `reconciliation` | 🔧 Wired | — | MinT reconciliation + diagnostics |
| `forecast_inference` | 🔧 Wired | — | Forward-looking predictions + daily allocation |

**✅ Implemented** — nodes complete, pipeline runs, outputs validated, unit tests in place.
**🔧 Wired** — `pipeline.py` and `nodes.py` in place and registered; not yet end-to-end tested.

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

```bash
# Run only the currently validated slice (ingestion → monthly features → Prophet model input)
kedro run --pipeline=data_ingestion
kedro run --pipeline=feature_engineering_monthly
kedro run --pipeline=model_input_preparation

# Run a specific wired-but-not-yet-validated stage (use with care)
kedro run --pipeline=feature_engineering_weekly
kedro run --pipeline=train_monthly
kedro run --pipeline=train_weekly
kedro run --pipeline=model_selection
kedro run --pipeline=reconciliation
kedro run --pipeline=forecast_inference

# Composed pipeline shortcuts
kedro run --pipeline=training          # train_monthly + train_weekly
kedro run --pipeline=full_experiment   # ingestion → feature engineering → model input → training → selection
kedro run --pipeline=inference         # forecast_inference + reconciliation

# Run the full default pipeline (all stages in sequence)
kedro run
```

### Pipeline Execution Order

```
data_ingestion
    → feature_engineering_monthly
    → feature_engineering_weekly
        → model_input_preparation
            → train_monthly + train_weekly
                → model_selection
                    → reconciliation
                        → forecast_inference
```

The first three stages (up to and including `model_input_preparation`) are validated and produce real outputs. Stages after that are wired and will be fully validated in the next development iteration.

---

## Data Layer — Current Outputs

The following catalog artifacts have been produced by the implemented pipelines:

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
└── 05_model_input/
    ├── monthly_prophet_modeling_data.parquet
    ├── monthly_prophet_train.parquet
    ├── monthly_prophet_validation.parquet
    ├── monthly_prophet_test.parquet
    ├── monthly_prophet_full_train.parquet
    ├── monthly_prophet_future_3m.parquet
    ├── monthly_prophet_future_6m.parquet
    └── monthly_prophet_split_metadata.json
```

Layers `06_models/`, `07_model_output/`, and `08_reporting/` will be populated once the training and inference pipelines are validated.

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
│   ├── 02_intermediate/               # Cleaned/preprocessed data
│   ├── 03_primary/                    # Domain-level aggregated data
│   ├── 04_feature/                    # Feature-engineered datasets
│   ├── 05_model_input/                # Final model input tables and splits
│   ├── 06_models/                     # Trained model artifacts
│   ├── 07_model_output/               # Predictions and forecast outputs
│   └── 08_reporting/                  # Evaluation reports and plots
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
│       ├── model_input_preparation/   # ✅ Merge monthly features → Prophet train/val/test splits
│       ├── train_monthly/             # 🔧 Train Prophet, CatBoost, SARIMAX at monthly level
│       │   ├── prophet/
│       │   ├── catboost/
│       │   └── sarimax/
│       ├── train_weekly/              # 🔧 Train Prophet, CatBoost, SARIMAX at weekly level
│       │   ├── prophet/
│       │   ├── catboost/
│       │   └── sarimax/
│       ├── model_selection/           # 🔧 Score candidates on test set and elect champions
│       ├── reconciliation/            # 🔧 MinT reconciliation for monthly–weekly coherence
│       └── forecast_inference/        # 🔧 Forward-looking forecasts + optional daily allocation
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

Current test coverage includes: `data_ingestion`, `feature_engineering_monthly`, `model_input_preparation`, and shared `metrics`.

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
