# hdf-pipelines

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Kedro-based pipeline layer for the **Hierarchical Demand Forecasting PoC** — a Master's final project in Applied AI. This package (`hdf_pipelines`) orchestrates the full forecasting lifecycle: data ingestion, feature engineering, model training, selection, reconciliation, and inference across monthly and weekly temporal granularities.

The pipeline is the backbone of a multi-layer system that also includes a FastAPI serving layer and a Streamlit viewer app. All data I/O follows Kedro's data engineering convention using a versioned catalog.

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
# Run the full default pipeline (ingestion → training → reconciliation → inference)
kedro run

# Run a specific pipeline stage
kedro run --pipeline=data_ingestion
kedro run --pipeline=feature_engineering_monthly
kedro run --pipeline=feature_engineering_weekly
kedro run --pipeline=model_input_preparation
kedro run --pipeline=train_monthly
kedro run --pipeline=train_weekly
kedro run --pipeline=model_selection
kedro run --pipeline=reconciliation
kedro run --pipeline=forecast_inference

# Run composed pipeline shortcuts
kedro run --pipeline=training          # train_monthly + train_weekly
kedro run --pipeline=full_experiment   # ingestion → feature engineering → model input → training → selection
kedro run --pipeline=inference         # forecast_inference + reconciliation
```

### Pipeline Execution Order (default)

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

---

## Project Structure

```
pipelines/
├── conf/
│   ├── base/                          # Shared configuration (catalog, parameters)
│   │   ├── catalog.yml                # Dataset definitions (all I/O goes here)
│   │   ├── parameters.yml             # Global parameters
│   │   ├── parameters_<pipeline>.yml  # Per-pipeline parameters
│   │   └── logging.yml                # Logging configuration
│   └── local/                         # Local overrides and credentials (gitignored)
│
├── data/
│   ├── 01_raw/                        # Raw input data (gitignored)
│   ├── 02_intermediate/               # Cleaned/preprocessed data
│   ├── 03_primary/                    # Domain-level model-ready data
│   ├── 04_feature/                    # Feature-engineered datasets
│   ├── 05_model_input/                # Final model input tables
│   ├── 06_models/                     # Trained model artifacts
│   ├── 07_model_output/               # Predictions and forecast outputs
│   └── 08_reporting/                  # Evaluation reports and plots
│
├── src/hdf_pipelines/
│   ├── __init__.py                    # Package version
│   ├── __main__.py                    # CLI entry point
│   ├── settings.py                    # Kedro project settings (OmegaConfigLoader)
│   ├── pipeline_registry.py           # Registers all pipelines (including composed ones)
│   └── pipelines/
│       ├── data_ingestion/            # Load and validate raw data
│       ├── feature_engineering_monthly/  # Monthly-level feature construction
│       ├── feature_engineering_weekly/   # Weekly-level feature construction
│       ├── model_input_preparation/   # Merge features → model-ready datasets
│       ├── train_monthly/             # Train monthly forecasting models
│       ├── train_weekly/              # Train weekly forecasting models
│       ├── model_selection/           # Compare and select best model per horizon
│       ├── reconciliation/            # Temporal hierarchical reconciliation
│       └── forecast_inference/        # Generate final forecasts using selected models
│
├── tests/                             # pytest tests
├── notebooks/                         # Exploratory analysis (not production logic)
├── docs/                              # Sphinx documentation
└── pyproject.toml
```

---

## Catalog and Parameters

All datasets are defined in `conf/base/catalog.yml`. Never read or write data files directly from node code — always go through the catalog.

Per-pipeline parameters live in `conf/base/parameters_<pipeline_name>.yml`. Global parameters are in `conf/base/parameters.yml`.

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

---

## Visualization

```bash
kedro viz run    # Launch Kedro-Viz at http://localhost:4141
```

Kedro-Viz renders the full DAG of nodes, datasets, and pipeline dependencies — useful for understanding execution flow and debugging catalog wiring.

---

## Notebooks

Notebooks in `notebooks/` are for exploratory analysis only. Production logic must live in pipeline nodes under `src/hdf_pipelines/pipelines/`.

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
