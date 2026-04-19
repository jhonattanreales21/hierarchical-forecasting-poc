# Hierarchical Demand Forecasting PoC

Proof of concept for temporal hierarchical demand forecasting of a critical SKU.
The system implements a **Monthly benchmark → Weekly anchor → optional Daily allocation**
hierarchy supported by exogenous variables, modular Kedro pipelines, MLflow experiment
tracking, and an application layer built with Streamlit and FastAPI.

Developed as a final master's project (MIAA 2025-2026) by **Jhonattan Reales** and **Andres Cano**.

---

## Mono-repo Structure

```
hierarchical-demand-forecasting-poc/
├── pipelines/          # Kedro project: ingestion, feature engineering,
│                       # training (monthly/weekly/daily), evaluation
├── shared/             # Internal library: schemas, metrics, loaders, viz helpers
├── app/                # Streamlit forecast viewer (Stage 3)
├── api/                # FastAPI serving layer (Stage 3)
├── pyproject.toml      # uv workspace root (not a Python package)
└── .python-version     # Python 3.12
```

---

## Getting Started

```bash
uv sync
```

This installs all workspace packages (`pipelines`, `shared`, `app`, `api`)
and their dependencies into a shared `.venv` at the repo root.

---

## Commands

### Kedro (pipelines/)

```bash
# Install all dependencies
uv sync

# Run the full default pipeline (ingestion → features → training → selection → reconciliation → inference)
uv run --package hierarchical_demand_forecasting_poc kedro run

# Run only data ingestion and feature engineering
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline data_ingestion
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline feature_engineering_monthly
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline feature_engineering_weekly

# Run only training (both monthly and weekly, all model families)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline training

# Run only monthly training
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline train_monthly

# Run only weekly training
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline train_weekly

# Run model selection (champion selection on test data)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline model_selection

# Run reconciliation only
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline reconciliation

# Run inference (generates forecast outputs)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline inference

# Run the full experiment (all stages except final inference)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline full_experiment
```

### Streamlit (app/)

```bash
uv run --package hdf_app streamlit run app/app.py
```

### FastAPI (api/)

```bash
uv run --package hdf_api uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
# Run all services (pipelines + app + api)
docker compose -f docker/docker-compose.yml up
```

---

## Blueprint

See [`pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md`](pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md)
for the full strategic blueprint: methodology, scope, model candidates, evaluation protocol,
and AI-assisted development guidelines.
