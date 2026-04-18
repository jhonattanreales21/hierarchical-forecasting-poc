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
# TODO: fill in Stage 2
```

### Streamlit (app/)

```bash
# TODO: fill in Stage 3
```

### FastAPI (api/)

```bash
# TODO: fill in Stage 3
```

### Docker

```bash
# TODO: fill in Stage 3
```

---

## Blueprint

See [`pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md`](pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md)
for the full strategic blueprint: methodology, scope, model candidates, evaluation protocol,
and AI-assisted development guidelines.
