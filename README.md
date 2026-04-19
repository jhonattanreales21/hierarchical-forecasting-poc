# Hierarchical Demand Forecasting PoC

![CI — shared](https://github.com/jhonattanreales21/hierarchical-forecasting-poc/actions/workflows/ci-shared.yml/badge.svg)
![CI — pipelines](https://github.com/jhonattanreales21/hierarchical-forecasting-poc/actions/workflows/ci-pipelines.yml/badge.svg)
![CI — app](https://github.com/jhonattanreales21/hierarchical-forecasting-poc/actions/workflows/ci-app.yml/badge.svg)
![CI — api](https://github.com/jhonattanreales21/hierarchical-forecasting-poc/actions/workflows/ci-api.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-workspace-7C3AED?logo=astral&logoColor=white)
![Kedro](https://img.shields.io/badge/kedro-1.3-FFC900?logoColor=black)
![License](https://img.shields.io/badge/license-MIT-22C55E)

Proof of concept for **temporal hierarchical demand forecasting** of a critical SKU.
The system implements a **Monthly → Weekly → Daily** forecasting hierarchy supported by
exogenous variables, reproducible Kedro pipelines, MLflow experiment tracking, and an
application layer built with Streamlit and FastAPI.

Developed as a master's final project (MIAA 2025–2026) at **Universidad Icesi**.

| Author | Email |
|--------|-------|
| Jhonattan Reales | jhonatanreales21@gmail.com |
| Andres Cano | andres.cano.consulting@gmail.com |

---

## How It Works

The forecasting system is built around a strict temporal hierarchy:

1. **Monthly layer (primary)** — The main analytical and business-facing output. Models are trained and evaluated to improve forecast accuracy from a 82.3% baseline to ≥ 85%. This is the core deliverable and the focus for stakeholder reporting.
2. **Weekly layer (anchor)** — A 14-week operational complement that adds short-term interpretability. Valuable as a secondary signal, but must not compromise the monthly layer.
3. **Daily layer (disabled by default)** — An optional disaggregation extension. Controlled by `daily_allocation.enabled` in `conf/base/parameters/forecast_inference.yml`.

Three model families are evaluated at each active layer:

| Model | Role | Layers |
|-------|------|--------|
| **SARIMAX** | Structured statistical baseline with seasonal + exogenous terms | Monthly |
| **Prophet** | Existing benchmark; robust to trend changes and seasonality | Monthly & Weekly |
| **CatBoost** | Main tabular candidate with full exogenous variable support | Monthly & Weekly |

After training, forecasts at all granularities are **reconciled** using MinT (`mint_shrink`) to ensure temporal coherence — weekly forecasts within a month are consistent with the monthly total. The final outputs are exposed through a Streamlit app and a FastAPI layer.

---

## Tech Stack

| Area | Tool |
|------|------|
| Pipeline orchestration | [Kedro](https://kedro.org/) ~1.3 |
| Package & env management | [uv](https://docs.astral.sh/uv/) (workspace mono-repo) |
| Experiment tracking | [MLflow](https://mlflow.org/) |
| Modeling | Statsmodels (SARIMAX), Prophet, CatBoost |
| Data contracts | [Pydantic](https://docs.pydantic.dev/) v2 |
| Forecast app | [Streamlit](https://streamlit.io/) |
| Serving layer | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI | GitHub Actions |
| Linting & formatting | [Ruff](https://docs.astral.sh/ruff/) |
| Testing | pytest |
| Language | Python 3.12 |

---

## Mono-repo Structure

```
hierarchical-demand-forecasting-poc/
├── pipelines/                  # Kedro project — all data and ML logic
│   ├── conf/base/              # Catalog, parameters, pipeline configs
│   ├── data/                   # Layered data store (01_raw → 08_reporting)
│   ├── docs/                   # Pipeline-level blueprint and proposals
│   └── src/
│       └── hierarchical_demand_forecasting_poc/
│           └── pipelines/      # data_ingestion, feature_engineering, train_*, ...
├── shared/                     # Internal library: schemas, metrics, loaders, viz
│   └── src/shared/
│       ├── schemas.py          # Pydantic contracts (ForecastRecord, BacktestResult, ...)
│       ├── metrics.py          # MAPE, RMSE, MASE
│       ├── loaders.py          # Data loading helpers
│       └── viz.py              # Visualization utilities
├── app/                        # Streamlit forecast viewer (hdf_app)
│   ├── app.py                  # Entrypoint
│   └── pages/                  # 01_project_overview … 05_evaluation_report
├── api/                        # FastAPI serving layer (hdf_api)
│   ├── main.py                 # App factory + /health
│   └── routers/forecast.py     # /forecast/latest, /backtest, /champion
├── docs/                       # Project-level documentation
│   ├── architecture.md         # System design and data flow
│   ├── data-catalog.md         # Kedro data layer reference
│   └── contributing.md         # Branching, commits, local checks
├── docker/                     # Dockerfiles + docker-compose.yml
├── .github/workflows/          # CI workflows per package
├── pyproject.toml              # uv workspace root (not a Python package)
└── .python-version             # Python 3.12
```

---

## Getting Started

### Prerequisites

- [Python 3.12](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Docker](https://docs.docker.com/get-docker/) _(optional, for containerised runs)_

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/jhonattanreales21/hierarchical-forecasting-poc.git
cd hierarchical-forecasting-poc

# 2. Install all workspace packages into a shared virtual environment
uv sync

# 3. Place raw input files in the data layer
cp your_demand_data.csv pipelines/data/01_raw/raw_demand_data.csv
cp your_exogenous_data.csv pipelines/data/01_raw/raw_exogenous_data.csv

# 4. Run the full pipeline
uv run --package hierarchical_demand_forecasting_poc kedro run

# 5. Launch the Streamlit app
uv run --package hdf_app streamlit run app/app.py
```

---

## Commands

### Kedro (pipelines/)

```bash
# Run the full default pipeline (ingestion → features → training → selection → reconciliation → inference)
uv run --package hierarchical_demand_forecasting_poc kedro run

# Run only data ingestion
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline data_ingestion

# Run training (monthly + weekly, all model families)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline training

# Run model selection (champion selection on test data)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline model_selection

# Run inference (generates forecast outputs under data/07_model_output/)
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline inference

# Run the full experiment without final inference
uv run --package hierarchical_demand_forecasting_poc kedro run --pipeline full_experiment
```

### Streamlit (app/)

```bash
uv run --package hdf_app streamlit run app/app.py
# Opens at http://localhost:8501
```

### FastAPI (api/)

```bash
uv run --package hdf_api uvicorn api.main:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs
```

### Docker

```bash
# Build and run all services (pipelines + app + api)
docker compose -f docker/docker-compose.yml up
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System design, data flow diagram, temporal hierarchy |
| [docs/data-catalog.md](docs/data-catalog.md) | Kedro data layer reference (01_raw → 08_reporting) |
| [docs/contributing.md](docs/contributing.md) | Branching strategy, commit style, local checks |
| [pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md](pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md) | Full strategic blueprint: methodology, scope, success criteria |
| [pipelines/docs/kedro_functional_logic_proposal.md](pipelines/docs/kedro_functional_logic_proposal.md) | Detailed Kedro pipeline logic proposal |

---

## Acknowledgments

Developed as part of the Master's program in Applied Artificial Intelligence at Universidad Icesi. Thanks to our tutors and peers for their guidance and feedback throughout the project.
