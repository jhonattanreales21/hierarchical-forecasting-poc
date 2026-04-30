# CLAUDE.md

## Project Identity

**Hierarchical Demand Forecasting PoC** — Proof of concept for temporal hierarchical demand forecasting of a critical SKU. Academic final project (Master's in Applied AI) developed by **Jhonattan Reales** and **Andres Cano** at Universidad Icesi.

The project is built as a **Forecasting + MLOps + Application Layer** system, not a forecasting model in isolation. It must reach a level of organization and technical maturity befitting a master's final project.

## North Star

Build a functional, reproducible, and academically defensible PoC that shows how temporal hierarchical forecasting (monthly benchmark + weekly forecasting), supported by exogenous variables and clean architecture, can improve demand forecasting for a critical SKU. The **monthly layer is the primary analytical and business-facing component**; the weekly layer is a secondary enhancement.

## Tech Stack

- **Python** >= 3.10
- **uv** (package and workspace management — use `uv` instead of `pip`)
- **Kedro** ~1.3.1 (pipeline orchestration and project structure)
- **Pandas / Scikit-learn / Statsmodels** (data manipulation and utilities)
- **Prophet / CatBoost / SARIMAX** (core model candidates)
- **Nixtla NeuralForecast (N-HiTS)** (optional exploratory model)
- **MLflow** (experiment tracking and artifact versioning)
- **Streamlit** (forecast viewer application)
- **FastAPI** (serving layer)
- **Docker** (reproducibility and deployment)
- **pytest + ruff** (testing and linting)

## Mono-repo Packages

The repository is structured as a **uv workspace** with four packages:

| Package | Directory | Role |
|---------|-----------|------|
| `hdf_pipelines` | `pipelines/` | Kedro project — all data and ML logic |
| `hdf_shared` | `shared/` | Internal library — schemas, metrics, loaders, viz utilities |
| `hdf_app` | `app/` | Streamlit forecast viewer |
| `hdf_api` | `api/` | FastAPI serving layer |

## Common Commands

```bash
# Environment setup
uv sync --all-packages              # Install all workspace dependencies
uv tree                             # Verify available workspace packages

# Kedro pipelines (from pipelines/ directory)
cd pipelines
uv run kedro pipeline list          # List registered pipelines
uv run kedro run                    # Run the full default pipeline
uv run kedro run --pipeline <name>  # Run a specific pipeline
uv run kedro catalog list           # List all catalog datasets
uv run kedro viz run                # Launch Kedro-Viz visualizer

# App and API (from repo root)
uv run --package hdf_app streamlit run app/app.py
uv run --package hdf_api uvicorn hdf_api.main:app --reload --port 8000

# Testing
uv run --package hdf_pipelines pytest pipelines/ --cov

# Linting
uv run --package hdf_pipelines ruff check pipelines/
uv run --package hdf_app ruff check app/
uv run --package hdf_api ruff check api/
uv run --package hdf_shared ruff check shared/

# Kedro pipeline scaffolding
uv run kedro pipeline create <pipeline_name>
```

## Project Structure

```
pipelines/                          # hdf_pipelines — Kedro project
  conf/
    base/                           # Shared config (catalog, parameters)
      catalog.yml
      parameters.yml
      parameters_<pipeline>.yml     # Per-pipeline parameters
    local/                          # Local overrides and credentials (gitignored)
  data/
    01_raw/                         # Raw input CSVs (gitignored)
    02_intermediate/                # Cleaned, validated parquets
    03_primary/                     # Domain-level weekly and monthly demand
    04_feature/                     # Feature-engineered datasets
    05_model_input/                 # Train / val / test splits per model
    06_models/                      # Trained artifacts and champion registry
    07_model_output/                # Raw and reconciled forecasts
    08_reporting/                   # Evaluation reports and diagnostics
  src/hdf_pipelines/
    settings.py                     # Kedro settings (OmegaConfigLoader)
    pipeline_registry.py            # Auto-discovers and registers all pipelines
    pipelines/<pipeline_name>/
      nodes.py                      # Pure functions (business logic)
      pipeline.py                   # Pipeline wiring (node → pipeline)
  docs/                             # Blueprint, architecture, forecasting docs
  notebooks/                        # Kedro-coupled notebooks (kedro jupyter notebook)

shared/                             # hdf_shared — reusable utilities
app/                                # hdf_app — Streamlit viewer
api/                                # hdf_api — FastAPI serving layer
notebooks/                          # General EDA notebooks (not pipeline-coupled)
```

> **Notebook rule:** notebooks that access the Kedro `DataCatalog` live in `pipelines/notebooks/` and must be launched with `uv run kedro jupyter notebook`. EDA notebooks unrelated to the pipeline go in the root `notebooks/` folder.

## Kedro Pipelines

The forecasting lifecycle is organized into nine stage-oriented pipelines:

| Pipeline | Responsibility |
|----------|---------------|
| `data_ingestion` | Load, clean, and validate raw demand and exogenous data |
| `feature_engineering_monthly` | Aggregate to monthly granularity; build lag, rolling, and calendar features |
| `feature_engineering_weekly` | Aggregate to weekly granularity; build operational and exogenous features |
| `model_input_preparation` | Build train / val / test splits and backtesting windows per model family |
| `train_monthly` | Train and tune monthly candidates (Prophet, CatBoost, SARIMAX) |
| `train_weekly` | Train and tune weekly candidates (Prophet, CatBoost, SARIMAX) |
| `model_selection` | Evaluate candidates on held-out test data; select champion models |
| `reconciliation` | Apply temporal hierarchical reconciliation (default: `mint_shrink`) |
| `forecast_inference` | Generate final predictions for batch or on-demand use |

Within `train_monthly` and `train_weekly`, use **namespaces** to separate model families (e.g., `monthly.prophet`, `monthly.catboost`) rather than creating additional top-level pipelines.

## Temporal Hierarchy

The forecasting hierarchy is strictly temporal:

1. **Monthly** — **Primary decision layer.** Greatest evaluation effort, stakeholder focus, and narrative weight. Central benchmark aligned with the business planning horizon.
2. **Weekly** — Secondary enhancement layer. Operational complement; must not compromise monthly quality.
3. **Daily** — Low-priority exploratory extension. Pragmatic disaggregation only if feasible; not a core deliverable.

Primary coherence target: monthly ↔ weekly. Full monthly ↔ weekly ↔ daily is treated as secondary scope. Default reconciliation: `mint_shrink` (configured in `conf/base/parameters/reconciliation.yml`).

## Model Candidates

- **SARIMAX** — Structured statistical baseline, especially relevant for the monthly layer
- **Prophet** — Existing benchmark and early starting point
- **CatBoost** — Main tabular candidate with exogenous variables
- **N-HiTS (Nixtla)** — Optional neural benchmark, only after the monthly layer is stable

## Evaluation

Primary metrics: **MAPE**, **RMSE**, **MASE**. Secondary: bias, interval coverage, horizon-specific error.

Validation must be time-aware: **rolling-origin** or equivalent time-based backtesting. Never use random splits for time series.

## Development Guidelines

### Naming and Structure

- Classes: `PascalCase` · Methods and variables: `snake_case` · Private: `_prefixed`
- Descriptive names — avoid `df1`, `temp`, `x`, `obj` unless scope is trivially local
- One responsibility per function; prefer early validation with actionable error messages

### Architecture Rules

- All production logic lives in Kedro pipelines under `pipelines/src/`. Do not place heavy business logic in notebooks.
- Separate concerns strictly: ingestion → feature engineering → training → evaluation → serving.
- Reusable utilities (metrics, loaders, schemas) belong in `shared/`, not duplicated across pipelines.
- Exogenous variables are **core model inputs**, not optional add-ons.
- The application layer (Streamlit / FastAPI) is part of the product, not a presentation artifact.
- All dataset I/O must go through `catalog.yml` — no hardcoded paths inside nodes.

### Code Style

- Follow ruff configuration in `pyproject.toml` (line-length 88, isort, pyflakes, pycodestyle, pylint subset).
- Nodes must be **pure functions**: DataFrames and parameters in, DataFrames and artifacts out.
- Use type hints on all function signatures.

### What NOT To Do

- Do not redefine the methodological direction, scope boundaries, or hierarchy priorities.
- Do not add major dependencies without justification.
- Do not replace agreed project choices (Kedro, MLflow, model candidates) with speculative alternatives.
- Do not use `pip` directly — use `uv` for all package management.
- Do not commit data files, credentials, or `.env` files.

## Configuration

- **OmegaConfigLoader** (see `settings.py`)
- Base config: `conf/base/` — shared across environments
- Local overrides: `conf/local/` — gitignored, never committed
- Per-pipeline parameters: `conf/base/parameters_<pipeline_name>.yml`
- All datasets defined in `conf/base/catalog.yml`

## Architecture Style

**Local-first with cloud option**: must run cleanly in local development, but remain structurally ready for deployment to AWS (EC2, optional S3-backed storage). Docker supports reproducibility and portability.

## Key References

| Document | Location | Purpose |
|----------|----------|---------|
| Blueprint | `pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md` | Strategic direction, scope, methodology |
| Architecture | `pipelines/docs/architecture.md` | System design and data flow |
| Kedro proposal | `pipelines/docs/kedro_functional_logic_proposal.md` | Pipeline design rationale |
| Contributing | `CONTRIBUTING.md` | Full contribution guide and PR checklist |

Consult the **blueprint** before making architectural or methodological decisions.

## Contributing

### Branching

`main` → production-ready · `dev` → integration · feature branches → isolated development

```
<type>/<short-description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `hotfix`

Examples: `feat/monthly-catboost-training`, `fix/reconciliation-mint-weights`, `docs/update-architecture`

### Commits — Conventional Commits

```
<type>(scope): clear description
```

Examples:
```
feat(monthly): add catboost training pipeline with optuna tuning
fix(reconciliation): correct mint_shrink matrix inversion
refactor(shared): centralize forecast metric utilities
test(pipelines): add smoke test for data ingestion pipeline
docs(blueprint): update weekly layer scope definition
```

### Pull Requests

Title: `[TYPE] Short clear description` — types: `FEATURE`, `FIX`, `REFACTOR`, `DOCS`, `TEST`, `CHORE`

Every PR needs: summary, list of changes, validation notes. Keep PRs focused on one topic.

### CI/CD

Automated validation via **GitHub Actions** on PRs to `main` and `dev`: linting (ruff), tests (pytest), Docker build check.
