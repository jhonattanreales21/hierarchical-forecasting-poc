# CLAUDE.md

## Project Identity

**Hierarchical Demand Forecasting PoC** -- Proof of concept for temporal hierarchical demand forecasting of a critical SKU. This is an academic final project (Master's in Applied AI) developed by **Jhonattan Reales** and **Andres Cano**.

The project is built as a Forecasting + MLOps + Application Layer system, not a forecasting model in isolation. It must reach a high level of organization and technical maturity befitting a master's final project.

## North Star

Build a functional, reproducible, and academically defensible PoC that shows how temporal hierarchical forecasting (monthly benchmark + weekly forecasting), supported by exogenous variables and clean architecture, can improve demand forecasting for a critical SKU. The monthly layer is the primary analytical and business-facing component; the weekly layer is a secondary enhancement.

## Tech Stack

- **Python** >= 3.10
- **Kedro** ~1.3.1 (pipeline orchestration and project structure)
- **uv** (package and environment management -- use `uv` instead of `pip`)
- **Pandas** (data manipulation)
- **Scikit-learn / Statsmodels / Prophet / CatBoost** (modeling)
- **Nixtla NeuralForecast (N-HiTS)** (optional exploratory model)
- **MLflow** (experiment tracking and artifact versioning)
- **Streamlit** (forecast viewer application)
- **FastAPI** (serving layer)
- **Docker** (reproducibility and deployment)
- **pytest** (testing)
- **ruff** (linting and formatting)

## Common Commands

```bash
# Environment setup
uv sync                              # Install all dependencies
uv pip install -r requirements.txt   # Alternative: install from requirements.txt

# Run Kedro pipelines
kedro run                            # Run the default (all) pipeline
kedro run --pipeline=<name>          # Run a specific pipeline

# Kedro utilities
kedro catalog list                   # List all datasets in the catalog
kedro pipeline list                  # List registered pipelines
kedro viz run                        # Launch Kedro-Viz pipeline visualizer

# Testing
pytest                               # Run all tests (with coverage)
pytest tests/test_run.py             # Run Kedro bootstrap test

# Linting
ruff check src/                      # Lint source code
ruff format src/                     # Format source code

# Kedro pipeline scaffolding
kedro pipeline create <pipeline_name>  # Create a new pipeline module
```

## Project Structure

```
conf/
  base/                     # Shared configuration (catalog, parameters)
    catalog.yml             # Data catalog definitions
    parameters.yml          # Global parameters
    parameters_<pipeline>.yml  # Per-pipeline parameters
  local/                    # Local overrides and credentials (gitignored)
  logging.yml               # Logging configuration

data/
  01_raw/                   # Raw input data (gitignored)
  02_intermediate/          # Cleaned/preprocessed data
  03_primary/               # Domain-level model-ready data
  04_feature/               # Feature-engineered datasets
  05_model_input/           # Final model input tables
  06_models/                # Trained model artifacts
  07_model_output/          # Predictions and forecast outputs
  08_reporting/             # Evaluation reports, plots

src/hdf_pipelines/
  __init__.py               # Package version
  __main__.py               # CLI entry point
  settings.py               # Kedro project settings (OmegaConfigLoader)
  pipeline_registry.py      # Auto-discovers and registers all pipelines
  pipelines/                # All Kedro pipeline modules go here
    <pipeline_name>/
      __init__.py
      nodes.py              # Pure functions (business logic)
      pipeline.py           # Pipeline wiring (node -> pipeline)

tests/                      # pytest tests
docs/                       # Project documentation and blueprint
notebooks/                  # Kedro-coupled notebooks; launch with `kedro jupyter notebook` to get catalog injected
```

> **Notebook placement rule:** notebooks that access the Kedro `DataCatalog` (e.g., loading pipeline outputs) live in `pipelines/notebooks/` and must be launched with `kedro jupyter notebook` from `pipelines/`. General-purpose EDA notebooks unrelated to the pipeline layer live in the root-level `notebooks/` folder.

## Temporal Hierarchy

The forecasting hierarchy is strictly temporal:

1. **Monthly** -- **Primary decision layer and main modeling priority.** Receives the greatest evaluation effort, attention from stakeholders, and narrative focus. Central benchmark aligned with the business decision horizon.
2. **Weekly** -- Secondary enhancement layer. Valuable operational complement, but must not compromise the quality of the monthly layer.
3. **Daily** -- Low-priority exploratory extension. Pragmatic disaggregation only if feasible; not a core deliverable.

Primary coherence target: monthly <-> weekly. Full monthly <-> weekly <-> daily is desirable but treated as secondary scope.

## Model Candidates

- **SARIMAX** -- Structured statistical baseline
- **Prophet** -- Existing benchmark
- **CatBoost** -- Main tabular candidate with exogenous variables
- **N-HiTS (Nixtla)** -- Optional neural benchmark

## Evaluation

Metrics: **MAPE**, **RMSE**, **MASE** (primary). Bias, interval coverage, horizon-specific error (secondary).

Validation must be time-aware: rolling-origin or equivalent time-based backtesting. Never use random splits for time series.

## Development Guidelines

### Naming and Structure

- Classes: `PascalCase` (e.g., `Matcher`, `MatchResult`, `ExperimentAnalysis`)
- Methods and variables: `snake_case`
- Private methods and attributes: prefixed with `_`
- Descriptive names — avoid `df1`, `temp`, `data2`, `x`, or `obj` unless the scope is extremely local and obvious
- Keep functions focused on one responsibility
- Prefer early validation and actionable error messages

### Architecture Rules

- All production logic lives in Kedro pipelines under `src/.../pipelines/`. Do not place heavy business logic in notebooks.
- Separate concerns: ingestion, feature engineering, training, evaluation, serving.
- Exogenous variables are core model inputs, not optional add-ons.
- The application layer (Streamlit/FastAPI) is part of the product, not a presentation artifact.

### Code Style

- Follow ruff configuration in pyproject.toml (line-length 88, isort, pyflakes, pycodestyle, pylint subset).
- Nodes must be pure functions: receive DataFrames/parameters in, return DataFrames/artifacts out.
- Use type hints on all function signatures.
- Prefer modular, readable, and testable code.

### What NOT To Do

- Do not redefine the methodological direction, scope boundaries, or hierarchy priorities.
- Do not add major dependencies without justification.
- Do not replace agreed project choices (Kedro, MLflow, the model candidates) with speculative alternatives.
- Do not break the Kedro data layer conventions (use catalog.yml for all dataset I/O).
- Do not use `pip` directly -- use `uv` for all package management.
- Do not commit data files, credentials, or `.env` files.

### Testing

Lightweight but real: unit tests for critical utility functions, especially feature generation, reconciliation logic, and evaluation helpers. Tests demonstrate engineering discipline.

## Configuration

- **OmegaConfigLoader** is the config loader (see settings.py)
- Base config: `conf/base/` -- shared across environments
- Local config: `conf/local/` -- overrides and credentials (gitignored)
- Parameters per pipeline: `conf/base/parameters_<pipeline_name>.yml`
- All datasets defined in `conf/base/catalog.yml`

## Architecture Style

The project is **local-first with cloud option**: must run cleanly in local development, but remain structurally ready for deployment to AWS (EC2, optional S3-backed storage). Docker supports reproducibility and portability.

## Blueprint Reference

The full strategic blueprint is at `pipelines/docs/demand_forecast_temporal_hierarchical_blueprint.md`. Consult it for detailed methodology, scope boundaries, success criteria, and business context before making architectural decisions.


## Contributing preferences

###  Branching Strategy

- `main` → Production-ready code
- `dev` → Integration branch
- Feature branches → Isolated development of new functionality

Branch naming convention:

```
<type>/<short-description>
```

Accepted types:
- `feat/*` — New functionality
- `fix/*` — Bug fixes
- `refactor/*` — Internal code restructuring without behavior changes
- `docs/*` — Documentation changes
- `test/*` — Test additions or restructuring
- `chore/*` — Maintenance tasks, tooling, config updates
- `hotfix/*` — Urgent production or release-related fixes

Examples: `feat/exact-matching-1k`, `fix/balance-table-categorical-bug`, `docs/readme-overview-update`

Branch names should be short, descriptive, and written in lowercase using hyphens.

###  Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(optional-scope): clear description
```

Accepted commit types:
- `feat` — New functionality
- `fix` — Bug fix
- `refactor` — Internal restructuring
- `docs` — Documentation
- `test` — Tests
- `chore` — Tooling, dependencies, config
- `perf` — Performance improvements

Examples:
```
feat(matching): add exact matching without replacement
fix(statistics): correct pooled variance in smd calculation
refactor(data): centralize panel validation helpers
docs(readme): rewrite overview and examples
test(rct): add coverage for clustered ols estimation
```

Commit messages should describe the actual change, not the intention to work on something.

###  Pull Requests

**PR Title format:**

```
[<TYPE>] <Short clear description>
```

PR types: `FEATURE`, `FIX`, `REFACTOR`, `DOCS`, `TEST`, `CHORE`

Examples:
```
[FEATURE] Add exact matcher fit and match flow
[FIX] Correct balance table output for binary variables
[DOCS] Improve README overview and usage examples
```

**PR Description:**

Every PR should include, at minimum:

- A short summary of the change
- The main changes introduced
- How the change was validated
- Notes for reviewers when relevant

Recommended structure:

```md
## Summary
Brief description of the change and why it is needed.

## Changes
- 
- 

## Validation
Describe how the changes were validated.

## Notes for reviewers
Optional reviewer guidance.
```

PRs should remain focused. Avoid mixing unrelated refactors, docs updates, and new features in the same PR unless they are tightly coupled.

###  Testing

- Unit tests required for core logic
- Integration tests for end-to-end workflows
- Test coverage priorities:
  - Public standalone functions
  - Public classes and result objects
  - Validation logic
  - Error paths for invalid inputs
  - Key statistical or matching behaviors

###  CI/CD

- Automated validation on PRs to `main` and `dev`
- Checks: `isort`, `black`, `pylint`, `ruff` via Azure Pipelines
- Build of installable package (.whl)
- Publication or deployment of .whl into Databricks shared folder or Azure Artifacts
