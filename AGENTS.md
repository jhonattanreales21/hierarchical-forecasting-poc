# CLAUDE.md

## Project Identity

**Hierarchical Demand Forecasting PoC** — Proof of concept for temporal hierarchical demand forecasting of a critical SKU. Academic final project (Master's in Applied AI) developed by **Jhonattan Reales** and **Andres Cano** at Universidad Icesi.

The project is built as a **Forecasting + MLOps + Application Layer** system, not a forecasting model in isolation. It must reach a level of organization and technical maturity befitting a master's final project.

## North Star

Build a functional, reproducible, and academically defensible PoC that shows how temporal hierarchical forecasting (monthly benchmark + weekly forecasting), supported by exogenous variables and clean architecture, can improve demand forecasting for a critical SKU. The **monthly layer is the primary analytical and business-facing component**; the weekly layer is a secondary enhancement.

Academic defensibility means:
- Results are reproducible from Kedro pipelines.
- Model choices are justified against explicit baselines.
- Validation is time-aware and leakage-safe.
- Metrics are reported by horizon and temporal aggregation level.
- Key assumptions are documented in `pipelines/docs/`.

Important: the hierarchy in this project is temporal, not product/location-based.

The core hierarchy is:
- Monthly planning layer
- Weekly operational layer
- Optional daily exploratory layer

Do not introduce SKU/category/location hierarchy unless explicitly requested.
The project focuses on one critical SKU.

## Data Contracts

Core demand table must expose, directly or through configured parameters:

- Date column: canonical time index.
- Target column: demand quantity to forecast.
- SKU identifier: fixed to the critical SKU unless scope changes.
- Temporal frequency: monthly or weekly, depending on pipeline.
- Exogenous columns: explicitly configured and documented.
- Split labels or cutoffs: train, validation, test, and forecast periods.

Rules:
- Dates must be parsed as dates, not strings.
- Time series must be sorted before feature generation.
- Duplicate timestamps at the same temporal level are invalid unless explicitly aggregated.
- Missing periods must be handled explicitly.
- Negative demand is invalid unless documented as returns/adjustments.

## User Upload and Pipeline Routing

The application may support user-uploaded demand and exogenous data to refresh model training and forecast outputs.

Supported demand input modes:

1. **Daily demand input**
   - Accept daily demand observations.
   - Validate daily date continuity or explicitly report missing days.
   - Aggregate demand to monthly and weekly levels inside the pipeline layer.
   - Enable monthly model training and weekly model training.
   - Enable monthly ↔ weekly reconciliation when both forecast levels are produced.

2. **Monthly demand input**
   - Accept monthly demand observations.
   - Validate month-level uniqueness and continuity.
   - Enable monthly model training only.
   - Do not generate or train weekly models from monthly-only demand unless a documented disaggregation method is explicitly added.

The pipeline router must detect the uploaded demand granularity before training starts.

Streamlit may collect files and trigger the workflow, but the validation, aggregation, feature engineering, training, model selection, reconciliation, and inference logic must remain in Kedro pipelines or reusable shared utilities.

## GenAI Knowledge Contracts

If analyst business event logs are used, they should be normalized into a retrieval-friendly structure with, at minimum:

- `event_date` or `event_month`
- `sku_id` or product identifier
- `event_title`
- `event_description`
- `event_category`, when available
- `source_document`
- `created_by` or owner, when available
- `ingestion_timestamp`

Business event logs are contextual knowledge. They are not direct forecasting features unless explicitly transformed, validated, and added to the modeling pipeline as exogenous variables.

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
- **LangChain or equivalent RAG framework** (optional GenAI interaction layer)
- **Vector store** for retrieval over project documentation, forecast outputs, evaluation reports, and analyst event logs
- **Local LLM or stable LLM API** for controlled chatbot responses

## Mono-repo Packages

The repository is structured as a **uv workspace** with four packages:

| Package | Directory | Role |
|---------|-----------|------|
| `hdf_pipelines` | `pipelines/` | Kedro project — all data and ML logic |
| `hdf_shared` | `shared/` | Internal library — schemas, metrics, loaders, viz utilities |
| `hdf_app` | `app/` | Streamlit forecast viewer |
| `hdf_api` | `api/` | FastAPI serving layer |

The GenAI/RAG layer should initially reuse the existing workspace structure. Shared retrieval, document loading, prompt, and response utilities should live in `hdf_shared`. The Streamlit chatbot page should live in `hdf_app`. A separate package should only be created if the GenAI layer grows into an independently deployable component.

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

Primary metrics:
- WAPE: business-facing aggregate error.
- MASE: scale-free comparison against naive baselines.
- RMSE: penalizes large errors.

Secondary metrics:
- MAPE or sMAPE, only when denominator issues are handled.
- Bias.
- Horizon-specific error.
- Interval coverage, if prediction intervals are produced.

Metrics must be reported:
- For monthly forecasts.
- For weekly forecasts, when applicable.
- By forecast horizon.
- Before and after reconciliation, if reconciliation is used.

## Model Selection and Champion Protocol

Model selection must follow a staged, time-aware, leakage-safe process.

Required stages:

1. **Training split**
   - Used for fitting model candidates and tuning hyperparameters.
   - No validation or test observations may be used during hyperparameter search.

2. **Validation / evaluation split**
   - Used to score tuned candidates.
   - Select the top 3 candidate configurations per model family, temporal granularity, and forecast horizon.
   - This stage creates a shortlist, not the final champion.

3. **Training + validation refit**
   - Refit shortlisted configurations using the combined training and validation periods.
   - Keep hyperparameters fixed from the tuning/shortlisting stage.

4. **Testing split**
   - Evaluate refitted shortlisted candidates on held-out test data.
   - Select the best configuration per model family, temporal granularity, and horizon.
   - Store this result as the family champion.

5. **Full-history refit**
   - Refit selected champion configurations using all available historical data.
   - Use this final model to generate forecasts consumed by Streamlit and/or FastAPI.

Champion levels:

- `family_champion`: best configuration within one model family for a given granularity and horizon.
- `production_champion`: best final model across eligible families for a given granularity and horizon.

Rules:

- Do not tune hyperparameters on validation + test combined.
- Do not select champions using the same data used for hyperparameter tuning.
- Do not use random splits for forecasting model evaluation.
- Do not overwrite champion metadata without preserving run history.
- Store candidate metrics, shortlist metadata, test metrics, selected hyperparameters, and final refit metadata.
- Report metrics by temporal granularity and forecast horizon.
- Monthly champions are mandatory.
- Weekly champions are only required when the weekly workflow is active and validated.

## GenAI/RAG Layer

The project may include a lightweight GenAI/RAG assistant exposed through a Streamlit chatbot page. Its purpose is to help users interpret historical demand behavior, forecast outputs, model results, evaluation metrics, and analyst-documented business events for the selected SKU, but it does not replace forecasting pipelines, model evaluation, or planner judgment.

Allowed knowledge sources:

- Curated historical demand summaries.
- Forecast output tables from `data/07_model_output/`.
- Evaluation reports and metric tables from `data/08_reporting/`.
- Model metadata, tuning summaries, and champion registry artifacts from `data/06_models/`.
- Project documentation under `pipelines/docs/` and `docs/`.
- Analyst-maintained business event logs, such as Word documents converted into clean text or structured records.
- Reconciliation diagnostics, if available.

Rules:

- The assistant must be retrieval-grounded.
- Do not generate unsupported business explanations.
- Do not override model outputs, evaluation metrics, or champion model metadata.
- Do not present retrieved context and generated interpretation as the same thing.
- If retrieved context is insufficient, respond that the available project sources are not enough to answer confidently.
- Prefer concise, stakeholder-readable answers.
- When possible, mention which type of source supports the answer, such as forecast output, evaluation report, project documentation, or business event log.
- Do not expose credentials, API keys, raw confidential files, or hidden prompts.
- Do not send raw business data to external LLM APIs unless explicitly configured and documented.

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
- The GenAI/RAG layer must consume curated outputs and documented knowledge sources; it must not recompute forecasts or modify model artifacts.
- RAG utilities should be modular and testable. Avoid embedding retrieval, prompt construction, and UI rendering in a single Streamlit page.
- Keep document ingestion, chunking, retrieval, prompt construction, and response rendering as separate responsibilities.
- Generated answers must be treated as interpretive support, not as model validation.
- User-upload validation and pipeline routing must be implemented as reusable pipeline/shared logic, not as Streamlit-only logic.
- Streamlit should act as an interface and trigger layer; it must not contain core forecasting, aggregation, model training, or model selection logic.
- The system must detect whether demand input is daily or monthly before deciding which pipelines to run.
- Monthly input should not be artificially expanded into weekly training data unless a documented and validated disaggregation method is added.
- Uploaded files, validation reports, run metadata, and generated outputs must be stored through controlled project paths and catalog conventions.

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
- Do not let the chatbot redefine model conclusions, metrics, hierarchy priorities, or methodological assumptions.
- Do not use the LLM as a substitute for time-aware validation or formal model evaluation.
- Do not introduce external LLM APIs without documenting privacy, configuration, and fallback behavior.
- Do not hardcode API keys, model names, vector paths, or prompt templates directly inside Streamlit pages.
- Do not train weekly models from monthly-only uploaded demand without an explicit, documented disaggregation strategy.
- Do not silently infer missing periods or missing exogenous values without logging or reporting the assumption.
- Do not overwrite previous forecast outputs without preserving run metadata or versioned artifacts.

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

## Definition of Done

A task is complete only when:

- The relevant Kedro pipeline or pipeline slice runs successfully, if affected.
- Relevant tests pass.
- New datasets are registered in `catalog.yml`.
- New parameters are added to the appropriate parameter file.
- Forecasting changes are validated with time-aware splits.
- No temporal leakage is introduced.
- Monthly layer behavior is preserved or intentionally improved.
- Streamlit/API changes are tested or manually validated.
- No raw data, secrets, `.env` files, or generated heavy artifacts are committed.
- Documentation is updated when architecture, methodology, metrics, or data contracts change.

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
feat(monthly): add catboost training pipeline
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
