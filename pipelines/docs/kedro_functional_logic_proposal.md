# Kedro Pipeline Proposal and Functional Logic

## Overview

The Kedro architecture for this POC should prioritize **clear stage separation**, **modularity**, and **controlled flexibility**, without over-fragmenting the project into too many top-level pipelines. The goal is to keep the solution professional and scalable while remaining practical for an academic-industrial proof of concept.

The recommended approach is to define a small set of **stage-oriented pipelines**, and then use **parameters, modular pipelines, and namespaces** to specialize behavior by temporal granularity (`monthly`, `weekly`) and model family (`prophet`, `catboost`, `sarimax`).

This avoids creating an excessive number of independent pipelines such as `train_monthly_prophet`, `train_weekly_prophet`, `train_monthly_catboost`, etc., while still preserving clean separation of responsibilities.

---

## Design Principles

### 1. Stage-based architecture over model-based fragmentation
Top-level pipelines should represent **major lifecycle stages** of the forecasting solution, not every possible combination of model and granularity.

### 2. Monthly and weekly are first-class analytical views
The project has two main modeling views:
- **Monthly forecasting**, mainly as a strategic benchmark and business-aligned forecasting layer
- **Weekly forecasting**, as the operational anchor of the solution

These should be reflected in the feature engineering and model training logic, but not necessarily as fully duplicated architecture.

### 3. Reuse through modular pipelines and namespaces
Whenever possible, the same logical pipeline should be reused with different parameters or namespaces. This keeps the codebase easier to maintain and improves consistency across workflows.

### 4. Training, champion selection, reconciliation, and inference should remain conceptually distinct
These are separate responsibilities and should not be mixed into one large pipeline. This improves readability, reproducibility, and future extensibility.

### 5. Inference logic should be reusable for both periodic and on-demand predictions
The same inference core should support both:
- scheduled forecasting runs
- user-triggered or API-triggered prediction requests

---

## Proposed Top-Level Pipelines

The following top-level pipelines are recommended for the Kedro project:

1. `data_ingestion`
2. `feature_engineering_monthly`
3. `feature_engineering_weekly`
4. `model_input_preparation`
5. `train_monthly`
6. `train_weekly`
7. `model_selection`
8. `reconciliation`
9. `forecast_inference`

This structure keeps the architecture focused, understandable, and aligned with the forecasting lifecycle.

---

## Pipeline Responsibilities

## 1. `data_ingestion`

### Purpose
Load, clean, standardize, and consolidate the raw data needed by the forecasting system.

### Responsibilities
- Read raw demand data
- Read exogenous variables
- Apply schema normalization
- Perform basic cleaning and validation
- Align temporal fields and keys
- Produce curated base datasets for downstream feature engineering

### Expected outputs
- clean demand dataset
- clean exogenous dataset
- unified base input tables for monthly and weekly processing

### Notes
This pipeline should remain focused on **data readiness**, not on feature creation or modeling logic.

---

## 2. `feature_engineering_monthly`

### Purpose
Generate the monthly-level analytical dataset used for monthly model training, evaluation, and forecasting.

### Responsibilities
- Aggregate data to monthly granularity
- Create lag-based features
- Create rolling statistics
- Create calendar-based features
- Integrate monthly exogenous signals
- Produce a model-ready monthly base table

### Expected outputs
- monthly feature table
- monthly modeling base dataset

### Notes
This pipeline should contain only logic relevant to monthly forecasting.

---

## 3. `feature_engineering_weekly`

### Purpose
Generate the weekly-level analytical dataset used for weekly model training, evaluation, and forecasting.

### Responsibilities
- Aggregate data to weekly granularity
- Create lag-based features
- Create rolling statistics
- Create seasonality-oriented features
- Integrate weekly exogenous signals
- Produce a model-ready weekly base table

### Expected outputs
- weekly feature table
- weekly modeling base dataset

### Notes
This pipeline is especially important because the weekly view is expected to be the operational anchor of the solution.

---

## 4. `model_input_preparation`

### Purpose
Prepare the exact model inputs required for training, validation, backtesting, test evaluation, and final fitting.

### Responsibilities
- Build temporal train/validation/test splits
- Generate backtesting windows
- Prepare model-specific data structures
- Support both full-data and partial-data training schemes
- Keep the split logic centralized and reproducible

### Typical use cases
- prepare train/validation/test datasets for CatBoost
- prepare Prophet-compatible datasets
- prepare SARIMAX-compatible endogenous/exogenous arrays
- prepare rolling-origin windows for evaluation

### Expected outputs
Examples may include:
- `monthly_catboost_train`
- `monthly_catboost_validation`
- `monthly_catboost_test`
- `weekly_prophet_train`
- `weekly_prophet_validation`
- `weekly_sarimax_test`
- backtesting fold definitions
- metadata describing split boundaries and horizon settings

### Notes
This pipeline should be treated as a **shared preparation layer**, not as a collection of isolated mini-pipelines. The detailed output datasets can vary by model and granularity, but the architectural role should remain unified.

---

## 5. `train_monthly`

### Purpose
Train and tune monthly forecasting models across the selected model families.

### Responsibilities
- receive prepared monthly datasets
- run training and hyperparameter tuning
- evaluate candidates on validation data
- track metrics and artifacts
- save the best candidate per model family

### Supported model families
- Prophet
- CatBoost
- SARIMAX

### Expected outputs
- trained monthly candidate models
- tuning results
- validation metrics
- experiment artifacts
- monthly model comparison summaries

### Recommended implementation approach
Use **modular pipelines or namespaces** per model family within the monthly training stage, instead of creating a fully independent top-level pipeline for each model.

Example conceptual structure:
- `train_monthly.prophet`
- `train_monthly.catboost`
- `train_monthly.sarimax`

### Notes
This pipeline should identify the best monthly candidate of each model family, but not yet define the final production champion.

---

## 6. `train_weekly`

### Purpose
Train and tune weekly forecasting models across the selected model families.

### Responsibilities
- receive prepared weekly datasets
- run training and hyperparameter tuning
- evaluate candidates on validation data
- track metrics and artifacts
- save the best candidate per model family

### Supported model families
- Prophet
- CatBoost
- SARIMAX

### Expected outputs
- trained weekly candidate models
- tuning results
- validation metrics
- experiment artifacts
- weekly model comparison summaries

### Recommended implementation approach
As with the monthly pipeline, prefer **modular reuse and namespaces** over many independent pipelines.

Example conceptual structure:
- `train_weekly.prophet`
- `train_weekly.catboost`
- `train_weekly.sarimax`

### Notes
This pipeline should mirror the monthly training logic while respecting weekly-specific features, horizons, and evaluation settings.

---

## 7. `model_selection`

### Purpose
Evaluate the best trained candidates on held-out test data and select the final champion models for forecasting use.

### Responsibilities
- load the best candidate of each family from training/tuning
- evaluate candidates on test data
- compare models under final selection criteria
- select champion models by granularity and horizon
- persist champion metadata for downstream use

### Expected outputs
- monthly champion model(s)
- weekly champion model(s)
- test-set comparison report
- final selection summaries
- champion registry or metadata artifact

### Recommended selection logic
Champion selection may be defined:
- per temporal granularity
- per forecast horizon
- per primary metric
- with secondary tie-breakers if needed

A practical setup would be:
- monthly champion for each target horizon
- weekly champion for each target horizon

### Notes
This pipeline is an important methodological layer because it separates:
- tuning and validation
from
- final unbiased model selection on test data

---

## 8. `reconciliation`

### Purpose
Apply temporal hierarchical reconciliation to the selected forecasts, ensuring consistency across forecasting levels.

### Responsibilities
- load monthly and weekly forecast outputs
- reconcile forecasts across temporal hierarchy
- support at least two reconciliation methods
- generate reconciled forecast tables
- retain metadata about the method used

### Expected outputs
- reconciled monthly forecasts
- reconciled weekly forecasts
- reconciliation diagnostics
- method-specific artifacts or summaries

### Notes
This pipeline should remain independent because reconciliation is not just a post-processing detail; it is one of the key methodological elements of the solution.

---

## 9. `forecast_inference`

### Purpose
Generate final predictions using champion models, for either scheduled forecasting or on-demand inference.

### Responsibilities
- load latest curated inputs
- apply the same preprocessing logic required for inference
- load monthly and weekly champion models
- produce forecast outputs
- optionally pass outputs to reconciliation
- save prediction tables for downstream consumption

### Expected outputs
- monthly forecast CSV/table
- weekly forecast CSV/table
- reconciled forecast outputs
- optional confidence intervals or metadata
- outputs ready for dashboard or API consumption

### Supported execution modes
- **Periodic batch mode**: scheduled production-like forecast generation
- **On-demand mode**: triggered by a user request, uploaded file, or API call

### Notes
The core inference logic should be reused in both scenarios. Only the trigger and input source should change.

---

## Recommended Functional Flow

The high-level functional flow should be:

`data_ingestion -> feature_engineering_monthly / feature_engineering_weekly -> model_input_preparation -> train_monthly / train_weekly -> model_selection -> reconciliation -> forecast_inference`

This sequence supports both experimentation and production-like execution in a clean and explainable way.

---

## Recommended Use of Modular Pipelines and Namespaces

To keep the Kedro project elegant and maintainable, namespaces can be used to organize repeated training logic.

Examples:
- `monthly.prophet`
- `monthly.catboost`
- `monthly.sarimax`
- `weekly.prophet`
- `weekly.catboost`
- `weekly.sarimax`

This allows the codebase to stay modular without multiplying top-level pipelines unnecessarily.

A good rule is:
- use **top-level pipelines** for major lifecycle stages
- use **modular pipelines and namespaces** for repeated logic across granularity/model variants

---

## What Should Not Be Over-Engineered

The following should be avoided in the first implementation:

### 1. Too many top-level pipelines
Avoid creating one independent top-level pipeline for every model and granularity combination unless the project later grows enough to justify it.

### 2. Mixing training and final champion selection
Training/tuning and final selection should remain separate stages.

### 3. Mixing reconciliation directly into training
Reconciliation should happen after strong candidate forecasts already exist.

### 4. Creating separate inference implementations for batch and on-demand use
The inference core should be shared.

### 5. Treating every dataset variation as an architectural component
Datasets like `train_catboost`, `val_prophet`, or `test_sarimax` are important outputs, but they should not define the architecture by themselves.

---

## Final Recommendation

The Kedro design should remain **stage-oriented, modular, and parameter-driven**. The architecture does not need a large number of rigid pipelines to look professional. On the contrary, a more refined and mature design is one where:

- the major forecasting stages are clearly separated,
- monthly and weekly workflows are explicitly supported,
- repeated logic is reused through modular pipelines and namespaces,
- champion selection and reconciliation are treated as first-class stages,
- inference is designed from the beginning to support both periodic and on-demand forecasting.

This structure is strong enough for the POC, clear enough for documentation, and scalable enough to support a future FastAPI + Streamlit serving layer.