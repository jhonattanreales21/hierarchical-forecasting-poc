# Kedro Pipeline Proposal and Functional Logic

## Overview

The Kedro architecture for this POC should prioritize **clear stage separation**, **modularity**, and **controlled flexibility**, without over-fragmenting the project into too many top-level pipelines. The goal is to keep the solution professional and scalable while remaining practical for an academic-industrial proof of concept.

The recommended approach is to define a small set of **stage-oriented pipelines**, and then use **parameters, modular pipelines, and namespaces** to specialize behavior by temporal granularity (`monthly`, `weekly`) and model family (`prophet`, `catboost`, `sarimax`).

This avoids creating an excessive number of independent pipelines such as `train_monthly_prophet`, `train_weekly_prophet`, `train_monthly_catboost`, etc., while still preserving clean separation of responsibilities.

---

## Design Principles

### 1. Stage-based architecture over model-based fragmentation
Top-level pipelines should represent **major lifecycle stages** of the forecasting solution, not every possible combination of model and granularity.

### 2. Monthly-first forecasting architecture

The project follows a **monthly-first forecasting architecture**.

This means that the monthly layer is the primary analytical, modeling, evaluation, reporting, and stakeholder-facing layer of the solution. It is the level that best reflects the current business planning process and the main horizon at which forecast quality will be judged.

The weekly layer remains important, but it should be treated as an operational enhancement. Its role is to provide additional short-term granularity, support the 14-week planning context, and enrich the monthly forecast with operational detail. However, it must not redefine the main modeling objective or compromise the quality, completeness, or clarity of the monthly workflow.

The daily layer, if implemented, should be treated as an optional downstream extension or allocation layer, not as a core modeling target.

Practical implications:

- monthly forecasting receives the highest modeling priority;
- monthly evaluation receives the highest reporting emphasis;
- monthly outputs are the main artifacts exposed to stakeholders;
- weekly outputs should complement and remain coherent with the monthly planning view;
- daily outputs should only be included when they can be produced pragmatically and safely;
- no weekly or daily workflow should delay, weaken, or overcomplicate the monthly MVP.

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
The weekly view is an operational enhancement that provides short-term granularity, but the monthly layer remains the primary modeling, evaluation, and business-facing layer.

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
- backtesting fold definitions
- metadata describing split boundaries and horizon settings

### Notes
This pipeline should be treated as a **shared preparation layer**, not as a collection of isolated mini-pipelines. The detailed output datasets can vary by model and granularity, but the architectural role should remain unified.

---

## 5. `train_monthly`

### Purpose
Train and tune monthly forecasting models across the selected model families. This is the primary modeling pipeline of the project and the main source of business-facing forecast outputs.

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
This pipeline should identify the best monthly candidate of each model family, but not yet define the final production champion. The monthly training workflow is the core modeling workflow of the project. It should receive the highest priority in terms of modeling effort, evaluation rigor, artifact completeness, and stakeholder-facing reporting.

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
- weekly champion model(s), when the weekly workflow is active
- validation metrics by model family, horizon, and granularity
- test metrics by model family, horizon, and granularity
- WAPE / MASE / RMSE comparison tables
- secondary diagnostics tables, including bias and horizon-specific error
- raw vs reconciled metric comparison, when reconciliation is applied
- model ranking report
- final selection summaries
- champion registry or metadata artifact
- model selection audit report

### Recommended selection logic
Champion selection may be defined:
- per temporal granularity
- per forecast horizon
- per primary metric
- with secondary tie-breakers if needed

A practical setup would be:
- monthly champion for each target horizon
- weekly champion for each target horizon

### Champion selection protocol

The `model_selection` pipeline must implement a staged, time-aware, and leakage-safe champion selection protocol. Its purpose is to avoid selecting models directly from the final test period and to ensure that final application forecasts are generated from models trained with the maximum available historical information.

The protocol should be applied independently by:

- temporal granularity: `monthly` and, when active, `weekly`;
- model family: `prophet`, `catboost`, `sarimax`;
- forecast horizon: for example 3, 6, and 12 months for the monthly workflow, and the configured weekly horizon when the weekly workflow is active.

#### Stage 1 — Time-based data split

The available historical data must be split into ordered temporal blocks:

- `training`: used to fit model candidates and tune hyperparameters;
- `validation` / `evaluation`: used to compare tuned candidates and shortlist the best configurations;
- `testing`: held out until the final comparison stage.

Random splits must not be used because they break the temporal structure of the forecasting problem and may introduce leakage.

#### Stage 2 — Hyperparameter tuning and validation shortlist

For each model family, temporal granularity, and forecast horizon, hyperparameter tuning is performed using the training period.

Each tuned configuration is evaluated on the validation period. Based on the official metrics and relevant diagnostics, the pipeline should select a controlled shortlist of candidate configurations.

The default shortlist rule is:

- select the top 3 candidate configurations;
- per model family;
- per temporal granularity;
- per forecast horizon.

This stage produces candidate configurations, not final champions.

#### Stage 3 — Refit on training + validation and evaluate on test

The shortlisted configurations are refitted using the combined `training + validation` data.

These refitted candidates are then evaluated on the held-out testing period. This test-period evaluation is used to select the best configuration within each model family, temporal granularity, and forecast horizon.

The output of this stage is the `family champion`.

A family champion is the best validated configuration within a specific model family for a specific granularity and horizon.

Examples:

- best monthly Prophet configuration for a 3-month horizon;
- best monthly CatBoost configuration for a 6-month horizon;
- best monthly SARIMAX configuration for a 12-month horizon;
- best weekly Prophet configuration for the configured weekly horizon, when the weekly workflow is active.

#### Stage 4 — Select production champion

After family champions are selected, the pipeline may compare them across eligible model families to define a `production champion` for each granularity and horizon.

The project may therefore distinguish between two champion levels:

- `family champion`: best configuration within a model family, granularity, and horizon;
- `production champion`: final selected model across all eligible families for a given granularity and horizon.

If the application exposes multiple model options, family champions may be made available for comparison. If the application exposes one official forecast, the production champion must be selected using predefined metrics and documented tie-breaker criteria.

#### Stage 5 — Final refit for application forecasts

Once the champion configuration is selected, the model should be refitted using all available historical data approved for final training.

This final refit is used to generate the forecasts consumed by the application layer.

The final application forecast should not be generated from a model trained only on the original training split if validation and testing data are already available and approved for final refitting.

### Evaluation metrics and selection criteria

Champion selection should not depend on a single metric. Forecasting models may behave differently depending on whether the priority is aggregate accuracy, robustness against naive baselines, large-error penalization, bias control, or horizon stability.

For this reason, the `model_selection` pipeline must evaluate each candidate using a standard metric set:

- `WAPE`: primary business-facing aggregate error metric.
- `MASE`: scale-free metric used to compare performance against naive forecasting baselines.
- `RMSE`: error metric used to penalize large forecast misses.

These metrics must be computed consistently for each:

- temporal granularity: `monthly` and, when active, `weekly`;
- model family: `prophet`, `catboost`, `sarimax`;
- forecast horizon: for example 3, 6, and 12 months in the monthly workflow;
- evaluation stage: validation/evaluation and testing;
- forecast version: raw and reconciled, when reconciliation is applied.

The final champion should be selected using predefined criteria based on the official metric set, not by optimizing one metric in isolation.

A recommended decision rule is:

1. Use `WAPE` as the primary business-facing ranking metric.
2. Use `MASE` to verify that the candidate improves meaningfully over naive baselines.
3. Use `RMSE` to detect candidates with unacceptable large forecast errors.
4. Use secondary diagnostics as tie-breakers or rejection criteria.

Secondary diagnostics may include:

- forecast bias;
- horizon-specific degradation;
- stability across forecast horizons;
- performance before and after reconciliation;
- business plausibility;
- interpretability and operational simplicity;
- consistency with future exogenous assumptions.

A model should not be selected as champion only because it has the lowest value for one metric if it performs poorly on other critical diagnostics or produces forecasts that are not business-plausible.

### Expected champion artifacts

The `model_selection` pipeline should persist enough metadata to make the selection process auditable and reproducible.

Expected artifacts include:

- validation ranking by model family, granularity, and horizon;
- shortlisted top 3 configurations per family, granularity, and horizon;
- test evaluation results for shortlisted candidates;
- selected family champion metadata;
- selected production champion metadata, when applicable;
- champion registry artifact;
- selected hyperparameters;
- model family;
- temporal granularity;
- forecast horizon;
- training, validation, and test cutoffs;
- validation and test metrics;
- MLflow run identifiers, when available;
- notes about future exogenous assumptions.

### Notes

This pipeline is a core methodological layer of the project. It must preserve the distinction between:

- hyperparameter tuning on training data;
- candidate shortlisting on validation data;
- final unbiased comparison on test data;
- final refit using all available historical data for application forecast generation.

This separation is required for academic defensibility, reproducibility, and leakage-safe evaluation.

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
Reconciliation should preserve the monthly forecast as the controlling business-facing layer. Weekly forecasts should be adjusted or interpreted in a way that remains coherent with the monthly planning view.

The primary reconciliation target is monthly ↔ weekly coherence. Full monthly ↔ weekly ↔ daily reconciliation is optional and should be treated as a secondary extension.

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

`data_ingestion -> feature_engineering_monthly / feature_engineering_weekly -> model_input_preparation -> train_monthly / train_weekly -> model_selection -> forecast_inference_raw -> reconciliation -> publish_forecast_outputs`

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