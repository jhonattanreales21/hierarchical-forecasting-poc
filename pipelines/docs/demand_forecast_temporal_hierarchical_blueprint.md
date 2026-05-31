# Demand Forecast through Temporal Hierarchical Modelling Blueprint

## 1. Purpose

This document is the official blueprint for the project. It is intentionally written to be easy to read by both humans and machines. Its purpose is to capture the strategic direction, scope, architecture, implementation logic, and development guardrails of the initiative in a single reference.

The project should be understood as a **Forecasting + MLOps + Application Layer** proof of concept rather than as a forecasting model in isolation.

## 2. Project Overview

The project aims to build a functional proof of concept to improve demand forecast accuracy for a critical SKU through a temporal hierarchical forecasting approach supported by exogenous variables, modular pipelines, and a lightweight but real MLOps foundation.

The intended solution moves beyond a process dominated by aggregated statistical forecasting and manual planner adjustments. The target direction is a structured system built around:

- a monthly forecasting layer as the primary analytical, modeling, and business-facing goal,
- a weekly forecast as an operational complement and short-term anchor,
- an optional daily allocation or reconciliation layer,
- experiment tracking and artifact versioning,
- a reproducible repository structure,
- and an application layer for forecast consumption.

This project is also intended to demonstrate applied AI and engineering maturity in an academic setting while remaining pragmatic and feasible within a short delivery window.

## 3. North Star

Build a functional, reproducible, and academically defensible proof of concept that shows how temporal hierarchical forecasting, supported by exogenous variables and a clean technical architecture, can improve demand forecasting practice for a critical SKU and serve as a credible base for future evolution.

## 4. Project Identity

### 4.1 What this project is

- A proof of concept for applied demand forecasting.
- A forecasting project with a clear temporal hierarchy.
- A lightweight MLOps implementation focused on reproducibility and modularity.
- A technical-academic project with a functional application layer.
- A repository intentionally structured to be understandable by AI coding assistants such as Codex or Claude Code.

### 4.2 What this project is not

- A full enterprise forecasting platform.
- A complete orchestration or CI/CD program.
- A large-scale benchmarking project with foundation models.
- An advanced explainability platform.
- A fully cloud-native enterprise data platform.

These excluded areas may be referenced as future work, but they are not part of the core delivery criteria.

## 5. Core Objective

The project has three non-negotiable outputs:

1. A functional proof of concept, ideally including a deployed Streamlit application.
2. A solid academic report with a defensible methodology.
3. A technically robust project base that could realistically serve as the seed for future internal initiatives.

Success should therefore be judged both by forecasting performance and by implementation quality.

## 6. Functional Scope

### 6.1 In scope

- Forecasting for one critical SKU.
- Monthly benchmark modeling as the central forecasting layer.
- Weekly forecasting as an operational enhancement and short-term anchor.
- Use of exogenous variables already collected by the team.
- Time-based backtesting and reproducible evaluation.
- Model and artifact tracking with MLflow.
- Forecast output generation for downstream app consumption.
- Streamlit-based forecast viewer.
- Support for user-provided demand updates at either daily or monthly granularity, with internal routing to the appropriate monthly-only or monthly-plus-weekly training workflow.
- Lightweight GenAI/RAG assistant for stakeholder interaction with historical series, forecasts, model outputs, metrics, and documented business events.
- FastAPI included as a real serving path, even if initially limited.
- Modular project structure using Kedro.
- Basic unit tests for critical functions.

### 6.2 Secondary scope if time allows

- More advanced daily shares modeling.
- Full monthly ↔ weekly ↔ daily coherence.
- Streamlit-triggered or API-triggered retraining workflows based on validated user-uploaded demand and future exogenous variables.
- More advanced GenAI-based narrative generation, automated forecast commentary, and richer explanation workflows.
- Optional benchmark using Nixtla NeuralForecast with N-HiTS.
- On-demand prediction requests through FastAPI.

### 6.3 Out of core scope

- Deep explainability workflows.
- Extensive benchmarking with foundation models.
- Enterprise observability and monitoring stack.
- Enterprise-grade data lake architecture.
- Full industrial-grade CI/CD.

## 7. Business and Modeling Context

The project focuses on a single SKU that is especially important and difficult to forecast. The current business process mainly forecasts at the monthly level, but the proposed solution aims to provide more granular outputs, especially at the weekly level and potentially at the daily level.

The hierarchy of interest is strictly temporal:

- Monthly
- Weekly
- Daily

The monthly forecast is the primary decision-making and stakeholder-facing layer of the project. It reflects the level at which the current business process is mainly planned, evaluated, and communicated.

The weekly forecast is an important operational complement. It provides a more granular short-term view, especially considering the 14-week exposure window related to supply and replenishment decisions, but it should not displace the monthly layer as the core modeling, evaluation, and reporting priority.

The business context also includes a 14-week exposure window related to supply and replenishment decisions. This should be treated as an operating context for decision-making, not necessarily as the only forecasting horizon definition.

## 8. Methodological Principles

### 8.1 Pragmatism over idealization

The project should prioritize feasible, testable, and demonstrable solutions over theoretically ideal but high-risk designs.

### 8.2 Monthly layer as the primary decision model

The monthly forecasting layer is the methodological and business-facing center of the project. It should receive the greatest modeling effort, evaluation rigor, reporting emphasis, and application-level visibility.

The weekly forecast should be developed as an operational anchor and enhancement layer. Its role is to provide additional short-term granularity and support the 14-week planning context, but without compromising the quality, completeness, or clarity of the monthly forecasting workflow.

### 8.3 Minimum valuable temporal coherence

The primary coherence target is **monthly ↔ weekly**, with the monthly forecast acting as the controlling and business-facing layer. Weekly forecasts should be consistent with the monthly planning view and should enrich it operationally rather than redefine the main project objective.

Reaching **monthly ↔ weekly ↔ daily** is desirable, but should be treated as an extension if time becomes constrained.

### 8.4 Application layer matters

The application is part of the product, not just a presentation artifact.

### 8.5 MLOps should be real but lightweight

The project should show versioning, modularity, tracking, artifact management, and reproducibility without trying to mimic a full enterprise platform.

### 8.6 AI-readable architecture

The repository and blueprint should be explicit enough for AI-assisted development tools to contribute without redefining the project’s methodological direction.

## 9. Modeling Direction

### 9.1 Official model candidates

- **SARIMAX** as a structured statistical baseline, especially relevant for the monthly layer.
- **Prophet** as an existing benchmark and early starting point.
- **CatBoost** as the main tabular candidate with exogenous variables, particularly for monthly forecasting.
- **Nixtla NeuralForecast (preferably N-HiTS)** as an optional exploratory model if complexity remains manageable and only after the core monthly layer is stable.

### 9.2 Monthly forecasting as the primary decision layer

The monthly forecasting layer is the **main modeling priority** of the project and the most important output for decision-making. This is the level that receives the greatest attention from stakeholders and tutors, so the architecture, evaluation effort, and reporting narrative should primarily revolve around monthly performance.

The monthly model remains intentionally central because:

- it aligns directly with the main business decision horizon,
- it reflects the level of forecasting currently most relevant for planning,
- it provides the strongest benchmark for measuring improvement,
- it supports a more feasible and robust integration of exogenous variables,
- and it offers the clearest path for delivering a defensible and useful forecasting POC.

### 9.3 Weekly forecasting as a secondary enhancement layer

The weekly layer should be treated as a **valuable extension**, not as the core success criterion of the project. Its role is to enrich the solution with a more granular operational perspective when feasible, but without shifting attention away from the monthly forecasting objective.

Weekly forecasting is still useful because:

- it can provide additional short-term interpretability,
- it may serve as an operational complement to the monthly layer,
- and it can strengthen the technical ambition of the solution if implemented successfully.

However, it should be developed only insofar as it does not compromise the quality, rigor, and completeness of the monthly layer.

### 9.4 Daily allocation philosophy

The daily layer should be treated as a **low-priority exploratory extension**. Given the high variability and instability of the data at that level, a fully modeled daily forecasting layer is unlikely to be feasible or reliable within the scope of the project.

Therefore:

- if a reasonable daily shares or disaggregation approach is feasible, it may be included as an additional plus,
- but it should not be considered a core project deliverable,
- and the absence of a formal daily model should not be seen as a weakness if the monthly layer is strong and the weekly layer is adequately supported.

Pragmatic disaggregation strategies based on historical intra-week patterns, calendar effects, and reconciliation constraints are acceptable if daily outputs are needed for demonstration purposes.

## 10. Exogenous Variables and Market Intelligence

Exogenous variables are a core part of the solution and are already being collected by the team. This blueprint assumes their availability and treats them as central model inputs without documenting them exhaustively here.

Market intelligence refers to business judgment and planner knowledge currently used in manual adjustments. In this project, market intelligence should be acknowledged as part of the narrative context, but not treated as a complex modeling stream unless the team later formalizes it.

### 10.1 Business Event Log as Contextual Knowledge

The project may include a business event log maintained by analysts, containing relevant month-by-month events for the selected SKU. This information should be treated as contextual knowledge for interpretation, not as a direct modeling input unless it is later formalized into structured exogenous variables.

The event log can be used by the GenAI/RAG layer to help users understand historical demand movements, forecast behavior, unusual periods, and relevant business context. However, this information should not override model outputs, evaluation metrics, or documented forecasting assumptions.

When used in the application layer, the event log should be ingested, cleaned, indexed, and retrieved as a source-backed knowledge base. Responses generated from this source should clearly distinguish between documented events, model results, and generated interpretation.

## 10.2 User Input and Data Refresh Logic

The project should support a future MVP workflow in which users can upload updated demand data and future exogenous variables through the application layer. These inputs should be validated, standardized, and routed into the appropriate Kedro pipeline workflow.

The system should support two demand input modes:

1. **Daily demand input**  
   When the user provides demand at daily granularity, the system should aggregate the demand into both monthly and weekly analytical views. This mode enables monthly model training as the primary layer and weekly model training as an operational enhancement. When weekly forecasts are generated, monthly ↔ weekly reconciliation may be applied.

2. **Monthly demand input**  
   When the user provides demand at monthly granularity, the system should train and update only the monthly forecasting workflow. Weekly model training should not be forced from monthly-only demand because the required lower-level temporal signal is not available.

The input validation layer should detect the temporal granularity of the uploaded demand data, validate required columns, parse dates, check duplicate periods, identify missing periods, validate negative demand values, and verify that exogenous variables are available for the required forecast horizon.

Future exogenous variables should be treated as required forecast drivers when the selected model depends on them. The system should clearly distinguish between historical observed demand used for retraining and future exogenous assumptions used for prediction.

This logic should remain part of the controlled data and pipeline layer. Streamlit may collect the uploaded files and trigger the workflow, but it should not contain the core validation, aggregation, feature engineering, training, or inference logic.

## 11. Evaluation Strategy

Evaluation must combine business-facing interpretation and technical rigor.

### 11.1 Business-facing layer

- Forecast accuracy improvement toward the target threshold (75%).
- Better support for planning decisions.
- Potential reduction of stockout risk and planning instability.

### 11.2 Technical evaluation layer

The technical evaluation layer should combine business-facing interpretability with rigorous forecast comparison. Metrics should be reported consistently across models, horizons, and temporal aggregation levels.

Primary metrics:

- **WAPE**: business-facing aggregate error. This should be the main metric for communicating forecast accuracy because it is easier to interpret at an aggregate planning level and is less sensitive to item-level denominator issues than MAPE.
- **MASE**: scale-free comparison against naive baselines. This metric is important for evaluating whether the model provides real improvement over simple time-series benchmarks.
- **RMSE**: error metric that penalizes large deviations. This is useful for detecting models that may perform acceptably on average but still generate large forecast misses.

Secondary metrics:

- **MAPE or sMAPE**, only when denominator issues are handled and clearly documented.
- **Bias**, to identify systematic overforecasting or underforecasting.
- **Horizon-specific error**, to understand how performance changes across forecast horizons (for 2 and 3 months ahead).
- **Interval coverage**, if prediction intervals are produced.

Metrics must be reported:

- For monthly forecasts as the primary decision and reporting layer.
- For weekly forecasts, when the weekly layer is implemented and validated.
- By forecast horizon, especially for the target horizons used in the project.
- Before and after reconciliation, if reconciliation is applied.

### 11.3 Validation protocol

Evaluation must be time-aware, leakage-safe, and reproducible. The recommended validation pattern is rolling-origin backtesting or an equivalent time-based validation strategy.

Random splits should not be used for model evaluation because they break the temporal structure of the forecasting problem and may introduce leakage.

Model selection should be based on held-out temporal periods and should report performance by horizon. For the monthly layer, evaluation should explicitly support the relevant forecast horizons defined by the project, such as 3, 6, and 12 months when applicable.

### 11.4 Model Selection and Champion Protocol

Model selection must follow a staged, time-aware, and leakage-safe protocol. The goal is to avoid selecting models directly on the final test period and to ensure that the final application forecasts are generated from models trained with the maximum available historical information.

The protocol is structured in four stages.

#### Stage 1 — Time-based data split

The available historical data must be split into three ordered temporal blocks:

- **Training period**: used to fit model candidates and tune hyperparameters.
- **Validation / evaluation period**: used to evaluate tuned candidates and shortlist the best configurations.
- **Testing period**: held out until the final model comparison stage.

Random splits must not be used because they break the temporal structure of the forecasting problem and may introduce leakage.

#### Stage 2 — Hyperparameter tuning and validation shortlist

For each active temporal granularity, model family, and forecast horizon, hyperparameter tuning is performed using the training period.

Each model configuration is evaluated on the validation/evaluation period. Based on the primary metrics and relevant secondary diagnostics, the top candidate configurations are selected.

The default shortlist rule is:

- select the top 3 candidate configurations;
- per model family;
- per temporal granularity;
- per forecast horizon.

This stage produces a controlled set of candidate configurations, but it does not yet define the final champion model.

#### Stage 3 — Refit on training + validation and final testing

The shortlisted candidate configurations are refitted using the combined training + validation data.

These refitted candidates are then evaluated on the held-out testing period. This test-period evaluation is used to select the best configuration for each model family, temporal granularity, and forecast horizon.

The output of this stage is the **family champion**.

A family champion is the best validated configuration within a specific model family for a specific granularity and horizon.

Examples:

- best monthly Prophet configuration for a 3-month horizon;
- best monthly CatBoost configuration for a 6-month horizon;
- best weekly Prophet configuration for a 14-week horizon, when the weekly workflow is active.

#### Stage 4 — Final champion refit and application forecast generation

Once the champion configuration is selected, the model is refitted using all available historical data.

This final refit is used to generate the forecasts consumed by the application layer.

The final forecast artifacts must include:

- selected model family;
- selected hyperparameters;
- temporal granularity;
- forecast horizon;
- training data cutoff;
- metrics from validation/evaluation and testing;
- final forecast generation timestamp;
- assumptions about future exogenous variables;
- model and run identifiers, when MLflow is used.

The final application forecast should not be generated from a model trained only on the original training split if validation and testing data are already available and approved for final refitting.

#### Champion hierarchy

The project may distinguish between two champion levels:

- **Family champion**: best configuration within a model family, granularity, and horizon.
- **Production champion**: final selected model across all eligible families for a given granularity and horizon.

If the application exposes multiple model options, family champions may be made available for comparison. If the application exposes one official forecast, the production champion must be selected using predefined metrics and documented tie-breaker criteria.

#### Selection criteria

The primary selection criteria should be based on the project’s official metrics:

- **WAPE** as the main business-facing aggregate error metric.
- **MASE** as a scale-free comparison against naive baselines.
- **RMSE** to penalize large forecast misses.

Secondary tie-breakers may include:

- forecast bias;
- horizon-specific error;
- stability across horizons;
- performance before and after reconciliation, when applicable;
- business plausibility of the forecast;
- interpretability and operational simplicity;
- consistency with future exogenous assumptions.

The model with the lowest error on a single metric is not automatically the final champion if it performs poorly on other critical diagnostics or produces forecasts that are not business-plausible.

### 11.5 Reporting expectations

Evaluation outputs should be designed for both technical review and stakeholder interpretation.

At minimum, reporting artifacts should include:

- monthly model performance by metric and horizon;
- validation and test performance separated clearly;
- champion model selection criteria;
- comparison against baseline or benchmark models;
- weekly performance only when the weekly layer is implemented and validated;
- reconciliation diagnostics when monthly ↔ weekly reconciliation is applied;
- clear notes about assumptions, limitations, and any denominator handling used for WAPE or MASE.

The monthly layer should receive the greatest evaluation emphasis because it is the primary modeling, decision-making, and stakeholder-facing layer of the project.

## 12. Technical Architecture

### 12.1 Architecture style

The project should be designed as **local-first with cloud option**.

This means:

- it must run cleanly in local development,
- but it should remain structurally ready for deployment to AWS services such as EC2 and optional S3-backed storage.

### 12.2 Base technology stack

- **Python**
- **Pandas**
- **Scikit-learn / Statsmodels / Prophet / CatBoost / optional Nixtla tools**
- **MLflow**
- **Kedro**
- **Streamlit**
- **FastAPI**
- **Docker**
- **GitHub**
- **pytest**
- **Optional GenAI/RAG stack**: LangChain or equivalent framework, vector store, and either a local LLM or a stable external LLM API.

### 12.3 Deployment direction

The solution should be conceived so that it can be packaged and deployed to a cloud environment such as EC2. Docker should support reproducibility and portability.

## 13. Kedro Role in the Project

Kedro is intentionally included as the organizing framework for the repository and the main pipelines.

Its role is to:

- structure data and modeling workflows into modular pipelines,
- separate concerns across ingestion, feature engineering, training, evaluation, and serving,
- improve reproducibility,
- and make the repository easier to navigate for both humans and AI tools.

Kedro should not become the focus of the project. It is a means of enforcing professional structure, not an end in itself.

## 14. Application and Serving Layer

The application layer is a required part of the proof of concept.

### 14.1 First expected version

- Streamlit viewer
- Precomputed forecasts
- Model and horizon selection
- Readable forecast outputs for stakeholders

### 14.2 Intended evolution if time allows

- FastAPI backend
- On-demand forecast requests
- Cleaner separation between backend inference and frontend consumption

FastAPI should therefore be treated as a real architectural component, even if the first delivery behaves mostly as a viewer.

### 14.3 GenAI/RAG Interaction Layer

The application layer may include a GenAI-powered chatbot to help users interact with the forecasting system in natural language. This assistant should be implemented as a lightweight RAG layer connected to the Streamlit app.

The chatbot should help users answer questions about:

- historical behavior of the selected SKU time series;
- relevant month-by-month business events documented by analysts;
- generated forecasts and forecast horizons;
- model candidates, champion model selection, and basic model assumptions;
- evaluation metrics and their interpretation;
- forecast validity, limitations, and known caveats;
- differences between monthly and weekly outputs when both are available.

The assistant should retrieve information from controlled project sources, such as:

- historical demand summaries;
- forecast output tables;
- model metadata and champion registry artifacts;
- evaluation reports and metric tables;
- reconciliation diagnostics, if available;
- project documentation;
- analyst-maintained business event logs.

The GenAI layer must be source-grounded. It should not invent business explanations, override forecast outputs, or present unsupported claims as facts. When the retrieved context is insufficient, the assistant should explicitly state that the available project sources do not contain enough information to answer confidently.

This layer is intended to improve forecast consumption, interpretation, and stakeholder usability. It is not intended to replace model evaluation, formal explainability, or planner judgment.

### 14.4 User Upload and Pipeline Triggering

A MVP version of the application may allow users to upload updated demand data and future exogenous variables. The application should act as the user-facing entry point for this workflow, while the forecasting logic remains inside Kedro pipelines.

FastAPI may act as the backend entry point for training and forecast refresh requests. In this workflow, the API receives or references uploaded files, validates request metadata, triggers the appropriate Kedro pipeline route, and exposes updated forecast outputs to the Streamlit application.

The expected application behavior is:

- allow users to upload demand data and exogenous variables;
- validate the files at a basic interface level;
- send the files to a controlled staging area or backend endpoint;
- trigger the appropriate Kedro workflow;
- show the status of the run;
- expose the updated forecasts once the run finishes.

The pipeline routing logic should depend on detected demand granularity:

- daily demand input → monthly and weekly workflows;
- monthly demand input → monthly workflow only.

This workflow should preserve reproducibility by saving uploaded inputs, validation reports, run metadata, generated forecasts, and model artifacts through the project’s catalog and tracking conventions.

## 15. Suggested Logical Project Areas

The repository should remain general, modular, and clearly oriented around the **primary monthly forecasting layer**. At a high level, it should separate concerns such as:

- data ingestion and cleaning,
- temporal aggregation and feature engineering,
- monthly forecasting as the main modeling and evaluation layer,
- weekly forecasting as a secondary enhancement layer,
- optional reconciliation and lightweight disaggregation logic when useful,
- evaluation, backtesting, and model comparison,
- artifact generation and reporting,
- inference, serving, and application layer.
- GenAI/RAG knowledge preparation and chatbot interaction layer for forecast interpretation and business context retrieval;
- user-uploaded input validation, temporal granularity detection, and pipeline routing;
- application-triggered forecast refresh workflows for MVP-like retraining and forecast updates;

The architecture should make it explicit that the **monthly layer is the core analytical and business-facing component** of the solution, while the weekly layer acts as an additional level of technical and operational value if it can be implemented without compromising the quality of the monthly workflow. Daily-level outputs, if included at all, should be treated only as optional downstream derivations rather than as a primary modeling focus.

The exact folder layout can later be refined around Kedro conventions, modular pipelines, and namespaces, but the blueprint should keep this functional separation visible from the beginning. 

The GenAI/RAG layer should remain downstream of the forecasting and evaluation pipelines. It should consume curated outputs and documented knowledge sources, not redefine model results, metrics, or methodological assumptions.

## 16. Testing Expectations

Testing is intentionally lightweight in scope.

The minimum expectation is:

- unit tests for critical utility functions,
- especially around feature generation, reconciliation logic, and evaluation helpers.

Testing is included mainly to demonstrate engineering discipline rather than to emulate a full QA program.

If the GenAI/RAG layer is implemented, lightweight tests should validate document ingestion, retrieval behavior, prompt construction, and failure handling when no relevant context is available.

If user-uploaded retraining workflows are implemented, tests should cover input schema validation, temporal granularity detection, daily-to-weekly/monthly aggregation, monthly-only routing, and error handling for invalid or incomplete exogenous variables.

## 17. AI-Assisted Development Guidelines

This project is expected to be developed in part with support from AI coding tools such as Codex or Claude Code. For that reason, the repository and blueprint should be explicit, modular, and constraint-aware.

### 17.1 What AI tools may help with

- scaffolding repository components,
- implementing modular pipeline code,
- refactoring utilities,
- generating documentation,
- writing repetitive boilerplate,
- drafting tests,
- helping convert notebooks into production-style modules.

### 17.2 What AI tools should not redefine

- the methodological north star,
- the scope boundaries,
- the hierarchy priorities,
- the main evaluation principles,
- the repository’s architectural logic.

### 17.3 Rules for AI-generated contributions

- Do not place heavy business logic in notebooks if it belongs in Kedro pipelines.
- Do not break the separation between data, features, training, evaluation, and app layers.
- Do not add major new dependencies without justification.
- Do not replace agreed project choices with speculative alternatives.
- Always document assumptions, inputs, and outputs.
- Prefer modular, readable, and testable code.
- Maintain consistency with the project’s Kedro-based structure.

In short, AI should accelerate implementation, but it should not steer the project away from its defined rails.

## 18. Success Criteria

The project should be considered successful if it achieves most of the following:

- a functional and stable proof of concept,
- a solid academic narrative,
- a technically organized repository,
- reproducible model training and evaluation,
- a visible application layer,
- a credible improvement path over the current forecasting process,
- a lightweight GenAI/RAG assistant that helps users interpret forecasts, metrics, historical behavior, and documented business events, if included in the final application scope;
- and a result that could plausibly serve as a basis for future work inside the partner organization.

Reaching a specific forecast metric threshold is important, but it is not the only definition of success.

## 19. Short Executive Summary

This project aims to build a functional and reproducible proof of concept for demand forecasting of a critical SKU using a temporal hierarchical approach. The central logic is a **primary monthly forecasting layer + weekly operational complement**, supported by exogenous variables, modular Kedro pipelines, MLflow-based artifact tracking, and an application layer built around Streamlit with FastAPI as an architectural serving path.

The project is intentionally scoped to remain pragmatic. Its main goal is not to build a full enterprise platform, but to deliver a technically credible, academically strong, and operationally meaningful prototype that demonstrates learning, rigor, and future potential.
