# Demand Forecast through Temporal Hierarchical Modelling Blueprint

## 1. Purpose

This document is the official blueprint for the project. It is intentionally written to be easy to read by both humans and machines. Its purpose is to capture the strategic direction, scope, architecture, implementation logic, and development guardrails of the initiative in a single reference.

The project should be understood as a **Forecasting + MLOps + Application Layer** proof of concept rather than as a forecasting model in isolation.

## 2. Project Overview

The project aims to build a functional proof of concept to improve demand forecast accuracy for a critical SKU through a temporal hierarchical forecasting approach supported by exogenous variables, modular pipelines, and a lightweight but real MLOps foundation.

The intended solution moves beyond a process dominated by aggregated statistical forecasting and manual planner adjustments. The target direction is a structured system built around:

- a monthly benchmark model,
- a weekly anchor forecast as the main operational layer,
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
- Monthly benchmark modeling.
- Weekly anchor forecasting as the central forecasting layer.
- Optional daily disaggregation or reconciliation.
- Use of exogenous variables already collected by the team.
- Time-based backtesting and reproducible evaluation.
- Model and artifact tracking with MLflow.
- Forecast output generation for downstream app consumption.
- Streamlit-based forecast viewer.
- FastAPI included as a real serving path, even if initially limited.
- Modular project structure using Kedro.
- Basic unit tests for critical functions.

### 6.2 Secondary scope if time allows

- More advanced daily shares modeling.
- Full monthly ↔ weekly ↔ daily coherence.
- Initial GenAI-based narrative or explanation layer.
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

The weekly forecast is the main intended operational layer. The monthly model remains relevant as a benchmark, a baseline reference, and a bridge to the current business process.

The business context also includes a 14-week exposure window related to supply and replenishment decisions. This should be treated as an operating context for decision-making, not necessarily as the only forecasting horizon definition.

## 8. Methodological Principles

### 8.1 Pragmatism over idealization

The project should prioritize feasible, testable, and demonstrable solutions over theoretically ideal but high-risk designs.

### 8.2 Weekly anchor as the core model

The weekly anchor forecast is the methodological center of the project. Other components should support, benchmark, or extend that layer.

### 8.3 Minimum valuable temporal coherence

The primary coherence target is **monthly ↔ weekly**. Reaching **monthly ↔ weekly ↔ daily** is desirable, but should be treated as an extension if time becomes constrained.

### 8.4 Application layer matters

The application is part of the product, not just a presentation artifact.

### 8.5 MLOps should be real but lightweight

The project should show versioning, modularity, tracking, artifact management, and reproducibility without trying to mimic a full enterprise platform.

### 8.6 AI-readable architecture

The repository and blueprint should be explicit enough for AI-assisted development tools to contribute without redefining the project’s methodological direction.

## 9. Modeling Direction

### 9.1 Official model candidates

- **SARIMAX** as a structured statistical baseline.
- **Prophet** as an existing benchmark and early starting point.
- **CatBoost** as the main tabular candidate with exogenous variables.
- **Nixtla NeuralForecast (preferably N-HiTS)** as an optional exploratory model if complexity remains manageable.

### 9.2 Monthly benchmark role

The monthly model remains intentionally in scope because:

- it aligns with current business practice,
- it provides a baseline comparison,
- it supports the improvement narrative,
- and it helps structure the exogenous variable workflow.

### 9.3 Weekly anchor role

The weekly anchor is the intended main output for operational decision-making. It should integrate temporal structure and exogenous business variables in a way that is more responsive and useful than a monthly-only forecast.

### 9.4 Daily allocation philosophy

If a proper daily shares model is feasible, it should be included. If not, pragmatic disaggregation strategies based on historical intra-week patterns, calendar structure, and reconciliation constraints are acceptable.

## 10. Exogenous Variables and Market Intelligence

Exogenous variables are a core part of the solution and are already being collected by the team. This blueprint assumes their availability and treats them as central model inputs without documenting them exhaustively here.

Market intelligence refers to business judgment and planner knowledge currently used in manual adjustments. In this project, market intelligence should be acknowledged as part of the narrative context, but not treated as a complex modeling stream unless the team later formalizes it.

## 11. Evaluation Strategy

Evaluation must combine business-facing interpretation and technical rigor.

### 11.1 Business-facing layer

- Forecast accuracy improvement toward the target threshold.
- Better support for planning decisions.
- Potential reduction of stockout risk and planning instability.

### 11.2 Technical evaluation layer

Primary metrics to emphasize:

- **MAPE**
- **RMSE**
- **MASE**

Additional metrics that may be explored if useful:

- Bias
- Interval coverage
- Horizon-specific error
- Accuracy summaries by aggregation level

### 11.3 Validation protocol

Evaluation must be time-aware and reproducible. The recommended validation pattern is rolling-origin or an equivalent time-based backtesting strategy.

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

## 15. Suggested Logical Project Areas

The repository should remain general and modular. At a high level, it should separate concerns such as:

- data ingestion and cleaning,
- temporal aggregation and feature engineering,
- monthly forecasting,
- weekly anchor forecasting,
- optional daily allocation and reconciliation,
- evaluation and backtesting,
- artifact generation,
- serving and application layer.

The exact folder layout can later be refined around Kedro conventions, but the blueprint should keep this logical separation visible.

## 16. Testing Expectations

Testing is intentionally lightweight in scope.

The minimum expectation is:

- unit tests for critical utility functions,
- especially around feature generation, reconciliation logic, and evaluation helpers.

Testing is included mainly to demonstrate engineering discipline rather than to emulate a full QA program.

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
- and a result that could plausibly serve as a basis for future work inside the partner organization.

Reaching a specific forecast metric threshold is important, but it is not the only definition of success.

## 19. Short Executive Summary

This project aims to build a functional and reproducible proof of concept for demand forecasting of a critical SKU using a temporal hierarchical approach. The central logic is a **monthly benchmark + weekly anchor + optional daily allocation**, supported by exogenous variables, modular Kedro pipelines, MLflow-based artifact tracking, and an application layer built around Streamlit with FastAPI as an architectural serving path.

The project is intentionally scoped to remain pragmatic. Its main goal is not to build a full enterprise platform, but to deliver a technically credible, academically strong, and operationally meaningful prototype that demonstrates learning, rigor, and future potential.
