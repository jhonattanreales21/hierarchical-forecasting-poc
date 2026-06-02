ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PIPELINES_DIR := $(ROOT_DIR)pipelines
KEDRO := cd $(PIPELINES_DIR) && uv run kedro

.DEFAULT_GOAL := help

.PHONY: help \
	run viz app \
	prophet sarimax catboost \
	compare \
	ingest fe-monthly model-input train-monthly model-selection infer \
	catalog pipeline-list

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Main shortcuts ─────────────────────────────────────────────────────────────

run: ## Run the full default pipeline (monthly_forecast_e2e: train all families → select → infer)
	$(KEDRO) run

viz: ## Launch Kedro-Viz in the browser
	$(KEDRO) viz run

app: ## Launch the Streamlit forecast viewer
	cd $(ROOT_DIR) && uv run --package hdf_app streamlit run app/app.py

# ── Per-model routes (raw → features → splits → train that family) ─────────────

prophet: ## Prophet monthly route (ingest → FE → splits → train)
	$(KEDRO) run --pipeline prophet_monthly_e2e

sarimax: ## SARIMAX monthly route (ingest → FE → splits → train)
	$(KEDRO) run --pipeline sarimax_monthly_e2e

catboost: ## CatBoost monthly route (ingest → FE → splits → train)
	$(KEDRO) run --pipeline catboost_monthly_e2e

compare: ## Train all monthly families and select the champion (no inference)
	$(KEDRO) run --pipeline monthly_training_comparison

# ── Individual stages ──────────────────────────────────────────────────────────

ingest: ## data_ingestion — load raw demand + exogenous data
	$(KEDRO) run --pipeline data_ingestion

fe-monthly: ## feature_engineering_monthly — monthly feature set
	$(KEDRO) run --pipeline feature_engineering_monthly

model-input: ## model_input_preparation — build train/val/test splits per family
	$(KEDRO) run --pipeline model_input_preparation

train-monthly: ## train_monthly — train all monthly families (Prophet + SARIMAX + CatBoost)
	$(KEDRO) run --pipeline train_monthly

model-selection: ## model_selection — select champion models
	$(KEDRO) run --pipeline model_selection

infer: ## forecast_inference — generate forecasts with champion models
	$(KEDRO) run --pipeline forecast_inference

# ── Kedro utilities ────────────────────────────────────────────────────────────

catalog: ## Describe all datasets used across the catalog
	$(KEDRO) catalog describe-datasets

pipeline-list: ## List all registered pipelines
	$(KEDRO) registry list
