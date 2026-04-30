PIPELINES_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))pipelines
KEDRO := cd $(PIPELINES_DIR) && uv run kedro

.PHONY: help \
	run \
	ingest \
	fe-monthly fe-weekly \
	model-input \
	train-monthly train-weekly train \
	model-selection \
	reconcile \
	infer \
	full-experiment \
	inference \
	viz \
	catalog pipeline-list

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Full pipeline ────────────────────────────────────────────────────────────

run: ## Run the full default pipeline (all stages)
	$(KEDRO) run

# ── Individual stages ────────────────────────────────────────────────────────

ingest: ## data_ingestion — load raw demand + exogenous data
	$(KEDRO) run --pipeline data_ingestion

fe-monthly: ## feature_engineering_monthly — monthly feature set
	$(KEDRO) run --pipeline feature_engineering_monthly

fe-weekly: ## feature_engineering_weekly — weekly feature set
	$(KEDRO) run --pipeline feature_engineering_weekly

model-input: ## model_input_preparation — build final model input tables
	$(KEDRO) run --pipeline model_input_preparation

train-monthly: ## train_monthly — train monthly forecasting models
	$(KEDRO) run --pipeline train_monthly

train-weekly: ## train_weekly — train weekly forecasting models
	$(KEDRO) run --pipeline train_weekly

train: ## training — train both monthly and weekly models
	$(KEDRO) run --pipeline training

model-selection: ## model_selection — select champion models
	$(KEDRO) run --pipeline model_selection

reconcile: ## reconciliation — temporal hierarchy reconciliation
	$(KEDRO) run --pipeline reconciliation

infer: ## forecast_inference — generate forecasts with champion models
	$(KEDRO) run --pipeline forecast_inference

# ── Composed shortcuts ───────────────────────────────────────────────────────

full-experiment: ## full_experiment — ingest → FE → model input → train → select
	$(KEDRO) run --pipeline full_experiment

inference: ## inference — forecast + reconciliation
	$(KEDRO) run --pipeline inference

# ── Kedro utilities ──────────────────────────────────────────────────────────

viz: ## Launch Kedro-Viz in the browser
	$(KEDRO) viz run

catalog: ## List all datasets registered in the catalog
	$(KEDRO) catalog list

pipeline-list: ## List all registered pipelines
	$(KEDRO) pipeline list
