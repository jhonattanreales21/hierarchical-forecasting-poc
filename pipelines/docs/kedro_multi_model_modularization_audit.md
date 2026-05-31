# Kedro Multi-Model Modularization Audit

## 1. Executive Summary

The project implements a hierarchical demand forecasting system with a **production-ready monthly Prophet MVP** sitting alongside **scaffolded but not yet implemented SARIMAX and CatBoost model families**. The codebase demonstrates strong architectural intent but exhibits **significant Prophet coupling** in shared artifacts (catalog names, parameter keys, node names, and inference logic) that must be decoupled before adding multi-family support.

**Key Findings:**

- **Prophet MVP Status:** Fully implemented, tested, and production-ready. Monthly pipeline (`monthly_mvp`) is stable and validated.
- **Multi-Model Readiness:** 60% ready. Infrastructure exists (namespaces, parameter blocks, placeholder nodes) but Prophet assumptions leak throughout the codebase.
- **Immediate Risks:** Implementing SARIMAX/CatBoost without refactoring will create brittle, family-specific branches in shared logic (feature engineering, model input, model selection, inference).
- **Recommended Path:** Before implementing SARIMAX/CatBoost, complete 5 refactoring tasks (listed in Section 12) to establish generic monthly model contracts.

---

## 2. Current Pipeline Map

### Registered Pipelines

**Validated & Active:**
- `__default__` → alias for `monthly_mvp`
- `monthly_mvp` → `prophet_monthly_e2e` (data ingestion → FE monthly → model input prep → train monthly Prophet → select Prophet champion → inference)
- `prophet_monthly_e2e` → same as `monthly_mvp`
- `data_ingestion` → standalone
- `feature_engineering_monthly` → standalone, all nodes
- `model_input_preparation` → standalone, Prophet-only nodes; SARIMAX/CatBoost stubs exist in catalog only

**Scaffolded & Experimental:**
- `experimental_training` → monthly + weekly training (includes all NotImplementedError stubs)
- `experimental_full_experiment` → full e2e including experimental selection/inference
- `experimental_inference` → inference + reconciliation (Prophet-only)
- `train_monthly` → composes Prophet (direct) + CatBoost (namespaced `train_monthly.catboost`) + SARIMAX (namespaced `train_monthly.sarimax`)
- `train_weekly` → scaffolded weekly (not in MVP)
- `model_selection` → generic 3-node pipeline expecting all 6 candidates; only Prophet-specific selection is implemented
- `forecast_inference` → Prophet-specific pipeline; stubs for weekly/daily

### Pipeline Composition Pattern

```
feature_engineering_monthly/
├── pipeline.py (3 nodes, model-agnostic + Prophet-specific)
│   ├── build_monthly_calendar_features
│   ├── build_monthly_exogenous_features
│   └── build_monthly_prophet_features ← Prophet-specific output

model_input_preparation/
├── pipeline.py (4 nodes, Prophet-only)
│   ├── prepare_monthly_prophet_modeling_data
│   ├── split_monthly_prophet_data
│   ├── build_monthly_prophet_future_regressors
│   └── build_monthly_prophet_split_metadata
└── catalog stubs for SARIMAX/CatBoost (orphans)

train_monthly/
├── prophet/
│   ├── pipeline.py (1 node)
│   │   └── train_and_evaluate_monthly_prophet_candidates
│   └── nodes.py (Optuna tuning)
├── catboost/
│   ├── pipeline.py (2 nodes, namespaced)
│   │   ├── tune_hyperparameters
│   │   └── train_best_candidate
│   └── nodes.py (NotImplementedError stubs)
└── sarimax/
    ├── pipeline.py (2 nodes, namespaced)
    │   ├── tune_hyperparameters
    │   └── train_best_candidate
    └── nodes.py (NotImplementedError stubs)

model_selection/
├── pipeline.py (3 generic nodes)
│   ├── evaluate_candidates_on_test (NotImplementedError)
│   ├── select_champion_models (NotImplementedError)
│   └── persist_champion_registry (NotImplementedError)
└── prophet/
    ├── pipeline.py (4 Prophet-specific nodes)
    │   ├── evaluate_monthly_prophet_prechampions_on_test
    │   ├── evaluate_monthly_prophet_rolling_origin_metrics
    │   ├── select_monthly_prophet_champion
    │   └── build_monthly_prophet_champion_model
    └── nodes.py (Prophet WAPE/MASE/RMSE evaluation, rolling-origin, champion selection)

forecast_inference/
├── pipeline.py (1 Prophet-specific node)
│   └── generate_monthly_prophet_forecasts
└── nodes.py (Prophet inference + stubs for weekly/daily allocation)
```

---

## 3. Namespace Readiness

### Current Namespace Usage

**Yes, namespaces are already in use in `train_monthly`:**

- `train_monthly.prophet` — direct imports, no namespace wiring
- `train_monthly.catboost` — namespace="train_monthly.catboost" applied
- `train_monthly.sarimax` — namespace="train_monthly.sarimax" applied

**Namespace Wiring in catalog inputs (CatBoost example):**
- Generic names in pipeline: `train`, `validation`, `parameters`, `tuning_result`, `candidate_model`
- Catalog input via namespace mapping (implied in Kedro composition):
  - `train` → `model_input_monthly_catboost_train`
  - `validation` → `model_input_monthly_catboost_validation`
  - `candidate_model` → `candidate_monthly_catboost`

### Naming Convention Analysis

**Current Pattern (Prophet):**
- Inputs: `monthly_prophet_train`, `monthly_prophet_validation`, `monthly_prophet_test`, etc.
- Training outputs: `monthly_prophet_tuning_results`, `monthly_prophet_validation_metrics`, `monthly_prophet_candidate_models`, `monthly_prophet_training_metadata`, `candidate_monthly_prophet`
- Selection outputs: `monthly_prophet_champion_model`, `monthly_prophet_champion_metadata`
- Inference outputs: `monthly_prophet_forecast_3m`, `monthly_prophet_forecast_6m`, `monthly_prophet_forecast_12m`, `monthly_prophet_forecast_latest`

**Current Pattern (CatBoost/SARIMAX stubs):**
- Inputs: `model_input_monthly_catboost_train`, `model_input_monthly_sarimax_train`, etc.
- Missing: tuning, training metadata, champion-specific outputs

**Recommendation:**

Use **`monthly_<family>`** for all family-specific catalog entries (not `model_input_monthly_catboost`):

| Layer | Current (Prophet) | Proposed Pattern | Example (CatBoost) |
|-------|---|---|---|
| Model Input | `monthly_prophet_train` | `monthly_<family>_train` | `monthly_catboost_train` |
| Training Output | `monthly_prophet_tuning_results` | `monthly_<family>_tuning_results` | `monthly_catboost_tuning_results` |
| Candidate | `candidate_monthly_prophet` | `candidate_monthly_<family>` | `candidate_monthly_catboost` |
| Champion | `monthly_prophet_champion_model` | `monthly_<family>_champion_model` | `monthly_catboost_champion_model` |

**Namespace + Catalog Synergy:**

Kedro namespaces allow sub-pipelines to use generic node names (`train`, `validation`, `tune_hyperparameters`) while mapping them to specific catalog entries via namespace prefixing. Current CatBoost/SARIMAX pipelines are wired correctly via namespace, but the Prophet pipeline does not use namespaces (direct composition). 

**Consistency Recommendation:**

- Option A (preferred): Apply namespace to Prophet too — `train_monthly.prophet`, then rename catalog entries to `monthly_prophet_*` for consistency.
- Option B: Remove namespaces from CatBoost/SARIMAX and use explicit long names in node inputs/outputs (less elegant, more verbose).

**Kedro-Viz Readability:**

Namespaces significantly improve readability when there are 3+ family-specific pipelines. Without them, the `train_monthly` stage appears as a single flat node in Kedro-Viz; with namespaces, it expands into three logical sub-pipelines. **Namespace all three families** (Prophet, CatBoost, SARIMAX) for clarity.

---

## 4. Prophet Coupling Findings

### Coupling Table

| Area | File / Module | Current Behavior | Coupling Type | Recommended Change | Risk |
|---|---|---|---|---|---|
| **Dataset Names** | `catalog.yml` lines 45–80 | `monthly_prophet_features`, `monthly_prophet_train`, `monthly_prophet_validation`, etc. | A: Correctly Prophet-specific | Keep Prophet names; introduce `monthly_<family>_*` for other families | Low — catalog is model-aware |
| **Feature Engineering Output** | `feature_engineering_monthly/nodes.py` | `build_monthly_prophet_features` returns flat table with Prophet-ready columns | D: Should be generalized | Create `build_monthly_model_features` that produces family-agnostic columns; Prophet-specific adapter later | Medium — affects FE pipeline contract |
| **Model Input Preparation** | `model_input_preparation/pipeline.py` | Only Prophet nodes exist; SARIMAX/CatBoost inputs are catalog stubs with no pipeline | D: Should be family-parameterized | Extract common monthly dataset split logic; parameterize model_family to route to family-specific adapters | High — blocks SARIMAX/CatBoost implementation |
| **Column Renaming (ds/y)** | `model_input_preparation/nodes.py` | Hardcoded rename to Prophet columns `ds` and `y` | D: Should be configurable | Move Prophet-specific naming into Prophet adapter node; model input prep produces generic columns | High — Prophet columns are not portable |
| **Training Node Names** | `train_monthly/prophet/pipeline.py` | `train_and_evaluate_monthly_prophet_candidates` — Prophet-specific name | C: Leaks into pipeline-level scope | Rename to `train_monthly_prophet_candidates` (still Prophet-specific but stage-scoped) | Low — internal to Prophet sub-pipeline |
| **Training Inputs/Outputs** | `train_monthly/prophet/pipeline.py` | Expects `monthly_prophet_train`, outputs `candidate_monthly_prophet` | A: Correctly Prophet-specific | OK as-is; other families will have their own inputs/outputs | Low — catalog-driven |
| **Metrics Functions** | `shared/metrics.py` (wape, mase, rmse, mape) | Generic implementations on numpy arrays | A: Correctly shared | Keep as-is; use in all families | Low — no coupling |
| **Metric Configuration** | `parameters/train_monthly.yml` lines 82–88 | `train_monthly.prophet.metrics.epsilon`, `mase_seasonal_period: 12` | A: Correctly family-specific | Propagate same structure to `train_monthly.catboost.metrics` and `train_monthly.sarimax.metrics` | Low — parameters are family-aware |
| **Model Input Preparation Parameters** | `parameters/model_input.yml` | Only `model_input_preparation.monthly_prophet` block exists; no CatBoost/SARIMAX blocks | D: Should be generic then family-specific | Create `model_input_preparation.monthly` (shared) + `model_input_preparation.monthly_prophet` (Prophet overrides) | High — blocks parameter-driven family routing |
| **Model Selection Pipeline** | `model_selection/pipeline.py` | Generic nodes expect all 6 candidates (monthly/weekly × 3 families) | B: Should become family-specific or multi-family | Keep generic pipeline for multi-family comparison; implement generic `evaluate_candidates_on_test` that dispatches per family | High — currently all NotImplementedError |
| **Model Selection Nodes** | `model_selection/nodes.py` | `evaluate_candidates_on_test` is NotImplementedError stub | B: Should become generic multi-family dispatcher | Implement as dispatcher: load model → family-specific predict logic → compute shared metrics | High — critical to MVP exit |
| **Model Selection Prophet** | `model_selection/prophet/pipeline.py` + nodes | 4 fully implemented Prophet-specific nodes with rolling-origin and tie-breaker logic | A: Correctly Prophet-specific but creates precedent | Document the Prophet-specific selection pattern; generalize rolling-origin to family-parameterized utility; tie-breaker logic is reusable | Medium — used as template for other families |
| **Forecast Inference Pipeline** | `forecast_inference/pipeline.py` | Node inputs hardcoded to Prophet catalogs: `monthly_prophet_champion_model`, `monthly_prophet_future_3m`, etc. | D: Should use metadata-driven champion loading | Change to load champion from generic `champion_monthly_model` + metadata; route prediction to family-specific adapter | High — blocks multi-family inference |
| **Forecast Inference Logic** | `forecast_inference/nodes.py` | `generate_monthly_prophet_forecasts` directly calls `model.predict()` and renames Prophet columns | D: Should dispatch based on model_family metadata | Move Prophet-specific logic into Prophet adapter; core inference orchestration should be family-agnostic | High — creates precedent for weekly/daily |
| **Inference Column Naming** | `forecast_inference/nodes.py` lines 19–37 | `_FORECAST_COLUMN_ORDER` hardcodes Prophet column order; assumes `yhat`, `yhat_lower`, `yhat_upper` | D: Should use family-specific adapters | Keep Prophet column names; other families produce their own output columns; merge at the end | Medium — output schema is documented |
| **Streamlit App** | `app/app.py` lines 45–50 | Hardcoded "Monthly Forecast" → "Forward-looking monthly demand forecast generated by the Prophet champion model." | C: Leaks into UI | Parameterize from metadata: display "generated by the [model_family] champion model" | Low — cosmetic, metadata exists |
| **Seasonal Naive Baseline** | `shared/metrics.py` + `train_monthly/prophet/nodes.py` | MASE uses hardcoded 12-month seasonality via `mase_seasonal_period` parameter; seasonal_naive implied but not explicit | A: Correctly preserved but implicit | Keep MASE as-is (family-agnostic); ensure CatBoost/SARIMAX tuning also computes MASE with same seasonal period for fair comparison | Low — metric is generic |

### Coupling Summary

- **High Risk (8 items):** Model input preparation (no SARIMAX/CatBoost pipeline), column naming, feature engineering generalization, model selection stub, inference dispatcher, metadata-driven champion loading.
- **Medium Risk (4 items):** Feature engineering contract, parameter organization, Prophet selection as template, seasonal naive baseline.
- **Low Risk (10 items):** Dataset names, metrics implementation, metric parameters, namespace naming, Streamlit display.

---

## 5. Model Input Preparation Findings

### Current State

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/src/hdf_pipelines/pipelines/model_input_preparation/pipeline.py`

**Implemented (Prophet only):**
1. `prepare_monthly_prophet_modeling_data` — joins calendar features, exogenous features, and demand; renames columns to `ds` and `y`
2. `split_monthly_prophet_data` — train/validation/test split using date-based logic
3. `build_monthly_prophet_future_regressors` — creates future feature frames for inference (3/6/12m horizons)
4. `build_monthly_prophet_split_metadata` — metadata dict with active_regressors, date ranges

**Catalog Stubs (no pipeline nodes):**
- `model_input_monthly_catboost_train/validation/test`
- `model_input_monthly_sarimax_train/validation/test`

### What Should Exist Before Multi-Family

**Generic Monthly Layer (no family-specific logic):**
1. Demand aggregation (demand_monthly, demand_weekly, demand_daily)
2. Exogenous variable aggregation (exogenous_monthly)
3. Generic feature engineering:
   - Calendar features (date-aware, granularity-specific)
   - Exogenous lags
   - Derived features (market_share_stress, uplift)
4. Generic train/validation/test split (date-range-based, granularity-specific)
5. Generic split metadata (date ranges, active features, row counts)

**Family-Specific Adapters (consume generic, produce family-ready):**
1. **Prophet Adapter:**
   - Rename columns to `ds` (date) and `y` (target)
   - Register regressors for Prophet API
   - Create future regressor frames for Prophet.predict()

2. **CatBoost Adapter:**
   - Produce feature-target split (X, y)
   - Ensure all features are numeric
   - No column renaming needed

3. **SARIMAX Adapter:**
   - Univariate target series (y_t)
   - Optional exogenous features as separate array
   - Handle date index / frequency

### Minimum Changes Needed

**For SARIMAX to consume monthly data:**

1. **Create generic monthly split datasets:**
   - `monthly_modeling_data` (union of demand + calendar + exogenous)
   - `monthly_train`, `monthly_validation`, `monthly_test` (splits of above)
   - `monthly_split_metadata` (date ranges, active features)

2. **Create SARIMAX adapter node:**
   - Input: `monthly_train`, `monthly_validation`, `monthly_test`
   - Output: `monthly_sarimax_train`, `monthly_sarimax_validation`, `monthly_sarimax_test`
   - Logic: extract target series + exogenous regressor frame

3. **CatBoost adapter node (similar):**
   - Output: `monthly_catboost_train`, `monthly_catboost_validation`, `monthly_catboost_test`
   - Logic: extract features and target; ensure numeric types

**Cost:** Add 2 nodes + 1 generic intermediate stage. Refactor Prophet adapter to consume `monthly_*` instead of creating its own.

---

## 6. Training Contract Findings

### Current Prophet Training Artifacts

**Location:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/src/hdf_pipelines/pipelines/train_monthly/prophet/`

**Pipeline:** `train_and_evaluate_monthly_prophet_candidates` (single node)

**Inputs:**
- `monthly_prophet_train` — training split with `ds`, `y`, `sku`, regressors
- `monthly_prophet_validation` — validation split (no `y`)
- `monthly_prophet_split_metadata` — metadata dict
- `params:train_monthly.prophet` — hyperparameter tuning config

**Outputs (6-tuple):**
1. `monthly_prophet_tuning_results` — ranked trial results (Parquet)
2. `monthly_prophet_validation_metrics` — detailed per-trial metrics (Parquet)
3. `monthly_prophet_prechampion_configs` — top-N configs (JSON)
4. `monthly_prophet_candidate_models` — fitted models (Pickle)
5. `monthly_prophet_training_metadata` — study summary (JSON)
6. `candidate_monthly_prophet` — rank-1 model (Pickle)

### Pattern Analysis

**Strengths:**
- Clear contract: train/validation inputs → ranked candidates + metadata
- Optuna study preserved for audit + reproducibility
- Top-N pre-champions (not just rank-1) enables downstream comparison
- Metadata artifact enables model-selection stage to track trial configuration

**Generalizability:**
- CatBoost/SARIMAX can follow same pattern:
  1. Train on `train_split`
  2. Evaluate on `validation_split`
  3. Rank candidates by objective metric
  4. Persist top-N configurations + fitted models

**Naming Convention (Family-Agnostic Template):**
```
monthly_<family>_tuning_results
monthly_<family>_validation_metrics
monthly_<family>_prechampion_configs
monthly_<family>_candidate_models
monthly_<family>_training_metadata
candidate_monthly_<family>
```

### Recommended Artifact Naming Convention

| Artifact | Current (Prophet) | Proposed Pattern | CatBoost Example |
|---|---|---|---|
| Tuning Results | `monthly_prophet_tuning_results` | `monthly_<family>_tuning_results` | `monthly_catboost_tuning_results` |
| Validation Metrics | `monthly_prophet_validation_metrics` | `monthly_<family>_validation_metrics` | `monthly_catboost_validation_metrics` |
| Pre-champion Configs | `monthly_prophet_prechampion_configs` | `monthly_<family>_prechampion_configs` | `monthly_catboost_prechampion_configs` |
| Candidate Models | `monthly_prophet_candidate_models` | `monthly_<family>_candidate_models` | `monthly_catboost_candidate_models` |
| Training Metadata | `monthly_prophet_training_metadata` | `monthly_<family>_training_metadata` | `monthly_catboost_training_metadata` |
| Rank-1 Candidate | `candidate_monthly_prophet` | `candidate_monthly_<family>` | `candidate_monthly_catboost` |

### Multi-Family Champion Registry

**Proposed Structure (generic layer):**
```json
{
  "monthly": {
    "prophet": {
      "champion_id": "prophet_candidate_003",
      "model_path": "data/06_models/champions/monthly_prophet_champion.pkl",
      "metadata": { ... }
    },
    "catboost": {
      "champion_id": "catboost_candidate_005",
      "model_path": "data/06_models/champions/monthly_catboost_champion.pkl",
      "metadata": { ... }
    },
    "sarimax": {
      "champion_id": "sarimax_candidate_002",
      "model_path": "data/06_models/champions/monthly_sarimax_champion.pkl",
      "metadata": { ... }
    }
  }
}
```

**Does NOT require separate Prophet/CatBoost/SARIMAX champion registries — one registry per granularity, indexed by family.**

---

## 7. Model Selection Findings

### Current State

**Generic Pipeline:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/src/hdf_pipelines/pipelines/model_selection/pipeline.py`

**Status:** All 3 nodes are `NotImplementedError` stubs.

1. `evaluate_candidates_on_test` — expects 6 candidate models + 6 test datasets
2. `select_champion_models` — expects evaluation report
3. `persist_champion_registry` — creates champion registry

**Prophet-Specific Pipeline:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/src/hdf_pipelines/pipelines/model_selection/prophet/`

**Status:** 4 fully implemented nodes:

1. `evaluate_monthly_prophet_prechampions_on_test` — scores top-N pre-champions on test set using WAPE/MASE/RMSE + rolling-origin M-2/M-3
2. `evaluate_monthly_prophet_rolling_origin_metrics` — computes horizon-aware metrics
3. `select_monthly_prophet_champion` — ranks pre-champions by primary metric + tie-breakers
4. `build_monthly_prophet_champion_model` — refits rank-1 model on train+validation for production

### Does Generic Pipeline Assume Prophet?

**Yes, critically:**
- Node inputs are hardcoded: `candidate_monthly_prophet`, `candidate_monthly_catboost`, `candidate_monthly_sarimax`, `candidate_weekly_*`
- No dispatch logic based on model type
- No way to score non-Prophet families

### Does It Support Horizon-Based Evaluation?

**Partially (Prophet only):**
- Prophet sub-pipeline computes `horizon_2_mape`, `horizon_3_mape` via `evaluate_monthly_prophet_rolling_origin_metrics`
- Generic pipeline has no horizon concept

### Minimum Generalization Needed

**For generic pipeline to compare Prophet vs SARIMAX:**

1. **Implement `evaluate_candidates_on_test` as dispatcher:**
   ```python
   for (family, candidate_model, test_dataset) in [(prophet, ...), (catboost, ...), (sarimax, ...)]:
       predictions = dispatch_predict(family, candidate_model, test_dataset)
       metrics = compute_shared_metrics(y_true, predictions)  # WAPE, MASE, RMSE
       rows.append({family, granularity, metrics, ...})
   return pd.DataFrame(rows)
   ```

2. **Extract family-specific logic into adapters:**
   - Prophet adapter: `predict()` on future regressor frame
   - CatBoost adapter: `predict()` on feature matrix
   - SARIMAX adapter: `get_forecast()` from fitted results

3. **Reuse rolling-origin and tie-breaker logic:**
   - Rolling-origin metrics (`evaluate_monthly_prophet_rolling_origin_metrics`) can stay Prophet-specific if other families don't need horizon metrics
   - Tie-breaker logic in `select_monthly_prophet_champion` is reusable (any metric can be primary/secondary)

**Cost:** Implement generic `dispatch_predict(family, model, test_data) → predictions` utility + `evaluate_candidates_on_test` dispatcher. Keep Prophet-specific rolling-origin as optional enhancement.

---

## 8. Forecast Inference Findings

### Current Implementation

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/src/hdf_pipelines/pipelines/forecast_inference/nodes.py`

**Node:** `generate_monthly_prophet_forecasts`

**Hardcoded Prophet Assumptions:**
1. **Catalog inputs:** `monthly_prophet_champion_model`, `monthly_prophet_champion_metadata`, `monthly_prophet_future_3m/6m/12m`
2. **Column naming:** Assumes `ds` (date), `yhat` (point forecast), `yhat_lower`/`yhat_upper` (intervals)
3. **Metadata schema:** Requires `champion_id`, `active_regressors`, `model_family`, `selection_metric`, etc.
4. **Inference call:** Direct `model.predict(future_df[['ds'] + active_regressors])`
5. **Output schema:** Hardcoded `_FORECAST_COLUMN_ORDER` with Prophet-specific columns

### Is It Hardcoded to Prophet?

**Strongly yes:**
- Line 12: Model object type-checked as Prophet (implicit in `predict()` signature)
- Lines 89–90: Hardcoded date/sku column names from params
- Lines 336–343: `_build_prophet_prediction_input` extracts `ds` + regressors (Prophet-specific)
- Lines 449–519: `_forecast_one_horizon` calls Prophet `predict()` directly
- Line 488: Assumes `yhat`, `yhat_lower`, `yhat_upper` columns exist

### Minimum Changes for Multi-Family

1. **Load champion metadata-driven:**
   ```python
   champion = load(champion_monthly_model)
   metadata = load(champion_monthly_metadata)
   model_family = metadata['model_family']
   ```

2. **Dispatch prediction based on metadata:**
   ```python
   if model_family == 'prophet':
       predictions = prophet_predict_adapter(champion, future_df, active_regressors)
   elif model_family == 'catboost':
       predictions = catboost_predict_adapter(champion, future_df, active_regressors)
   elif model_family == 'sarimax':
       predictions = sarimax_predict_adapter(champion, future_df, active_regressors)
   ```

3. **Family-specific adapters produce standard output:**
   - All adapters return DataFrame with columns: `[date, yhat, yhat_lower, yhat_upper, model_family, ...]`
   - Prophet adapter renames `yhat` (Prophet native)
   - CatBoost adapter computes intervals via quantile regression or bootstrap (not native)
   - SARIMAX adapter extracts `yhat` from `get_forecast()` result

4. **Merge with metadata:**
   ```python
   forecast = add_forecast_metadata(
       forecast_df,
       champion_metadata,
       model_family,
       ...
   )
   ```

**Core logic that remains generic:**
- Horizon calculation (`_add_horizon_month`)
- Metadata annotation (`_add_forecast_metadata`)
- Output validation + column reordering
- Run ID and timestamp generation

**What moves into Prophet adapter:**
- `_build_prophet_prediction_input` (Prophet-specific column extraction)
- `_forecast_one_horizon` (Prophet `predict()` call)

---

## 9. Catalog and Parameters Findings

### Catalog Entry Organization

**Current Grouping:**

**Generic monthly features (model-agnostic):**
- `monthly_calendar_features` (04_feature)
- `monthly_exogenous_features` (04_feature)

**Prophet-specific features:**
- `monthly_prophet_features` (04_feature) — combines above + Prophet formatting
- `monthly_prophet_modeling_data` (05_model_input) — calendar + exogenous + demand with Prophet column names
- `monthly_prophet_train/validation/test/full_train` (05_model_input)
- `monthly_prophet_future_3m/6m/12m` (05_model_input)
- `monthly_prophet_split_metadata` (05_model_input)

**Training artifacts:**
- `monthly_prophet_tuning_results`, `validation_metrics`, `prechampion_configs`, `candidate_models`, `training_metadata` (06_models/tuning)
- `candidate_monthly_prophet` (06_models/candidates)

**Selection artifacts:**
- `monthly_prophet_test_metrics`, `model_selection_summary`, `champion_test_forecast` (06_models/selection)
- `monthly_prophet_champion_model`, `monthly_prophet_champion_metadata` (06_models/champions)

**Inference artifacts:**
- `monthly_prophet_forecast_3m/6m/12m/latest` (07_model_output)
- `monthly_prophet_inference_metadata` (07_model_output)

**Orphaned stubs (no pipeline nodes):**
- `model_input_monthly_catboost_train/validation/test` (05_model_input)
- `model_input_monthly_sarimax_train/validation/test` (05_model_input)
- `candidate_monthly_catboost`, `candidate_monthly_sarimax` (06_models/candidates)
- `champion_monthly_model`, `champion_weekly_model`, `champion_registry` (06_models/champions) — generic names but Prophet data

### Parameter File Organization

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/conf/base/parameters/`

**train_monthly.yml:**
```yaml
train_monthly:
  prophet:
    enabled: true
    active_regressors: [...]
    tuning:
      optimizer: optuna
      objective:
        metric: wape
        direction: minimize
      max_trials: 30
      ...
    metrics:
      epsilon: 1.0
      mase_seasonal_period: 12
      business_success_precision_threshold: 0.85
      horizon_metrics:
        enabled: true
        horizons: [2, 3]
  
  catboost:
    enabled: true
    tuning:
      iterations: [200, 500, 1000]
      learning_rate: [0.01, 0.05, 0.1]
      depth: [4, 6, 8]
  
  sarimax:
    enabled: true
    tuning:
      order_grid: [...]
      seasonal_order_grid: [...]
```

**model_selection.yml:**
```yaml
model_selection:
  primary_metric: WAPE
  secondary_metrics: [MASE, RMSE]
  selection_strategy: best_by_horizon
  tie_breaker: RMSE
  
  monthly_prophet:
    date_column: ds
    target_column: y
    sku_column: sku
    selection:
      primary_metric: wape
      tie_breakers: [test_m3_wape, test_m2_wape, mase, rmse]
      ...
```

**Missing blocks:**
- `model_input_preparation.monthly_catboost`
- `model_input_preparation.monthly_sarimax`
- `model_selection.monthly_catboost`
- `model_selection.monthly_sarimax`

### Recommended Reorganization

**Create `parameters/model_input.yml` (new):**
```yaml
model_input_preparation:
  monthly:
    # Generic monthly layer
    date_column: month_start_date
    target_column: monthly_demand
    sku_column: sku
    split:
      train_end: "2023-12-31"
      validation_end: "2024-06-30"
      test_end: null
  
  monthly_prophet:
    # Prophet-specific overrides + adapter config
    output_columns:
      date: ds
      target: y
    active_regressors: [...]
    regressor_mode: additive
  
  monthly_catboost:
    # CatBoost-specific adapter config
    categorical_features: []
    numeric_features: [...]
  
  monthly_sarimax:
    # SARIMAX-specific config
    differencing: [1, 1]  # order=(1, 1, ...)
    seasonal_differencing: [1, 12]
```

**Create `parameters/train_monthly.yml` (reorganized):**
```yaml
train_monthly:
  # Generic monthly training config (shared across families)
  target_column: y
  date_column: ds
  sku_column: sku
  metrics:
    shared:
      epsilon: 1.0
      mase_seasonal_period: 12
      business_success_precision_threshold: 0.85
  
  prophet:
    enabled: true
    active_regressors: [...]
    tuning:
      optimizer: optuna
      objective: {metric: wape, direction: minimize}
      max_trials: 30
      top_n_prechampions: 3
      sampler: {name: tpe, seed: 42}
      search_space:
        changepoint_prior_scale: {type: float, low: 0.01, high: 0.5, log: true}
        seasonality_prior_scale: {type: float, low: 1.0, high: 10.0, log: true}
        ...
      fixed_params:
        yearly_seasonality: true
        weekly_seasonality: false
        daily_seasonality: false
        interval_width: 0.8
    regressors:
      mode: additive
    metrics:
      horizon_metrics: {enabled: true, horizons: [2, 3]}
  
  catboost:
    enabled: true
    tuning:
      iterations: [200, 500, 1000]
      learning_rate: [0.01, 0.05, 0.1]
      depth: [4, 6, 8]
    metrics:
      # Can override shared metrics here
  
  sarimax:
    enabled: true
    tuning:
      order_grid: [[1, 1, 1], [2, 1, 1], [1, 1, 2]]
      seasonal_order_grid: [[1, 1, 1, 12], [0, 1, 1, 12]]
```

---

## 10. Seasonal Naive / MASE Baseline Findings

### Current Implementation

**Location:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/shared/src/shared/metrics.py` lines 74–119

**Function:** `mase(y_true, y_pred, y_train, seasonality=12)`

**What it does:**
- Computes Mean Absolute Scaled Error
- Scales forecast MAE by in-sample seasonal naive benchmark: `mean(|y_train[t] - y_train[t-seasonality]|)`
- Result < 1 means model beats naive seasonal walk

**Where it's used:**
1. **Train monthly Prophet:** `train_and_evaluate_monthly_prophet_candidates` (line 513)
   - Called with `mase_seasonal_period=12` (monthly data)
   - Baseline: MAE of seasonal walk at lag-12
2. **Model selection Prophet:** Included in metrics DataFrame and tie-breaker list
3. **Shared metric suite:** Available to any model family

### Is It Safe for Multi-Family Use?

**Yes, with one caveat:**

- **Seasonal period must be consistent** across all model families for fair comparison
- Current Prophet tuning uses `mase_seasonal_period: 12` (from params)
- CatBoost/SARIMAX must use the same `12` when computing MASE on monthly data

**Risk:** If different families are tuned with different seasonal periods, MASE values are not comparable.

**Mitigation:** Store `mase_seasonal_period` in `model_input_preparation.monthly` parameters (generic layer), propagate to all family-specific training blocks.

### Seasonal Naive NOT Treated as a Candidate

**Good news:** There is no explicit seasonal naive model candidate in the codebase. MASE is used only as an *evaluation metric*, not as a model artifact.

**Risk:** If downstream code treats MASE = 1.0 as a threshold for model viability, that's implicit comparison to naive baseline, which is acceptable. No architectural risk here.

### What to Preserve

1. **MASE function signature:** Keep as-is (numpy arrays, explicit seasonality)
2. **Seasonal period parameter:** Ensure it flows from generic `model_input_preparation.monthly.mase_seasonal_period` → all family tuning blocks
3. **MASE computation in all training nodes:** CatBoost/SARIMAX tuning should call `shared.metrics.mase()` with same seasonal period as Prophet

---

## 11. Test Coverage and Safety Checks

### Protected Tests (Prophet MVP)

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/tests/test_pipeline_registry.py`

1. `test_all_expected_pipelines_registered` — validates all pipeline keys exist
2. `test_monthly_mvp_is_registered` — MVP is present
3. `test_default_is_monthly_mvp` — default pipeline = MVP
4. `test_default_has_no_scaffolded_weekly_nodes` — MVP excludes experimental nodes
5. `test_default_pipeline_is_not_empty` — has nodes

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/tests/test_train_monthly_prophet_nodes.py`

1. `test_train_and_evaluate_monthly_prophet_candidates_ranks_trials_and_emits_metadata` — tuning produces correct outputs
2. `test_monthly_prophet_optuna_outputs_preserve_stage4_contract` — outputs have required columns/schema

**File:** `/Users/jhonattan.reales/Documents/icesi/innovatec3/hierarchical-demand-forecasting-poc/pipelines/tests/test_forecast_inference_nodes.py` (exists but not shown)

**Coverage:** Feature engineering, model input preparation, forecast inference nodes also have tests.

### What Smoke Tests Should Exist (Before SARIMAX/CatBoost Refactoring)

1. **Monthly MVP still works:**
   - Run `monthly_mvp` pipeline to completion
   - Verify `monthly_prophet_forecast_latest` is produced
   - Check forecast shape, column names, date range

2. **Generic monthly datasets exist:**
   - `monthly_train`, `monthly_validation`, `monthly_test` can be loaded
   - All required columns present
   - No null values in key columns

3. **Prophet adapter works:**
   - `monthly_train` → `monthly_prophet_train` conversion preserves rows
   - Column rename (date_col → `ds`, target_col → `y`) succeeds
   - Regressor frame has correct shape

4. **Metric functions are importable:**
   - `shared.metrics.wape`, `mase`, `rmse` all callable
   - MASE returns sensible values (< inf) for monthly data

5. **Catalog entries match pipeline expectations:**
   - All hardcoded catalog names in nodes exist in `catalog.yml`
   - No broken dependencies

### Minimal Test Checklist Before Implementing SARIMAX

- [ ] `test_monthly_generic_datasets_exist`
- [ ] `test_prophet_adapter_preserves_rows`
- [ ] `test_shared_metrics_all_importable`
- [ ] `test_monthly_mvp_produces_forecast`
- [ ] `test_catalog_no_orphaned_entries`

---

## 12. Recommended Refactor Plan

### Phase 1: Create Generic Monthly Foundation (2–3 days)

**Goal:** Decouple Prophet from model_input_preparation stage.

**Changes:**

1. **Rename `model_input_preparation.py` inputs/outputs to generic:**
   - `demand_monthly` + `exogenous_monthly` → shared inputs
   - Create intermediate `monthly_modeling_data` (calendar + exogenous + demand with generic column names)
   - Create `monthly_train`, `monthly_validation`, `monthly_test` splits

2. **Refactor `build_monthly_prophet_features`:**
   - Keep as separate node but name it appropriately for its output
   - Clarify that it's a Prophet-specific adapter

3. **Create `model_input.yml` parameter file:**
   - Add `model_input_preparation.monthly` (generic split config)
   - Add `model_input_preparation.monthly_prophet` (Prophet adapter config)

4. **Create Prophet adapter node:**
   - Input: `monthly_train/validation/test`
   - Output: `monthly_prophet_train/validation/test`
   - Logic: rename columns to `ds`/`y`

5. **Add tests:**
   - `test_monthly_split_produces_expected_rows`
   - `test_prophet_adapter_column_rename`

**Risk:** Breaking change to `monthly_prophet_*` dataset consumers. Mitigate by keeping catalog entries but updating pipeline to produce them via adapter.

**Status:** Can merge to `main` without breaking MVP because inputs are refactored at pipeline level; catalog entries remain the same.

---

### Phase 2: Generalize Feature Engineering (1–2 days)

**Goal:** Separate model-agnostic features from Prophet-specific column naming.

**Changes:**

1. **Rename `build_monthly_prophet_features`:**
   - New name: `build_monthly_model_features` or leave as-is but add docstring clarification

2. **Document output schema explicitly:**
   - Calendar features (numeric, date-aware)
   - Exogenous features (numeric, with lags)
   - Derived features (stress, uplift)
   - All column names are model-agnostic (not `ds`/`y`)

3. **Update `parameters/feature_engineering.yml`:**
   - Add block: `feature_engineering_monthly.shared` (all families use this)
   - Prophet-specific tuning stays in `train_monthly.prophet.active_regressors`

**Risk:** None — feature engineering output is consumed only by `model_input_preparation` node, which will be refactored in Phase 1.

---

### Phase 3: Implement SARIMAX/CatBoost Model Input Adapters (1–2 days)

**Goal:** Add SARIMAX and CatBoost to the model_input_preparation stage.

**Changes:**

1. **Create `model_input_preparation/adapters.py` (optional):**
   - `build_monthly_sarimax_data(monthly_train, monthly_validation, monthly_test, params)` → (train, validation, test)
   - `build_monthly_catboost_data(monthly_train, monthly_validation, monthly_test, params)` → (train, validation, test)

2. **Add two nodes to `model_input_preparation/pipeline.py`:**
   - `prepare_monthly_sarimax_data`
   - `prepare_monthly_catboost_data`

3. **Update `parameters/model_input.yml`:**
   - Add `model_input_preparation.monthly_sarimax`
   - Add `model_input_preparation.monthly_catboost`

4. **Tests:**
   - `test_sarimax_adapter_produces_correct_shape`
   - `test_catboost_adapter_all_numeric`

**Status:** Creates `model_input_monthly_sarimax_train/validation/test` and `model_input_monthly_catboost_train/validation/test` as non-orphans.

---

### Phase 4: Implement CatBoost/SARIMAX Training Nodes (3–5 days per family)

**Goal:** Fill in `NotImplementedError` stubs in CatBoost and SARIMAX training.

**Changes:**

1. **CatBoost training:**
   - Implement `tune_hyperparameters` — grid search over iterations/learning_rate/depth
   - Implement `train_best_candidate` — refit on train+validation

2. **SARIMAX training:**
   - Implement `tune_hyperparameters` — grid search over order/seasonal_order
   - Implement `train_best_candidate` — refit on train+validation

3. **Ensure both families produce:**
   - `monthly_<family>_tuning_results`
   - `monthly_<family>_validation_metrics`
   - `monthly_<family>_prechampion_configs`
   - `monthly_<family>_candidate_models`
   - `monthly_<family>_training_metadata`
   - `candidate_monthly_<family>`

4. **Tests:**
   - `test_catboost_tuning_produces_expected_artifacts`
   - `test_sarimax_tuning_produces_expected_artifacts`
   - `test_all_families_produce_candidate_models`

**Status:** Creates fully functional training pipelines for all three families.

---

### Phase 5: Implement Multi-Family Model Selection (2–3 days)

**Goal:** Replace generic `NotImplementedError` nodes with functional dispatcher.

**Changes:**

1. **Implement `evaluate_candidates_on_test`:**
   - Load all 6 candidates (monthly/weekly × 3 families)
   - Dispatch predict based on family type
   - Compute shared metrics (WAPE, MASE, RMSE)
   - Return evaluation DataFrame

2. **Implement `select_champion_models`:**
   - Filter evaluation report by granularity
   - Rank by primary metric + apply tie-breakers
   - Return rank-1 model object for each granularity

3. **Implement `persist_champion_registry`:**
   - Build `champion_registry.json` mapping granularity → family → champion info

4. **Tests:**
   - `test_evaluate_candidates_computes_all_metrics`
   - `test_select_champion_ranking_correct`
   - `test_multi_family_champion_registry_structure`

**Status:** Enables Prophet/CatBoost/SARIMAX to compete fairly at model selection stage.

---

### Phase 6: Implement Multi-Family Forecast Inference (2–3 days)

**Goal:** Replace Prophet-specific inference with metadata-driven dispatch.

**Changes:**

1. **Refactor `generate_monthly_prophet_forecasts`:**
   - Load from generic `champion_monthly_model` + `champion_monthly_metadata`
   - Dispatch prediction based on `metadata['model_family']`
   - Call family-specific adapter

2. **Create inference adapters:**
   - `prophet_predict_adapter(model, future_df, active_regressors) → forecast_df`
   - `catboost_predict_adapter(model, future_df, active_regressors) → forecast_df`
   - `sarimax_predict_adapter(model, future_df, active_regressors) → forecast_df`

3. **Update catalog:**
   - Load from `champion_monthly_model` (generic) instead of `monthly_prophet_champion_model`
   - Load metadata from `champion_monthly_metadata` (generic)

4. **Tests:**
   - `test_forecast_inference_loads_champion_metadata`
   - `test_dispatch_selects_correct_adapter_by_family`
   - `test_all_families_produce_consistent_forecast_schema`

**Status:** Inference works for any elected champion family.

---

### Phase 7: Documentation and Polish (1 day)

**Goal:** Ensure refactored code is maintainable and understandable.

**Changes:**

1. **Update CLAUDE.md:**
   - Document family-specific vs generic layers
   - Explain namespace structure
   - Add diagram of data flow

2. **Docstring updates:**
   - Mark all `_<family>_adapter` functions clearly
   - Update parameter descriptions

3. **Changelog:**
   - Record breaking changes and migration steps

---

## 13. Suggested Target Structure

After all phases, the codebase should look like this:

```
pipelines/src/hdf_pipelines/pipelines/

├── data_ingestion/
│   ├── pipeline.py (unchanged)
│   └── nodes.py (unchanged)

├── feature_engineering_monthly/
│   ├── pipeline.py (unchanged, but outputs renamed for clarity)
│   └── nodes.py (unchanged)

├── model_input_preparation/
│   ├── pipeline.py (refactored)
│   │   ├── Generic monthly split node
│   │   ├── Prophet adapter node
│   │   ├── CatBoost adapter node
│   │   └── SARIMAX adapter node
│   ├── nodes.py (refactored)
│   ├── adapters.py (new)
│   │   ├── build_monthly_catboost_data()
│   │   ├── build_monthly_sarimax_data()
│   │   └── build_monthly_prophet_features()
│   └── tests/

├── train_monthly/
│   ├── pipeline.py (unchanged, composition still works)
│   ├── nodes.py (unchanged, generic utilities)
│   ├── prophet/
│   │   ├── pipeline.py (unchanged)
│   │   ├── nodes.py (unchanged)
│   │   └── tests/
│   ├── catboost/
│   │   ├── pipeline.py (unchanged)
│   │   ├── nodes.py (IMPLEMENTED)
│   │   └── tests/
│   └── sarimax/
│       ├── pipeline.py (unchanged)
│       ├── nodes.py (IMPLEMENTED)
│       └── tests/

├── model_selection/
│   ├── pipeline.py (unchanged, inputs still hardcoded but now functional)
│   ├── nodes.py (IMPLEMENTED)
│   │   ├── evaluate_candidates_on_test() — dispatcher
│   │   ├── select_champion_models() — ranker
│   │   └── persist_champion_registry() — serializer
│   ├── prophet/
│   │   ├── pipeline.py (unchanged)
│   │   ├── nodes.py (unchanged)
│   │   └── tests/
│   └── tests/

├── forecast_inference/
│   ├── pipeline.py (REFACTORED to load from champion_monthly_model)
│   ├── nodes.py (REFACTORED)
│   │   ├── generate_monthly_prophet_forecasts() — dispatcher
│   │   └── adapters/
│   │       ├── prophet_adapter.py (predict + column handling)
│   │       ├── catboost_adapter.py (predict + intervals)
│   │       └── sarimax_adapter.py (forecast + intervals)
│   ├── stubs/
│   │   ├── weekly_inference.py (unchanged)
│   │   └── daily_allocation.py (unchanged)
│   └── tests/

└── [other stages unchanged]

conf/base/parameters/

├── train_monthly.yml (UPDATED: split into shared + family blocks)
├── model_input.yml (NEW: generic + family-specific adapters)
├── model_selection.yml (UPDATED: add family-specific blocks)
├── evaluation.yml (unchanged)
├── feature_engineering.yml (unchanged)
├── forecast_inference.yml (UPDATED: add family-specific blocks)
└── [other parameter files]

conf/base/

└── catalog.yml (REORGANIZED: generic + family-specific entries)
    ├── 03_primary (unchanged)
    ├── 04_feature (clarified output schema)
    ├── 05_model_input
    │   ├── monthly_* (generic splits)
    │   ├── monthly_prophet_* (Prophet adapter outputs)
    │   ├── monthly_catboost_* (CatBoost adapter outputs)
    │   ├── monthly_sarimax_* (SARIMAX adapter outputs)
    │   └── [same for weekly]
    ├── 06_models
    │   ├── tuning: monthly_<family>_*, etc.
    │   ├── candidates: candidate_monthly_<family>
    │   └── champions: champion_monthly_model, champion_monthly_metadata (generic)
    └── 07_model_output
        ├── monthly_prophet_forecast_* (Prophet inference outputs)
        ├── monthly_catboost_forecast_* (future)
        └── [generic champion metadata path]
```

---

## 14. Open Questions / Decisions Needed

### 1. Prophet Namespace Question

**Question:** Should Prophet be refactored to use `namespace="train_monthly.prophet"` like CatBoost/SARIMAX, or keep it as direct composition?

**Options:**
- A: Apply namespace to all three families (consistent, cleaner Kedro-Viz)
- B: Remove namespace from CatBoost/SARIMAX and use long explicit names (more verbose, avoids refactoring Prophet)

**Recommendation:** Option A. Refactor all three to use namespaces.

**Decision Needed:** Architecture decision — consistency vs. minimal change.

---

### 2. Generic Monthly Layer Naming

**Question:** Call the intermediate split datasets `monthly_*` or `monthly_raw_*`?

**Options:**
- A: `monthly_train`, `monthly_validation`, `monthly_test` (simple, assumes "monthly" means pre-split)
- B: `monthly_raw_train`, `monthly_raw_validation`, `monthly_raw_test` (clarifies "raw" = before family adaptation)

**Recommendation:** Option A — simpler naming, context is clear from pipeline stage.

**Decision Needed:** Catalog entry naming convention.

---

### 3. Prophet Adapter Node Naming

**Question:** Rename `build_monthly_prophet_features` to `prepare_monthly_prophet_data` or similar?

**Current Name:** `build_monthly_prophet_features` (misleading — doesn't build features, consumes them)

**Options:**
- A: Keep as-is, add clarification in docstring
- B: Rename to `prepare_monthly_prophet_data`
- C: Rename to `adapt_monthly_data_for_prophet`

**Recommendation:** Option B — clearer intent.

**Decision Needed:** Naming consistency for family adapters.

---

### 4. Should Horizon Metrics be Generic?

**Question:** The `evaluate_monthly_prophet_rolling_origin_metrics` computes horizon-specific metrics (M-2, M-3). Should this be refactored to support any family?

**Options:**
- A: Keep Prophet-specific for now; SARIMAX/CatBoost don't compute horizon metrics yet
- B: Generalize immediately by extracting rolling-window logic into shared utility

**Recommendation:** Option A — defer. Prophet is the only family that uses horizon metrics currently. Once SARIMAX/CatBoost need them, generalize.

**Decision Needed:** Phasing of horizon metric generalization.

---

### 5. Multi-Family Champion Registry Schema

**Question:** Should the champion registry store all family results or only the elected champion?

**Current design:** Generic `persist_champion_registry` stub likely stores all candidates + elected champion.

**Options:**
- A: Store all six candidates (Prophet/CatBoost/SARIMAX × monthly/weekly) for audit trail
- B: Store only elected champions per granularity (minimal, production-ready)

**Recommendation:** Option B for now — focuses inference on one champion. If audit trail is needed, add it as a separate reporting artifact.

**Decision Needed:** Registry scope and schema.

---

### 6. Confidence Intervals for Non-Prophet Families

**Question:** Prophet natively produces `yhat_lower`/`yhat_upper`. CatBoost/SARIMAX don't. How should inference handle this?

**Options:**
- A: All families compute quantile-based intervals (slower)
- B: Only Prophet produces intervals; other families output `yhat` only
- C: Use bootstrapped intervals for CatBoost; statsmodels intervals for SARIMAX

**Recommendation:** Option B for MVP — simplest. Intervals are a feature, not a requirement. If needed later, Option C is modular.

**Decision Needed:** Confidence interval strategy per family.

---

### 7. Breaking Change to `monthly_prophet_*` Catalog Entries

**Question:** Phase 1 refactoring creates an intermediate `monthly_*` layer. Should existing code consuming `monthly_prophet_train` directly need to change?

**Current:** Some nodes directly depend on `monthly_prophet_train`.

**Impact:** Introducing `monthly_*` doesn't break existing dependencies if we keep both. Prophet adapter produces `monthly_prophet_*` as outputs.

**Recommendation:** Keep both for backward compatibility. Phase 2 can deprecate if needed.

**Decision Needed:** Backward compatibility strategy.

---

### 8. Weekly Inclusion in Model Selection

**Question:** Current generic model selection pipeline expects monthly AND weekly candidates. Should we implement weekly support simultaneously with multi-family monthly?

**Options:**
- A: Focus only on monthly (3 families). Weekly is future work.
- B: Implement both monthly and weekly (6 families total).

**Recommendation:** Option A — scope reduction. Weekly is scaffolded and can be added after monthly is solid.

**Decision Needed:** Product scope for initial multi-family release.

---

## Summary of Immediate Action Items

**Before implementing SARIMAX/CatBoost, complete these in order:**

1. ✓ Read and understand Prophet MVP end-to-end (already done)
2. **Phase 1:** Refactor `model_input_preparation` to create generic `monthly_*` layer + Prophet adapter (est. 2 days)
3. **Phase 2:** Clarify feature engineering output schema (est. 1 day)
4. **Phase 3:** Add SARIMAX/CatBoost model input adapters (est. 2 days)
5. **Phase 4:** Implement CatBoost/SARIMAX training nodes (est. 1 week)
6. **Phase 5:** Implement multi-family model selection (est. 3 days)
7. **Phase 6:** Implement metadata-driven forecast inference (est. 3 days)
8. **Testing:** Add smoke tests for each phase (ongoing, est. 2 days total)

**Total Estimated Effort:** 3–4 weeks (engineering + testing + review).

**Key Preservation:** `monthly_mvp` pipeline must remain passing and production-ready at all times. Use feature branches and thorough testing to avoid breaks.
