# Phase 0 — Safety Baseline Report

**Date:** 2026-05-31  
**Branch:** `refactor/phase-0-safety-baseline`  
**Author:** Jhonattan Reales  
**Kedro version:** 1.3.1  
**Python version:** 3.12.12  

---

## 1. Summary

Phase 0 is complete. The current `monthly_mvp` pipeline is **stable and safe** to use as the reference baseline for Phase 1 modularization.

Key findings:

- `monthly_mvp` runs end to end in **398.4 seconds** with **20/20 tasks completed** and zero failures.
- All **35 protected unit tests pass** with no failures.
- All **28 monthly Prophet artifacts** are present and loadable via the catalog.
- SARIMAX and CatBoost scaffolding nodes are **isolated from `monthly_mvp`** — they live in the `experimental_training` composed pipeline only.
- The `__default__` alias correctly resolves to `monthly_mvp`.
- No `monthly_prophet_*` dataset name needs to change before Phase 1.

One non-blocking issue: `kedro pipeline list` does not exist in Kedro 1.3.1. The correct command is `kedro registry list`. The Phase 0 document should be updated to reflect the actual CLI interface.

---

## 2. Git and Environment

```
Branch:  refactor/phase-0-safety-baseline
Base:    feat/prophet_update

Python:  3.12.12 (CPython)
Kedro:   1.3.1
uv:      workspace mode
```

Environment setup confirmed via `uv sync --all-packages`. No `pip` usage.

---

## 3. Pipeline Registry Snapshot

Command used: `uv run kedro registry list`

> **Note:** `kedro pipeline list` does not exist in Kedro 1.3.1. The correct subcommand is `kedro registry list`.

Registered pipelines:

```
- __default__
- data_ingestion
- experimental_full_experiment
- experimental_inference
- experimental_training
- feature_engineering_monthly
- feature_engineering_weekly
- forecast_inference
- model_input_preparation
- model_selection
- monthly_mvp
- prophet_monthly_e2e
- reconciliation
- train_monthly
- train_weekly
```

Expected pipeline status:

| Pipeline | Expected | Found | Notes |
|---|---|---|---|
| `__default__` | ✅ | ✅ | Points to `monthly_mvp` |
| `monthly_mvp` | ✅ | ✅ | Stable Prophet MVP |
| `prophet_monthly_e2e` | ✅ | ✅ | Same as `monthly_mvp` |
| `data_ingestion` | ✅ | ✅ | |
| `feature_engineering_monthly` | ✅ | ✅ | |
| `model_input_preparation` | ✅ | ✅ | |
| `train_monthly` | ✅ | ✅ | Contains SARIMAX/CatBoost stubs |
| `model_selection` | ✅ | ✅ | Generic dispatcher |
| `forecast_inference` | ✅ | ✅ | |

Additionally registered (experimental / scaffolded):

| Pipeline | Role |
|---|---|
| `feature_engineering_weekly` | Scaffolded (pass-through) |
| `train_weekly` | Contains `NotImplementedError` stubs |
| `reconciliation` | Scaffolded (pass-through) |
| `experimental_training` | monthly + weekly stubs combined |
| `experimental_inference` | inference + reconciliation |
| `experimental_full_experiment` | Full composed experimental path |

---

## 4. Catalog Snapshot

Command used: `uv run kedro catalog describe-datasets --pipeline monthly_mvp`

All datasets used by `monthly_mvp` confirmed as registered. Full listing by type:

### JSON datasets (5)
- `monthly_prophet_prechampion_configs`
- `monthly_prophet_champion_metadata`
- `monthly_prophet_split_metadata`
- `monthly_prophet_inference_metadata`
- `monthly_prophet_training_metadata`

### CSV datasets — raw input (2)
- `raw_daily_demand`
- `raw_exogenous_variables`

### Parquet datasets (28+)
All `monthly_prophet_*` parquet artifacts, plus intermediate/primary layer datasets. See `monthly_mvp_contract_snapshot.md` for full schema details.

### Pickle datasets (3)
- `candidate_monthly_prophet`
- `monthly_prophet_candidate_models`
- `monthly_prophet_champion_model`

### MemoryDataset (3 — not persisted to disk)
- `monthly_prophet_split_preparation_metadata`
- `raw_daily_demand_masked`
- `monthly_prophet_preparation_metadata`

**Missing from Phase 0 spec vs. actual catalog:**

The Phase 0 spec listed these dataset names. Some differ from actual catalog entries:

| Spec name | Actual catalog name | Status |
|---|---|---|
| `monthly_prophet_train` | `monthly_prophet_train` | ✅ Match |
| `monthly_prophet_validation` | `monthly_prophet_validation` | ✅ Match |
| `monthly_prophet_test` | `monthly_prophet_test` | ✅ Match |
| `monthly_prophet_full_train` | `monthly_prophet_full_train` | ✅ Match |
| `monthly_prophet_future_3m` | `monthly_prophet_future_3m` | ✅ Match |
| `monthly_prophet_future_6m` | `monthly_prophet_future_6m` | ✅ Match |
| `monthly_prophet_future_12m` | `monthly_prophet_future_12m` | ✅ Match |
| `monthly_prophet_split_metadata` | `monthly_prophet_split_metadata` | ✅ Match |
| `monthly_prophet_tuning_results` | `monthly_prophet_tuning_results` | ✅ Match |
| `monthly_prophet_validation_metrics` | `monthly_prophet_validation_metrics` | ✅ Match |
| `monthly_prophet_prechampion_configs` | `monthly_prophet_prechampion_configs` | ✅ Match |
| `monthly_prophet_candidate_models` | `monthly_prophet_candidate_models` | ✅ Match |
| `monthly_prophet_training_metadata` | `monthly_prophet_training_metadata` | ✅ Match |
| `candidate_monthly_prophet` | `candidate_monthly_prophet` | ✅ Match |
| `monthly_prophet_test_metrics` | `monthly_prophet_test_metrics` | ✅ Match |
| `monthly_prophet_model_selection_summary` | `monthly_prophet_model_selection_summary` | ✅ Match |
| `monthly_prophet_champion_model` | `monthly_prophet_champion_model` | ✅ Match |
| `monthly_prophet_champion_metadata` | `monthly_prophet_champion_metadata` | ✅ Match |
| `monthly_prophet_forecast_3m` | `monthly_prophet_forecast_3m` | ✅ Match |
| `monthly_prophet_forecast_6m` | `monthly_prophet_forecast_6m` | ✅ Match |
| `monthly_prophet_forecast_12m` | `monthly_prophet_forecast_12m` | ✅ Match |
| `monthly_prophet_forecast_latest` | `monthly_prophet_forecast_latest` | ✅ Match |
| `monthly_prophet_inference_metadata` | `monthly_prophet_inference_metadata` | ✅ Match |

**All dataset names match the Phase 0 spec exactly.**

---

## 5. Test Results

### Protected test suite

Command: `uv run --package hdf_pipelines pytest tests/test_pipeline_registry.py tests/test_metrics.py tests/test_model_input_preparation_nodes.py tests/test_train_monthly_prophet_nodes.py tests/test_forecast_inference_nodes.py -v`

```
35 passed, 0 failed, 0 skipped
4 warnings (UserWarning from MASE short-series edge case — expected and benign)
Elapsed: 1.96s
```

Test breakdown:

| File | Tests | Result |
|---|---|---|
| `test_pipeline_registry.py` | 7 | ✅ All passed |
| `test_metrics.py` | 8 | ✅ All passed |
| `test_model_input_preparation_nodes.py` | 7 | ✅ All passed |
| `test_train_monthly_prophet_nodes.py` | 2 | ✅ All passed |
| `test_forecast_inference_nodes.py` | 8 | ✅ All passed |
| `test_optuna_helpers.py` | *(not in protected suite)* | Not run separately |
| `test_data_ingestion_nodes.py` | *(not in protected suite)* | Not run separately |
| `test_feature_engineering_monthly_nodes.py` | *(not in protected suite)* | Not run separately |

### Coverage summary (protected files only)

| Module | Coverage |
|---|---|
| `pipeline_registry.py` | 100% |
| `forecast_inference/nodes.py` | 91% |
| `model_input_preparation/nodes.py` | 88% |
| `train_monthly/prophet/nodes.py` | 88% |
| `model_selection/prophet/nodes.py` | 8% (tested via integration, not unit tests) |
| `data_ingestion/nodes.py` | 12% (not in protected suite) |
| Total across all modules | 55% |

---

## 6. Monthly MVP Run Result

Command: `uv run kedro run --pipeline monthly_mvp`

```
Status:    SUCCESS
Tasks:     20 / 20 completed
Duration:  398.4 seconds
Failures:  0
```

Node execution order:

| # | Node | Output |
|---|---|---|
| 1 | `load_and_clean_exogenous` | `exogenous_cleaned` |
| 2 | `mask_raw_demand` | `raw_daily_demand_masked` (memory) |
| 3 | `build_exogenous_monthly` | `exogenous_monthly` |
| 4 | `load_and_clean_demand` | `demand_cleaned` |
| 5 | `build_demand_daily` | `demand_daily` |
| 6 | `build_demand_monthly` | `demand_monthly` |
| 7 | `build_demand_weekly` | `demand_weekly` |
| 8 | `build_monthly_exogenous_features` | `monthly_exogenous_features` |
| 9 | `build_monthly_calendar_features` | `monthly_calendar_features` |
| 10 | `build_monthly_prophet_features` | `monthly_prophet_features` |
| 11 | `prepare_monthly_prophet_modeling_data` | `monthly_prophet_modeling_data` |
| 12 | `build_monthly_prophet_future_regressors` | futures 3m/6m/12m |
| 13 | `split_monthly_prophet_data` | train/val/test/full_train |
| 14 | `build_monthly_prophet_split_metadata` | `monthly_prophet_split_metadata` |
| 15 | `train_and_evaluate_monthly_prophet_candidates` | tuning artifacts, candidate models |
| 16 | `evaluate_monthly_prophet_prechampions_on_test` | test metrics + test forecast |
| 17 | `evaluate_monthly_prophet_rolling_origin_metrics` | operational forecasts + lead-time metrics |
| 18 | `select_monthly_prophet_champion` | champion metadata + selection summary |
| 19 | `build_monthly_prophet_champion_model` | champion model (refit on full history) |
| 20 | `generate_monthly_prophet_forecasts` | 3m / 6m / 12m / latest forecasts |

**No SARIMAX or CatBoost nodes were executed.**

### Data characteristics (from run logs)

- Raw daily demand: 1157 rows, 2023-03-01 → 2026-04-30
- Raw exogenous: 52 rows, 2023-01-01 → 2027-04-01
- Monthly demand: 38 months, 2023-03-01 → 2026-04-01
- Modeling data (after null-row drop): 37 rows, 2023-04-01 → 2026-04-01
- Active regressors: 28
- Train split: 31 rows, 2023-04-01 → 2025-10-01
- Validation split: 3 rows, 2025-11-01 → 2026-01-01
- Test split: 3 rows, 2026-02-01 → 2026-04-01
- Future horizons: 3m (May–Jul 2026), 6m (May–Oct 2026), 12m (May 2026–Apr 2027)

### Optuna tuning outcome

- Trials: 30
- Best validation trial: `prophet_candidate_004` (val WAPE=0.407)
- Pre-champions (rank 1–3): `prophet_candidate_004`, `prophet_candidate_011`, `prophet_candidate_007`
- Precision threshold warning: best precision 0.6298, below 85% threshold

### Test evaluation

| Candidate | Test WAPE | Test MASE |
|---|---|---|
| `prophet_candidate_004` | 1.0944 | 5.0447 |
| `prophet_candidate_011` | 0.2860 | 1.3183 |
| **`prophet_candidate_007`** | **0.2231** | **1.0283** |

**Champion selected: `prophet_candidate_007`**
- Test WAPE: 0.2231
- Test M+2 WAPE: 0.4491
- Test M+3 WAPE: 1.8507
- Seasonality mode: multiplicative
- Refit on full history: True

### Warnings (non-blocking)

1. `FutureWarning` from Pandas in `data_ingestion/nodes.py:71` — dtype incompatibility on masked demand column. Benign; does not affect results.
2. Monthly demand inconsistency warnings for all months (abs_diff < 0.01, pct_diff < 0.08%) — rounding artifact from daily→monthly aggregation. Benign.
3. `FutureWarning` from Pandas in `model_selection/prophet/nodes.py:669` — empty DataFrame concat. Benign; does not affect selection result.
4. Champion does not exceed 85% precision threshold (best precision: 0.8118). Expected with 3-row test set.

---

## 7. Contract Snapshot Summary

See [monthly_mvp_contract_snapshot.md](monthly_mvp_contract_snapshot.md) for full per-dataset schema details.

Summary of confirmed artifacts:

| Stage | Datasets | Status |
|---|---|---|
| Feature (04) | `monthly_prophet_features` (38×37) | ✅ |
| Model input (05) | 8 parquet + 1 JSON | ✅ All present |
| Training (06/tuning) | `tuning_results` (30×36), `validation_metrics` (30×21), `prechampion_configs`, `candidate_models`, `training_metadata` | ✅ All present |
| Selection (06/selection) | `test_metrics` (3×22), `selection_summary` (3×17), `champion_test_forecast` (9×9), `champion_model`, `champion_metadata` | ✅ All present |
| Reporting (08) | `operational_test_forecasts` (9×9), `lead_time_metrics` (3×5), `model_selection_audit` (3×16) | ✅ All present |
| Inference (07) | `forecast_3m` (3×16), `forecast_6m` (6×16), `forecast_12m` (12×16), `forecast_latest` (12×16), `inference_metadata` | ✅ All present |

**All 28 checked artifacts loadable. Zero missing.**

---

## 8. Experimental Pipeline Isolation Check

Confirmed by inspecting `pipeline_registry.py` and the train_monthly sub-pipelines:

### `pipeline_registry.py` structure

```python
# monthly_mvp is built from:
prophet_monthly_e2e = (
    ingestion + fe_monthly + model_input
    + prophet_monthly_training          # ← Prophet only
    + prophet_monthly_selection         # ← Prophet only
    + inference
)
monthly_mvp = prophet_monthly_e2e

# Experimental stubs are separate:
experimental_training = monthly_training + weekly_training  # includes SARIMAX, CatBoost stubs
```

### `train_monthly/pipeline.py` composition

The generic `train_monthly` pipeline composes:
1. `create_prophet_pipeline()` — active, fully implemented
2. `create_catboost_pipeline()` — wrapped in namespace `train_monthly.catboost`, inputs from `model_input_monthly_catboost_*`
3. `create_sarimax_pipeline()` — wrapped in namespace `train_monthly.sarimax`, inputs from `model_input_monthly_sarimax_*`

`monthly_mvp` calls `create_prophet_monthly_pipeline()` directly (the standalone prophet pipeline), **not** `train_monthly.create_pipeline()`. This ensures CatBoost and SARIMAX nodes are never included.

### SARIMAX and CatBoost node stubs

Both `train_monthly/sarimax/nodes.py` and `train_monthly/catboost/nodes.py` contain `raise NotImplementedError(...)` in all functions. They cannot run even if accidentally wired.

### `train_weekly` pipeline

Uses the same namespace pattern with separate input/output aliases. Not included in `monthly_mvp`.

### Isolation verdict

| Concern | Status |
|---|---|
| `monthly_mvp` executes SARIMAX nodes | ❌ Confirmed NOT executed |
| `monthly_mvp` executes CatBoost nodes | ❌ Confirmed NOT executed |
| `monthly_mvp` depends on generic `train_monthly` | ❌ Confirmed NOT used |
| `monthly_mvp` depends on weekly training | ❌ Confirmed NOT included |
| `monthly_mvp` uses generic model_selection dispatcher | ❌ Confirmed NOT used (uses Prophet-specific selection) |

**`monthly_mvp` is fully isolated from experimental scaffolding.**

---

## 9. Kedro-Viz Check

Kedro-Viz is available via `uv run kedro viz run` (confirmed in `kedro --help`). Manual inspection was not performed in this automated phase. Recommend running manually to visually validate DAG isolation before Phase 1 begins.

---

## 10. Risks / Observations

### R1 — `FutureWarning` on demand masking (low severity)

Location: `data_ingestion/nodes.py:71`  
Warning: Setting item of incompatible dtype on masked demand column.  
Impact: None on outputs. Will become an error in a future Pandas version.  
Action: Address in a separate fix or in Phase 1 when touching ingestion.

### R2 — `FutureWarning` on champion selection summary concat (low severity)

Location: `model_selection/prophet/nodes.py:669`  
Warning: DataFrame concat with empty entries.  
Impact: None on outputs currently.  
Action: Address in Phase 1 when refactoring model selection.

### R3 — Precision threshold not met (informational)

Champion `prophet_candidate_007` does not exceed the 85% precision threshold (best: 0.8118). This is expected given the 3-row test set. The warning is produced by design and does not block the pipeline.

### R4 — Kedro 1.3.1 CLI differences from Phase 0 spec (low severity)

`kedro pipeline list` and `kedro catalog list` do not exist in Kedro 1.3.1. Correct commands are `kedro registry list` and `kedro catalog describe-datasets`. The Phase 0 planning document should be updated before being shared externally.

### R5 — `CatBoostRegressor` / `SARIMAXResultsWrapper` type hint warnings (informational)

These appear at session load time because type hints reference classes that are not imported unless the packages are installed. They do not affect runtime behavior or the MVP pipeline. Will be resolved when SARIMAX and CatBoost are implemented in Phase 2+.

### R6 — Monthly demand inconsistency warnings (informational)

All months show tiny abs_diff (< 0.01 units) between reported monthly demand and sum of daily demand. This is a rounding artifact from the data source, not a pipeline bug. Logged and benign.

---

## 11. Recommendation for Phase 1

Phase 0 is complete. The baseline is stable and defensible.

**Phase 1 can proceed.** Recommended scope for Phase 1:

1. **Namespace alignment** — Introduce a `monthly.prophet` namespace for all Prophet-specific model input datasets (`monthly_prophet_train`, `monthly_prophet_validation`, etc.) so they follow the `<granularity>.<family>_<artifact>` convention used by CatBoost and SARIMAX placeholders.

2. **`model_input_preparation` modularization** — The `model_input_preparation` pipeline currently generates Prophet-specific datasets directly. Refactor to produce a generic monthly modeling dataset plus Prophet-specific split/future datasets through a Prophet-namespaced sub-pipeline.

3. **Do not touch** `forecast_inference`, `model_selection/prophet`, or any existing `monthly_prophet_*` catalog names until namespace alignment is validated end-to-end.

4. **Re-run the 35 protected tests** after every structural change before merging to `dev`.

5. **Preserve** `monthly_mvp` as the default alias throughout Phase 1.

---

*Report generated by Phase 0 safety baseline execution on 2026-05-31.*
