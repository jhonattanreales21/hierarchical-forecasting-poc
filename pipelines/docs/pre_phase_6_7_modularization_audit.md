# Current Modularization Audit and Next-Step Roadmap

Audit date: 2026-05-31  
Repository: `hierarchical-demand-forecasting-poc`  
Scope: refreshed audit after recent changes in monthly `model_selection` and
`forecast_inference`. This audit did not implement new forecasting functionality.

Note: before this audit, the working tree already contained uncommitted source
changes in `pipeline_registry.py`, `forecast_inference`, `model_input_preparation`,
and tests. This report audits that current working tree state.

## 1. Executive Summary

| Area | Status | Summary |
|---|---:|---|
| Overall modularization | PARTIALLY READY | The monthly layer now has a generic model-input foundation, Prophet compatibility adapter, SARIMAX adapter/training, Prophet-vs-SARIMAX production champion selection, and metadata-driven generic forecast inference. |
| `monthly_mvp` protection | READY | `uv run kedro run --pipeline monthly_mvp` completed successfully in 397.7 sec. The route remains Prophet-only for training/legacy selection and does not touch CatBoost stubs. |
| Generic monthly route | READY WITH APP GAP | `uv run kedro run --pipeline monthly_forecast_e2e` completed successfully and selected SARIMAX as production champion, then wrote generic `monthly_forecast_*` outputs. |
| Forecast inference | PARTIALLY READY | Metadata-driven dispatch works for Prophet and SARIMAX. It still consumes Prophet-named future frames internally, labels them as generic sources in output metadata, and does not write legacy Prophet forecast aliases. |
| Streamlit compatibility | NOT READY | The app still reads `monthly_prophet_forecast_*`, `monthly_prophet_inference_metadata.json`, and Prophet-specific champion paths. The successful generic run writes `monthly_forecast_*` instead. |
| CatBoost readiness | NOT READY | CatBoost is installed, but monthly CatBoost nodes still raise `NotImplementedError`; no generic CatBoost adapter/artifact contract exists yet. |
| Highest-risk blocker | HIGH | The pipeline is ahead of the app: generic monthly forecasts are produced, but the Streamlit forecast page is still Prophet-path and Prophet-copy dependent. |
| Recommended next step | READY TO MIGRATE | Stabilize the generic champion/output contract and migrate or alias Streamlit forecast consumption before adding CatBoost. |

## 2. Commands Executed and Results

| Command | Result | Relevant summary | Related to recent changes? |
|---|---:|---|---:|
| `cd pipelines && uv run kedro pipeline list` | FAIL | Kedro CLI returned `No such command 'list'`. | No; CLI mismatch. |
| `cd pipelines && uv run kedro catalog list` | FAIL | Kedro CLI returned `No such command 'list'`. | No; CLI mismatch. |
| Python `register_pipelines()` inspection | PASS | Printed 18 registered pipeline keys and node lists. | Yes. |
| Python catalog inspection | PASS | Catalog contains generic monthly selection and generic forecast output datasets. | Yes. |
| `uv run --package hdf_pipelines pytest pipelines/tests/test_model_selection_monthly_nodes.py pipelines/tests/test_forecast_inference_nodes.py` | PASS | 22 passed. | Yes. |
| `uv run --package hdf_pipelines pytest pipelines/tests/test_model_input_preparation_nodes.py pipelines/tests/test_train_monthly_prophet_nodes.py pipelines/tests/test_train_monthly_sarimax_nodes.py` | PASS | 40 passed. | Yes. |
| `uv run --package hdf_pipelines pytest pipelines/tests/test_run.py pipelines/tests/test_metrics.py` | PASS | 12 passed. There is no current `test_pipeline_registry.py`; `test_run.py` is the registry smoke test. | Partial. |
| `uv run --package hdf_pipelines pytest pipelines/tests/ --cov` | FAIL | 89 passed, 1 failed, 80% total coverage. Failure is `test_load_and_clean_exogenous_strips_trailing_whitespace_from_column_names`, whose fixture omits required `expected_market_share`. | Indirect; blocks DoD but not caused by model selection/inference. |
| `cd pipelines && uv run kedro run --pipeline monthly_mvp` | PASS | 22/22 tasks completed in 397.7 sec. Prophet champion: `prophet_candidate_007`, held-out test WAPE 0.2231. | Yes; confirms MVP protection. |
| `cd pipelines && uv run kedro run --pipeline monthly_forecast_e2e` | PASS | 24/24 tasks completed. SARIMAX family champion and production champion: `sarimax_trial_010`, held-out test WAPE 0.2635. Generic monthly forecasts were written. | Yes; strongest route check. |
| `uv run python -c "import catboost"` | PASS | `catboost_available=True 1.2.10`. | Yes, for CatBoost readiness. |

Important warning from Kedro route runs: Kedro emits type-extraction warnings for
stub `train_best_candidate` return annotations such as `CatBoostRegressor`,
`Prophet`, and `SARIMAXResultsWrapper`. These do not fail the protected routes,
but they are noise to clean when those stubs are implemented or isolated.

## 3. Pipeline Registry and Routing

Evidence: `pipelines/src/hdf_pipelines/pipeline_registry.py`.

Registered keys:

`__default__`, `monthly_forecast_e2e`, `monthly_model_selection`,
`prophet_sarimax_comparison`, `monthly_mvp`, `prophet_monthly_e2e`,
`data_ingestion`, `feature_engineering_monthly`, `model_input_preparation`,
`train_monthly`, `model_selection`, `forecast_inference`,
`feature_engineering_weekly`, `train_weekly`, `reconciliation`,
`experimental_training`, `experimental_inference`,
`experimental_full_experiment`.

Findings:

- `__default__` points to `monthly_forecast_e2e`.
- `monthly_forecast_e2e` is now the canonical monthly route:
  ingestion -> monthly features -> generic splits -> Prophet training -> SARIMAX
  training -> monthly multi-family selection -> metadata-driven inference.
- `monthly_mvp` points to `prophet_monthly_e2e` and remains Prophet-only for
  training and legacy Prophet champion selection.
- `monthly_mvp` still runs the SARIMAX input adapter because that adapter is part
  of shared model-input preparation. It does not run SARIMAX training.
- CatBoost stubs are reachable from `train_monthly`, `experimental_training`, and
  `experimental_full_experiment`, but not from `monthly_mvp`, `monthly_forecast_e2e`,
  or `prophet_sarimax_comparison`.
- The legacy generic `model_selection` pipeline still contains `NotImplementedError`
  nodes and should stay out of recommended routes.

| Pipeline | Status | Includes Prophet | Includes SARIMAX | Includes CatBoost | Safe for MVP? | Notes |
|---|---:|---:|---:|---:|---:|---|
| `__default__` | READY WITH APP GAP | Yes | Yes | No | N/A | Same as `monthly_forecast_e2e`; passed and wrote generic forecasts. |
| `monthly_forecast_e2e` | READY WITH APP GAP | Yes | Yes | No | N/A | Passed; production champion was SARIMAX. |
| `prophet_sarimax_comparison` | READY | Yes | Yes | No | N/A | Same as default without inference. |
| `monthly_model_selection` | READY | Yes | Yes | No | N/A | Four-node selection/artifact builder. |
| `monthly_mvp` | READY | Yes | Adapter only | No | Yes | Passed; produces legacy Prophet artifacts. |
| `prophet_monthly_e2e` | READY | Yes | Adapter only | No | Yes | Alias target for `monthly_mvp`. |
| `train_monthly` | NOT READY | Yes | Yes | Yes | No | Includes CatBoost stubs. |
| `model_selection` | NOT READY | Mixed | Mixed | Mixed | No | Legacy generic selection stubs. |
| `forecast_inference` | READY WITH INPUT-NAME GAP | Yes | Yes | No | N/A | Works from generic champion artifacts; input frames are still Prophet-named. |
| `experimental_*` | NOT READY | Mixed | Mixed | Mixed | No | Stub-heavy and not part of default. |

## 4. Catalog and Artifact Layer

Evidence: `pipelines/conf/base/catalog.yml`.

Generic monthly foundation is catalog-backed:

| Dataset | Exists | Path | Notes |
|---|---:|---|---|
| `monthly_modeling_data` | Yes | `data/05_model_input/monthly_modeling_data.parquet` | Generic monthly modeling table. |
| `monthly_train` | Yes | `data/05_model_input/monthly_train.parquet` | Generic train split. |
| `monthly_validation` | Yes | `data/05_model_input/monthly_validation.parquet` | Generic validation split. |
| `monthly_test` | Yes | `data/05_model_input/monthly_test.parquet` | Generic held-out test split. |
| `monthly_full_train` | Yes | `data/05_model_input/monthly_full_train.parquet` | Full history after split construction. |
| `monthly_split_metadata` | Yes | `data/05_model_input/monthly_split_metadata.json` | Generic split metadata. |

Prophet compatibility artifacts are preserved:

| Dataset | Exists | Path | Notes |
|---|---:|---|---|
| `monthly_prophet_modeling_data` | Yes | `data/05_model_input/monthly_prophet_modeling_data.parquet` | Produced from generic data by adapter. |
| `monthly_prophet_train` / `validation` / `test` / `full_train` | Yes | `data/05_model_input/monthly_prophet_*.parquet` | Legacy-compatible `ds`/`y` shape. |
| `monthly_prophet_future_3m` / `6m` / `12m` | Yes | `data/05_model_input/monthly_prophet_future_*.parquet` | Still the actual future-frame inputs to generic inference. |
| `monthly_prophet_split_metadata` | Yes | `data/05_model_input/monthly_prophet_split_metadata.json` | Preserved for Prophet training/selection. |

SARIMAX artifacts are catalog-backed:

| Dataset | Exists | Path | Notes |
|---|---:|---|---|
| `monthly_sarimax_train` / `validation` / `test` / `full_train` | Yes | `data/05_model_input/monthly_sarimax_*.parquet` | Produced by SARIMAX adapter. |
| `monthly_sarimax_split_metadata` | Yes | `data/05_model_input/monthly_sarimax_split_metadata.json` | Includes frequency/exogenous contract. |
| `monthly_sarimax_tuning_results` | Yes | `data/06_models/tuning/monthly_sarimax_tuning_results.parquet` | Produced by training. |
| `monthly_sarimax_validation_metrics` | Yes | `data/06_models/tuning/monthly_sarimax_validation_metrics.parquet` | Produced by training. |
| `monthly_sarimax_prechampion_configs` | Yes | `data/06_models/tuning/monthly_sarimax_prechampion_configs.json` | Consumed by monthly selection. |
| `monthly_sarimax_candidate_models` | Yes | `data/06_models/tuning/monthly_sarimax_candidate_models.pkl` | Consumed by monthly selection. |
| `monthly_sarimax_training_metadata` | Yes | `data/06_models/tuning/monthly_sarimax_training_metadata.json` | Consumed by monthly selection and artifact builder. |

Generic monthly selection and forecast artifacts now exist and were generated by
`monthly_forecast_e2e`:

| Dataset | Exists | Latest observed shape | Path |
|---|---:|---:|---|
| `monthly_candidate_test_metrics` | Yes | 6 x 16 | `data/06_models/selection/monthly_candidate_test_metrics.parquet` |
| `monthly_family_champion_summary` | Yes | 2 x 11 | `data/06_models/selection/monthly_family_champion_summary.parquet` |
| `monthly_model_selection_summary` | Yes | 1 x 11 | `data/06_models/selection/monthly_model_selection_summary.parquet` |
| `champion_monthly_model` | Yes | Pickle | `data/06_models/champions/monthly_champion.pkl` |
| `champion_monthly_metadata` | Yes | JSON | `data/06_models/champions/champion_monthly_metadata.json` |
| `monthly_forecast_3m` | Yes | 3 x 17 | `data/07_model_output/monthly_forecast_3m.parquet` |
| `monthly_forecast_6m` | Yes | 6 x 17 | `data/07_model_output/monthly_forecast_6m.parquet` |
| `monthly_forecast_12m` | Yes | 12 x 17 | `data/07_model_output/monthly_forecast_12m.parquet` |
| `monthly_forecast_latest` | Yes | 12 x 17 | `data/07_model_output/monthly_forecast_latest.parquet` |
| `monthly_inference_metadata` | Yes | JSON | `data/07_model_output/monthly_inference_metadata.json` |

Missing or inconsistent names:

- `monthly_production_champion_summary` is still absent. The implemented dataset is
  `monthly_model_selection_summary`.
- Generic future-frame catalog entries such as `monthly_future_3m`,
  `monthly_future_6m`, and `monthly_future_12m` are absent. The inference node
  consumes `monthly_prophet_future_*` but labels output source datasets as
  `monthly_future_*`, which is misleading.
- Legacy `monthly_prophet_forecast_*` and
  `monthly_prophet_inference_metadata.json` catalog entries are absent, while the
  Streamlit app still expects those files.

## 5. Monthly Model Selection Audit

Evidence: `pipelines/src/hdf_pipelines/pipelines/model_selection/monthly/`.

Current behavior:

- Compares only Prophet and SARIMAX through `active_families: [prophet, sarimax]`.
- Excludes weekly candidates.
- Excludes CatBoost for now.
- Does not assume six candidates; the latest run scored 3 Prophet + 3 SARIMAX
  candidates for 6 total rows.
- Refits each candidate on train+validation for held-out test scoring.
- Selects one family champion per family.
- Selects one production champion across family champions.
- Builds `champion_monthly_model` and `champion_monthly_metadata`.
- `refit_champion.enabled: true` refits the elected production champion on full
  history before inference.

Latest observed generic selection output:

| Family | Family champion | Test WAPE | Test MASE | Test RMSE | Notes |
|---|---|---:|---:|---:|---|
| Prophet | `prophet_candidate_011` | 0.4712 | 2.1718 | 4.1748 | Best of 3 Prophet prechampions in generic selection. |
| SARIMAX | `sarimax_trial_010` | 0.2635 | 1.2144 | 2.5899 | Best of 3 SARIMAX prechampions. |
| Production | `sarimax_trial_010` | 0.2635 | 1.2144 | 2.5899 | Selected across family champions by WAPE. |

Important distinction: the legacy Prophet-only MVP selected
`prophet_candidate_007` with WAPE 0.2231 in the Prophet-specific selection path.
The generic selector refits and scores Prophet candidates differently and selected
`prophet_candidate_011` as the Prophet family champion. This is not automatically
a bug, but it should be documented because stakeholders may notice that the
Prophet-only champion differs from the generic-route Prophet champion.

## 6. Forecast Inference Audit

Evidence: `pipelines/src/hdf_pipelines/pipelines/forecast_inference/`.

Current node:

- `generate_monthly_champion_forecasts`

Inputs:

- `champion_monthly_model`
- `champion_monthly_metadata`
- `monthly_prophet_future_3m`
- `monthly_prophet_future_6m`
- `monthly_prophet_future_12m`
- `params:forecast_inference.monthly`

Outputs:

- `monthly_forecast_3m`
- `monthly_forecast_6m`
- `monthly_forecast_12m`
- `monthly_forecast_latest`
- `monthly_inference_metadata`

Current status:

| Component | Status | Evidence / risk |
|---|---:|---|
| Metadata-driven dispatch | PASS | Dispatch uses `champion_monthly_metadata["model_family"]`. Latest route dispatched to SARIMAX. |
| Prophet adapter | PASS | Unit-tested, preserves `ds`/regressor prediction behavior. |
| SARIMAX adapter | PASS | Unit-tested and end-to-end tested; latest route produced intervals via `get_forecast().conf_int`. |
| CatBoost adapter | NOT IMPLEMENTED | Explicitly rejected as unsupported. |
| Standard output schema | PASS | Forecast tables use 17-column canonical schema with `date`, `forecast`, intervals, family, horizon, champion, run, and source columns. |
| Generic forecast outputs | PASS | Latest route wrote `monthly_forecast_3m/6m/12m/latest`. |
| Generic future inputs | PARTIAL | Actual pipeline inputs are Prophet future frames. Output metadata labels sources as `monthly_future_*`, though those are not catalog datasets. |
| Legacy Prophet forecast aliases | FAIL | Not written; app expects `monthly_prophet_forecast_*`. |

Latest observed generic forecast output:

- Champion family: `sarimax`
- Champion ID: `sarimax_trial_010`
- Forecast windows:
  - 3m: 2026-05-01 to 2026-07-01
  - 6m: 2026-05-01 to 2026-10-01
  - 12m/latest: 2026-05-01 to 2027-04-01
- Prediction intervals: present, method `sarimax_get_forecast_conf_int`

## 7. Champion Metadata Contract

Latest generated `champion_monthly_metadata.json` keys:

`active_regressors`, `champion_id`, `champion_level`, `compatibility`,
`family_champions`, `granularity`, `inference_contract`, `metrics`,
`model_artifact`, `model_family`, `refit`, `selection`, `test_period`.

Useful fields now present:

- `model_family`: `sarimax`
- `champion_id`: `sarimax_trial_010`
- `granularity`: `monthly`
- `metrics`: WAPE, MASE, RMSE, bias from held-out test scoring.
- `selection.primary_metric`: `wape`
- `selection.selected_at`
- `test_period.start_date`, `end_date`, `n_rows`
- `refit.performed`, `refit.n_obs`, `refit.start_date`, `refit.end_date`
- `model_artifact.catalog_key`: `champion_monthly_model`
- `inference_contract.date_column`, `target_column`, `sku_column`,
  `active_regressors`, `forecast_horizons`, interval info, and SARIMAX order.

Remaining contract gaps:

| Field / contract | Status | Recommendation |
|---|---:|---|
| `selection_metric` / `selection_metric_value` top-level | Partial | Inference can derive these from nested fields, but top-level fields would simplify consumers. |
| `training_data_cutoff` top-level | Missing | Add alias from `refit.end_date`. |
| `model_artifact_reference` top-level | Partial | Current nested `model_artifact.catalog_key` is good; document it or alias it. |
| `selected_hyperparameters` top-level | Missing | Add family-specific selected config for auditability. |
| Prophet interval width | Partial | Available from fitted Prophet model, not surfaced in generic metadata. |
| SARIMAX result type | Missing | Add statsmodels result class/type for debugging serialized artifacts. |
| Future regressor requirements | Partial | `active_regressors` exists, but future-frame source and required columns should be explicit. |

Conclusion: the metadata is good enough for the current working Prophet/SARIMAX
inference path, but it should be formalized before app/API consumers and CatBoost
are layered on top.

## 8. App and API Consumption

Evidence: `app/utils/paths.py`, `app/utils/data_loaders.py`,
`app/pages/02_Monthly_Forecast.py`, `app/ui/page_blocks/monthly_blocks.py`,
`api/src/hdf_api/routers/forecast.py`.

Streamlit is still Prophet-specific:

| Consumer | Current expected path | Current generic output | Status |
|---|---|---|---:|
| Champion metadata | `monthly_prophet_champion_metadata.json` | `champion_monthly_metadata.json` | Not migrated |
| Inference metadata | `monthly_prophet_inference_metadata.json` | `monthly_inference_metadata.json` | Not migrated |
| Latest forecast | `monthly_prophet_forecast_latest.parquet` | `monthly_forecast_latest.parquet` | Not migrated |
| Horizon forecasts | `monthly_prophet_forecast_{h}m.parquet` | `monthly_forecast_{h}m.parquet` | Not migrated |
| Forecast schema | expects `ds` and Prophet-style columns | generic `date`, `forecast`, intervals | Not migrated |
| Page copy | says Prophet champion/MVP | generic champion may be SARIMAX | Not migrated |

The API forecast router remains 501-only and does not yet read either legacy or
generic artifacts. It can be introduced against the generic output schema later.

## 9. CatBoost Readiness

Evidence: `pipelines/src/hdf_pipelines/pipelines/train_monthly/catboost/`,
`pipelines/conf/base/catalog.yml`, `pipelines/conf/base/parameters/train_monthly.yml`.

| Item | Status | Notes |
|---|---:|---|
| Dependency | READY | `catboost 1.2.10` imports successfully. |
| Monthly CatBoost adapter | NOT READY | No `monthly_catboost_*` adapter from generic splits. |
| Training nodes | NOT READY | `tune_hyperparameters` and `train_best_candidate` raise `NotImplementedError`. |
| Catalog placeholders | PARTIAL | `model_input_monthly_catboost_*` and `candidate_monthly_catboost` exist, but not a full artifact parity contract. |
| Model selection integration | NOT READY | Generic monthly selector currently accepts Prophet/SARIMAX only. |
| Inference integration | NOT READY | Forecast inference explicitly rejects `model_family == "catboost"`. |
| Tests | NOT READY | No CatBoost adapter/training/selection/inference tests. |

CatBoost should not be added to `active_families` or default routes until it has
the same staged contract as Prophet/SARIMAX.

## 10. Risk Register

| Risk | Severity | Evidence | Mitigation |
|---|---:|---|---|
| Streamlit reads legacy Prophet outputs while default writes generic outputs | HIGH | `app/utils/paths.py` points to `monthly_prophet_forecast_*`; default writes `monthly_forecast_*`. | Migrate loaders/UI to generic schema or intentionally write compatibility aliases. |
| Output metadata labels source frames as generic although catalog inputs are Prophet-named | MEDIUM | Pipeline inputs are `monthly_prophet_future_*`; latest output source is `monthly_future_*`. | Add real generic future datasets or label source as `monthly_prophet_future_*` until migration. |
| Generic Prophet family champion differs from legacy Prophet MVP champion | MEDIUM | MVP chose `prophet_candidate_007`; generic selection chose `prophet_candidate_011`. | Document scoring/refit differences; add a regression test if equivalence is intended. |
| Full test suite has one failing data-ingestion fixture | MEDIUM | Missing `expected_market_share` in test fixture. | Update test fixture or make the new exogenous contract explicit in docs/tests. |
| CatBoost stubs remain reachable from `train_monthly` | HIGH | `train_monthly` includes CatBoost nodes that raise `NotImplementedError`. | Keep out of default; implement adapter/training before exposing route. |
| Legacy generic `model_selection` stubs remain registered | MEDIUM | `model_selection/nodes.py` has three `NotImplementedError` nodes. | Keep isolated or replace with monthly selector when scope is clear. |
| Long route runtime | MEDIUM | `monthly_mvp` took 397.7 sec; `monthly_forecast_e2e` reruns Prophet and SARIMAX fitting. | Add reduced-trial smoke parameters for CI/audit use. |

## 11. Recommended New Roadmap

This roadmap intentionally does not reuse the previous phase numbering.

### Milestone A — Contract Lock

Goal: make the generic monthly outputs the documented project contract.

- Freeze `monthly_forecast_3m`, `monthly_forecast_6m`,
  `monthly_forecast_12m`, `monthly_forecast_latest`, and
  `monthly_inference_metadata` as canonical forecast outputs.
- Decide whether `monthly_model_selection_summary` is final, or add
  `monthly_production_champion_summary` as an intentional alias.
- Add a short schema doc for `champion_monthly_metadata` and
  `monthly_forecast_*`.
- Add top-level metadata aliases for selection metric/value, training cutoff,
  selected hyperparameters, and model artifact reference.

### Milestone B — Generic Future Frames

Goal: remove the Prophet naming leak from generic inference.

- Add catalog datasets `monthly_future_3m`, `monthly_future_6m`,
  `monthly_future_12m`.
- Produce them from the generic monthly model-input layer.
- Keep Prophet future frames as compatibility outputs if the legacy route still
  needs them.
- Make `monthly_inference_metadata.source_future_frames` match the real catalog
  inputs.

### Milestone C — Application Migration

Goal: make Streamlit display the current monthly production champion, not only
the Prophet MVP baseline.

- Update app loaders to prefer generic artifacts and optionally fall back to
  Prophet legacy artifacts.
- Map generic forecast schema (`date`, `forecast`, `forecast_lower`,
  `forecast_upper`) to the monthly page UI.
- Replace Prophet-specific copy with current-champion copy driven by
  `model_family`.
- Keep a separate Prophet MVP/reference section only if useful for academic
  comparison.

### Milestone D — Route and Test Stabilization

Goal: make the new default route cheap and reliable enough for repeated checks.

- Add a reduced-trial smoke configuration or test fixture for
  `monthly_forecast_e2e`.
- Add an integration test that verifies `champion_monthly_metadata` and
  `monthly_forecast_latest` exist after the generic route.
- Fix the `expected_market_share` exogenous fixture failure.
- Clean or isolate Kedro type-extraction warnings from stub return annotations.

### Milestone E — CatBoost Adapter First

Goal: introduce CatBoost data contracts without touching production selection yet.

- Build `monthly_catboost_train`, `monthly_catboost_validation`,
  `monthly_catboost_test`, `monthly_catboost_full_train`, and
  `monthly_catboost_split_metadata` from generic monthly splits.
- Use numeric active regressors from the generic monthly configuration.
- Treat `sku` carefully: for the single-SKU PoC, drop it or document explicit
  categorical handling.
- Add adapter tests for schema, missing features, non-numeric values, and nulls.

### Milestone F — CatBoost Training Parity

Goal: make CatBoost produce Prophet/SARIMAX-equivalent training artifacts.

- Implement a small deterministic grid before Optuna.
- Emit `monthly_catboost_tuning_results`,
  `monthly_catboost_validation_metrics`,
  `monthly_catboost_prechampion_configs`,
  `monthly_catboost_candidate_models`, and
  `monthly_catboost_training_metadata`.
- Keep CatBoost out of `active_families` until these artifacts pass tests.

### Milestone G — CatBoost Selection and Inference

Goal: allow CatBoost to compete only after its contract is stable.

- Extend monthly model selection to include optional CatBoost artifacts.
- Add CatBoost to `active_families` only after the selector can handle missing
  optional families safely.
- Add a CatBoost forecast adapter only when CatBoost can become the production
  champion.
- Add tests for three-family selection and CatBoost inference.

## 12. Immediate Recommendation

Do Milestones A, B, and C before CatBoost. The current default route is now
methodologically stronger than the app can consume: it can elect SARIMAX and
produce generic forecasts, but Streamlit still asks for Prophet-named forecast
files. Closing that gap will make the PoC coherent for stakeholders and will make
CatBoost integration much safer.
