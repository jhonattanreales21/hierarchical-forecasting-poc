# Family-Champion Explainability (SHAP + native drivers)

## Purpose

Each monthly **family champion** (Prophet, SARIMAX, CatBoost) is accompanied by a
**driver-importance** artifact that answers "what drives this champion's demand
prediction?". The artifact is surfaced in the Streamlit **Evaluation Report** page as a
global importance bar and is part of the curated outputs the GenAI/RAG layer may cite.
It is an *interpretive support* artifact — it never overrides model selection, metrics,
or champion metadata.

## Where it runs

Inside the monthly model-selection pipeline
(`model_selection/monthly/pipeline.py`), the node
`generate_monthly_family_champion_explanations` runs after
`select_monthly_family_champions`, as a sibling of `build_monthly_champion_artifacts`.
It therefore flows automatically into `monthly_model_selection` and
`monthly_forecast_e2e` — no registry change. The node is **additive and side-effect
free**: a failure for any one family is logged and skipped, never breaking selection or
inference.

Toggle/configure via `model_selection.monthly.explainability` in
`conf/base/parameters/model_selection.yml`.

## Method per family (academically defensible)

Each family champion is explained on its **full-history refit** — the same Champion
Protocol stage-5 refit used to produce the production model — so the importance reflects
the production-representative model rather than a train-only candidate.

| Family | Method | `importance_type` | Importance meaning |
|--------|--------|-------------------|--------------------|
| CatBoost | `shap.TreeExplainer` (exact for trees, no background set) | `mean_abs_shap` | `mean(\|SHAP\|)` per feature |
| Prophet | `model.predict` component frame | `mean_abs_contribution` | mean absolute *centered* contribution of each regressor and trend/seasonality component (centering stops the trend's baseline level from dominating) |
| SARIMAX | fitted coefficients (`params`/`bse`/`pvalues`) | `abs_coefficient` | `\|coefficient\|` per exogenous term; AR/MA/trend/σ² excluded |

The values are **not comparable across families** (different statistics and scales). The
unified table tags every row with `importance_type`, and the app labels the axis and a
caption accordingly. SHAP applies cleanly only to the tree model; Prophet and SARIMAX
use their proper native explanations rather than a forced/approximate SHAP.

## Artifacts (catalog)

| Dataset | Type | Path | Role |
|---------|------|------|------|
| `monthly_family_champion_importance` | ParquetDataset | `data/08_reporting/monthly_family_champion_importance.parquet` | App-facing long-form importance table |
| `monthly_catboost_shap_explainer` | PickleDataset | `data/06_models/explainability/monthly_catboost_shap_explainer.pkl` | Fitted SHAP `TreeExplainer` (reuse without recompute) |
| `monthly_catboost_shap_values` | ParquetDataset | `data/06_models/explainability/monthly_catboost_shap_values.parquet` | Per-observation SHAP values (date + one column per feature) |
| `monthly_family_champion_explainability_metadata` | JSONDataset | `data/06_models/explainability/monthly_family_champion_explainability_metadata.json` | Per-family method, champion id, base value, provenance |

The importance table stores **all** features; `explainability.top_n_features` is only a
display hint consumed by the app. The SHAP explainer + per-observation values are
persisted so future local explanations (beeswarm, per-month waterfall) can be added
without recomputation.

## Importance table schema

`family, champion_id, feature, importance, importance_type, rank, coefficient,
std_err, pvalue, computed_at`

`rank` is 1-based within each family (1 = most important). `coefficient`, `std_err`, and
`pvalue` are populated for SARIMAX and `NaN` elsewhere.

## App display

`app/pages/05_Evaluation_Report.py` renders the section via
`render_champion_explainability` (`app/ui/page_blocks/evaluation_blocks.py`), using the
reusable `shared.viz.plot_feature_importance_bar`. A family selector defaults to the
production-champion family; the chart's axis label and caption reflect the selected
family's `importance_type`.
