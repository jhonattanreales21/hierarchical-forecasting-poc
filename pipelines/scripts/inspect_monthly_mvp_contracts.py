"""Lightweight contract inspector for the monthly forecasting artifacts.

Loads catalog datasets, prints schema summaries, and writes a Markdown report
to pipelines/docs/monthly_mvp_contract_snapshot.md.

Active artifact sources (as of the rolling-origin champion protocol):
  - monthly_candidate_metrics          (replaces monthly_candidate_test_metrics)
  - monthly_family_champion_summary    (replaces monthly_prophet_test_metrics / per-family tables)
  - monthly_model_selection_summary    (replaces monthly_prophet_model_selection_summary)
  - champion_monthly_metadata.json     (replaces monthly_prophet_champion_metadata)

Legacy files that may still exist on disk from old runs (do NOT use):
  - data/06_models/selection/monthly_candidate_test_metrics.parquet
  - data/06_models/selection/monthly_prophet_test_metrics.parquet
  - data/06_models/selection/monthly_prophet_champion_test_forecast.parquet

Usage (from pipelines/ directory):
    uv run python scripts/inspect_monthly_mvp_contracts.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "monthly_mvp_contract_snapshot.md"

TABULAR_DATASETS = [
    # 04_feature
    ("monthly_prophet_features", "feature"),
    # 05_model_input
    ("monthly_prophet_modeling_data", "model_input"),
    ("monthly_prophet_train", "model_input"),
    ("monthly_prophet_validation", "model_input"),
    ("monthly_prophet_test", "model_input"),
    ("monthly_prophet_full_train", "model_input"),
    ("monthly_prophet_future_3m", "model_input"),
    ("monthly_prophet_future_6m", "model_input"),
    ("monthly_prophet_future_12m", "model_input"),
    # 06_models — tuning (per-family rolling-origin metrics)
    ("monthly_prophet_tuning_results", "training"),
    ("monthly_prophet_rolling_origin_metrics", "training"),
    ("monthly_sarimax_rolling_origin_metrics", "training"),
    ("monthly_catboost_rolling_origin_metrics", "training"),
    # 06_models — multi-family model selection (active artifacts)
    ("monthly_candidate_metrics", "selection"),
    ("monthly_family_champion_summary", "selection"),
    ("monthly_model_selection_summary", "selection"),
    # 07_model_output / 08_reporting
    ("monthly_operational_test_forecasts", "reporting"),
    ("monthly_operational_lead_time_metrics", "reporting"),
    ("monthly_model_selection_audit", "reporting"),
    # 07_model_output — generic, family-agnostic champion forecasts (Phase 6)
    ("monthly_forecast_3m", "inference"),
    ("monthly_forecast_6m", "inference"),
    ("monthly_forecast_12m", "inference"),
    ("monthly_forecast_latest", "inference"),
]

JSON_DATASETS = [
    ("monthly_prophet_split_metadata", "model_input"),
    ("monthly_prophet_prechampion_configs", "training"),
    ("monthly_prophet_training_metadata", "training"),
    ("monthly_sarimax_training_metadata", "training"),
    ("monthly_catboost_training_metadata", "training"),
    # Active champion metadata — family-agnostic
    ("champion_monthly_metadata", "selection"),
    ("monthly_inference_metadata", "inference"),
]

PICKLE_DATASETS = [
    ("monthly_prophet_candidate_models", "training"),
    ("candidate_monthly_prophet", "training"),
    ("candidate_monthly_sarimax", "training"),
    ("candidate_monthly_catboost", "training"),
    # Active production champion
    ("champion_monthly_model", "selection"),
]


def _df_summary(name: str, stage: str, df: pd.DataFrame) -> str:
    date_cols = [c for c in df.columns if c in ("ds", "date", "month", "period")]
    date_col = date_cols[0] if date_cols else None
    min_date = max_date = "N/A"
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col])
            min_date = str(dates.min().date())
            max_date = str(dates.max().date())
        except Exception:
            pass

    null_counts = df.isnull().sum()
    null_report = "\n".join(
        f"  - {c}: {n}" for c, n in null_counts.items() if n > 0
    ) or "  (none)"

    dtype_lines = "\n".join(f"  - `{c}`: {t}" for c, t in df.dtypes.items())

    return f"""### `{name}` — *{stage}*

- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns
- **Date column:** `{date_col or 'not detected'}`
- **Date range:** {min_date} → {max_date}
- **Columns and dtypes:**
{dtype_lines}
- **Null counts (non-zero only):**
{null_report}

"""


def _json_summary(name: str, stage: str, data: dict | list) -> str:
    if isinstance(data, dict):
        top_keys = list(data.keys())
        sample = json.dumps(data, indent=2, default=str)[:800]
    else:
        top_keys = [f"list[{len(data)}]"]
        sample = json.dumps(data[:2], indent=2, default=str)[:800]

    return f"""### `{name}` — *{stage}*

- **Type:** `{type(data).__name__}`
- **Top-level keys:** {top_keys}
- **Sample (truncated):**

```json
{sample}
```

"""


def _pickle_summary(name: str, stage: str) -> str:
    return f"""### `{name}` — *{stage}*

- **Type:** Pickle artifact
- **Purpose:** See catalog.yml for physical path and expected object type.

"""


def main() -> None:
    bootstrap_project(PROJECT_ROOT)

    sections: list[str] = [
        "# Monthly Forecasting — Contract Snapshot\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}  \n**Pipeline:** `monthly_training_comparison + monthly_model_selection`  \n**Protocol:** rolling-origin champion selection\n\n",
        "---\n\n",
        "## Tabular Datasets\n\n",
    ]

    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        catalog = session.load_context().catalog

        for name, stage in TABULAR_DATASETS:
            try:
                df = catalog.load(name)
                sections.append(_df_summary(name, stage, df))
                print(f"  [OK] {name}: {df.shape}")
            except Exception as exc:
                sections.append(f"### `{name}` — *{stage}*\n\n- **Status:** MISSING — `{exc}`\n\n")
                print(f"  [MISSING] {name}: {exc}")

        sections.append("## JSON / Metadata Artifacts\n\n")

        for name, stage in JSON_DATASETS:
            try:
                data = catalog.load(name)
                sections.append(_json_summary(name, stage, data))
                print(f"  [OK] {name}")
            except Exception as exc:
                sections.append(f"### `{name}` — *{stage}*\n\n- **Status:** MISSING — `{exc}`\n\n")
                print(f"  [MISSING] {name}: {exc}")

        sections.append("## Model Artifacts (Pickle)\n\n")

        for name, stage in PICKLE_DATASETS:
            try:
                _ = catalog.load(name)
                sections.append(_pickle_summary(name, stage))
                print(f"  [OK] {name}")
            except Exception as exc:
                sections.append(f"### `{name}` — *{stage}*\n\n- **Status:** MISSING — `{exc}`\n\n")
                print(f"  [MISSING] {name}: {exc}")

    REPORT_PATH.write_text("".join(sections), encoding="utf-8")
    print(f"\nContract snapshot written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
