# Architecture

> High-level description of the system architecture and data flow.
> Expand this document as the PoC evolves.

## Overview

The system is structured as a mono-repo with four uv workspace packages:

| Package | Role |
|---------|------|
| `pipelines` | Kedro project — all data and ML logic |
| `shared` | Internal library — schemas, metrics, loaders, viz |
| `app` | Streamlit forecast viewer |
| `api` | FastAPI serving layer |

## Data Flow

```
Raw data (CSV)
    └─► data_ingestion
            └─► feature_engineering_monthly / _weekly
                    └─► model_input_preparation
                            └─► train_monthly / train_weekly
                                    └─► model_selection
                                            └─► reconciliation
                                                    └─► forecast_inference
                                                            └─► data/07_model_output/
                                                                    ├─ monthly_forecast_raw.parquet
                                                                    ├─ monthly_forecast_reconciled.parquet
                                                                    ├─ weekly_forecast_raw.parquet
                                                                    └─ weekly_forecast_reconciled.parquet
```

## Temporal Hierarchy

```
Monthly  <->  Weekly  <->  Daily (disabled)
[primary]     [anchor]     [optional extension]
```

- **Monthly** is the primary analytical and stakeholder-facing layer.
- **Weekly** is a 14-week operational complement.
- **Daily** allocation is disabled by default (`daily_allocation.enabled: false`).

## Reconciliation

Default method: `mint_shrink` (configured in `conf/base/parameters/reconciliation.yml`).
Primary target: monthly ↔ weekly coherence.
