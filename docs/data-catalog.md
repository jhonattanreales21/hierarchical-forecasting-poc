# Data Catalog

All datasets are defined in `pipelines/conf/base/catalog.yml`.
The Kedro data layer follows a numbered-layer convention.

| Layer | Path | Contents |
|-------|------|----------|
| 01_raw | `pipelines/data/01_raw/` | Raw demand and exogenous CSVs |
| 02_intermediate | `pipelines/data/02_intermediate/` | Cleaned, validated parquets |
| 03_primary | `pipelines/data/03_primary/` | Domain-level weekly and monthly demand |
| 04_feature | `pipelines/data/04_feature/` | Feature-engineered datasets |
| 05_model_input | `pipelines/data/05_model_input/` | Train / val / test splits per model |
| 06_models | `pipelines/data/06_models/` | Trained artifacts, champion registry, tuning results |
| 07_model_output | `pipelines/data/07_model_output/` | Raw and reconciled forecasts |
| 08_reporting | `pipelines/data/08_reporting/` | Evaluation reports, reconciliation diagnostics |

> Data files are gitignored. Place raw inputs in `pipelines/data/01_raw/` before running the pipeline.
