# Contributing

## Branching

| Prefix | Purpose |
|--------|---------|
| `feat/*` | New functionality |
| `fix/*` | Bug fixes |
| `refactor/*` | Internal restructuring |
| `docs/*` | Documentation changes |
| `test/*` | Test additions |
| `chore/*` | Tooling, config, maintenance |

Target branches: `feat/*` and `fix/*` go to `dev`; hotfixes go to `main`.

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(matching): add exact matching without replacement
fix(metrics): correct pooled variance in SMD calculation
docs(readme): rewrite overview and usage examples
```

## Running checks locally

```bash
# Lint all packages
uv run --package shared ruff check shared/
uv run --package hierarchical_demand_forecasting_poc ruff check pipelines/
uv run --package hdf_app ruff check app/
uv run --package hdf_api ruff check api/

# Tests
uv run --package hierarchical_demand_forecasting_poc pytest pipelines/ --cov
```
