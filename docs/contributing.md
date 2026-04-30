# Contributing

This guide explains how to contribute to the project and what process to follow before opening a pull request.

The goal is to keep contributions focused, reviewable, reproducible, and aligned with the project structure: a mono-repo with Kedro pipelines, shared utilities, a Streamlit app, a FastAPI API, and Docker support.

---

## Contribution Principles

Contributions should follow these principles:

- Keep changes small, focused, and easy to review.
- Prefer clear, explicit code over clever implementations.
- Do not mix unrelated changes in the same branch or pull request.
- Keep reusable logic out of notebooks and inside the appropriate package.
- Update tests and documentation when behavior, usage, or public interfaces change.
- Validate changes locally before opening a pull request.

Use the specialized documentation for deeper guidance:

- `docs/architecture.md` for architecture and package responsibilities.
- `docs/coding-standards.md` for style, typing, docstrings, logging, and errors.
- `docs/forecasting.md` for modeling, metrics, and forecasting conventions.
- `docs/setup.md` for detailed environment, Docker, and local service setup.

---

## Quick Setup

From the repository root:

```bash
# Install all workspace dependencies
uv sync --all-packages

# Optional: verify available workspace packages
uv tree
```

For Kedro commands, move into the `pipelines/` directory:

```bash
cd pipelines

# List registered pipelines
uv run kedro pipeline list

# Run a specific pipeline
uv run kedro run --pipeline data_ingestion
```

For local app and API checks:

```bash
# Streamlit app
uv run --package hdf_app streamlit run app/app.py

# FastAPI API
uv run --package hdf_api uvicorn hdf_api.main:app --reload --port 8000
```

---

## Recommended Contribution Flow

1. Create a branch from `dev`.
2. Make a small, focused change.
3. Add or update tests when relevant.
4. Run linting and tests locally.
5. Review your own diff before opening the pull request.
6. Open a pull request to `dev`, unless it is a release or hotfix flow.
7. Fill out the PR description with summary, changes, validation, and reviewer notes.

Avoid committing generated data, local configuration, credentials, `.env` files, notebook outputs, or unrelated formatting changes.

---

## Branching

| Prefix | Purpose |
|--------|---------|
| `feat/*` | New functionality |
| `fix/*` | Bug fixes |
| `refactor/*` | Internal restructuring |
| `docs/*` | Documentation changes |
| `test/*` | Test additions |
| `chore/*` | Tooling, config, maintenance |

Examples:

```
feat/monthly-prophet-training
refactor/forecast-metrics-utils
docs/update-architecture-guide
```

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(monthly): add prophet training pipeline
fix(metrics): correct mase calculation for short series
docs(readme): update project overview and usage examples
test(pipelines): add smoke test for data ingestion pipeline
chore(docker): update api service build configuration
```

## Pull Requests

### PR Title

Use the following format:

```sh
[<TYPE>] <Short clear description>
```

Examples:

```sh
[FEATURE] Add monthly Prophet training pipeline
[FIX] Resolve Kedro catalog path for model outputs
[DOCS] Improve README overview and setup instructions
[REFACTOR] Centralize forecast metric utilities
[TEST] Add smoke tests for data ingestion pipeline
[CHORE] Update Docker Compose services for app and API
```

Suggested PR types:

- `FEATURE`
- `FIX`
- `REFACTOR`
- `DOCS`
- `TEST`
- `CHORE`

### PR Description

Every PR should include, at minimum:

- A short summary of the change
- The main changes introduced
- How the change was validated
- Notes for reviewers when relevant

Recommended sections:

```md
## Summary
Brief description of the change and why it is needed.

## Changes
- 
- 
- 

## Validation
- Describe how the changes were validated.

## Notes for reviewers
- Optional reviewer guidance.
```

## Running checks locally

```bash
# Lint all packages
uv run --package shared ruff check shared/
uv run --package hdf_pipelines ruff check pipelines/
uv run --package hdf_app ruff check app/
uv run --package hdf_api ruff check api/

# Tests
uv run --package hdf_pipelines pytest pipelines/ --cov
```

---

## Pull Request Checklist

Before requesting review, confirm that:

- The branch name follows the expected convention.
- The PR has a clear title and description.
- The change is limited to one topic or objective.
- Local linting and tests were executed, or skipped with a clear reason.
- New or changed behavior is covered by tests when practical.
- Documentation was updated when setup, usage, commands, or outputs changed.
- No data files, credentials, `.env` files, or local-only artifacts were committed.
- Notebook outputs were removed before committing, when notebooks are included.
- Kedro datasets are read and written through the catalog, not through hardcoded paths.
- The app, API, or Docker changes were validated locally when they are affected.

---

## Review Expectations

Reviewers should check that the contribution is focused, understandable, and safe to merge.

A PR may require changes if it introduces:

- unrelated changes bundled together,
- hardcoded paths, credentials, or environment-specific assumptions,
- reusable logic hidden inside notebooks,
- direct data I/O inside Kedro nodes instead of catalog usage,
- missing validation for important input assumptions,
- failing linting or tests without explanation,
- unclear PR description or missing validation notes.
