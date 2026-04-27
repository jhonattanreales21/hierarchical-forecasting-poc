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

## Pull Requests

### PR Title

Use the following format:

```sh
[<TYPE>] <Short clear description>
```

Examples:

```sh
[FEATURE] Add exact matcher fit and match flow
[FIX] Correct balance table output for binary variables
[DOCS] Improve README overview and usage examples
[REFACTOR] Centralize matching input validations
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
