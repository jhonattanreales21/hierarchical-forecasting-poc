"""Tests for monthly family-champion explainability (SHAP + native drivers).

Covers:
- compute_catboost_shap_importance (real tiny CatBoost + shap.TreeExplainer)
- compute_prophet_regressor_importance (centered component contributions, fake model)
- compute_sarimax_coefficient_importance (x# → readable name mapping, fake results)
- assemble_family_importance_table (unified schema + within-family ranking)
- generate_monthly_family_champion_explanations (orchestration, guards, graceful skip)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from hdf_pipelines.pipelines.model_selection.monthly import nodes
from hdf_pipelines.pipelines.model_selection.monthly.explainability import (
    IMPORTANCE_COLUMNS,
    assemble_family_importance_table,
    compute_catboost_shap_importance,
    compute_prophet_regressor_importance,
    compute_sarimax_coefficient_importance,
)

# ── Test doubles ──────────────────────────────────────────────────────────────


class _FakeProphet:
    """Prophet test double returning a component frame from ``predict``."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {
                "ds": df["ds"].to_numpy(),
                "yhat": np.arange(n, dtype=float),
                "trend": np.linspace(100.0, 110.0, n),
                "promo": np.array([0.0, 2.0, -2.0, 0.0, 1.0, -1.0])[:n],
            }
        )


class _FakeSarimaxResults:
    """statsmodels SARIMAXResults test double (exog named x1, x2)."""

    param_names = ["x1", "x2", "ar.L1", "sigma2"]
    params = np.array([2.0, -1.0, 0.5, 1.0])
    bse = np.array([0.10, 0.20, 0.05, 0.30])
    pvalues = np.array([0.001, 0.200, 0.400, 0.000])


# ── compute_catboost_shap_importance ────────────────────────────────────────────


def test_compute_catboost_shap_importance_shapes_and_ranking() -> None:
    rng = np.random.default_rng(0)
    n = 80
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # f1 has a stronger effect than f2, so its mean |SHAP| should be larger.
    y = 3.0 * f1 - 1.0 * f2 + rng.normal(scale=0.05, size=n)
    feature_names = ["f1", "f2"]
    x_df = pd.DataFrame({"f1": f1, "f2": f2})

    model = CatBoostRegressor(
        iterations=40, depth=3, learning_rate=0.2, random_seed=0, verbose=False
    )
    model.fit(x_df, y)

    importance, shap_values, explainer, base_value = compute_catboost_shap_importance(
        model, x_df, feature_names
    )

    assert list(importance["feature"]) == feature_names  # input order preserved
    assert (importance["importance"] >= 0).all()
    assert importance["importance_type"].eq("mean_abs_shap").all()
    assert shap_values.shape == (n, len(feature_names))
    assert explainer is not None
    assert isinstance(base_value, float)

    importance_by_feature = dict(zip(importance["feature"], importance["importance"]))
    assert importance_by_feature["f1"] > importance_by_feature["f2"]


# ── compute_prophet_regressor_importance ─────────────────────────────────────────


def test_compute_prophet_regressor_importance_includes_regressor_and_component() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=6, freq="MS"),
            "y": np.linspace(10.0, 20.0, 6),
            "promo": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        }
    )

    importance = compute_prophet_regressor_importance(
        _FakeProphet(), df, ["promo"], include_components=True
    )

    features = set(importance["feature"])
    assert "promo" in features
    assert "[trend]" in features  # component included and bracketed
    assert (importance["importance"] >= 0).all()
    assert importance["importance_type"].eq("mean_abs_contribution").all()


def test_compute_prophet_regressor_importance_no_components() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=6, freq="MS"),
            "promo": np.array([0.0, 2.0, -2.0, 0.0, 1.0, -1.0]),
        }
    )
    importance = compute_prophet_regressor_importance(
        _FakeProphet(), df, ["promo"], include_components=False
    )
    assert list(importance["feature"]) == ["promo"]


# ── compute_sarimax_coefficient_importance ───────────────────────────────────────


def test_compute_sarimax_coefficient_importance_maps_exog_names() -> None:
    importance = compute_sarimax_coefficient_importance(
        _FakeSarimaxResults(), ["promo", "price"]
    )

    # Only the exogenous x1/x2 terms are kept; ar/sigma2 are excluded.
    assert set(importance["feature"]) == {"promo", "price"}
    assert importance["importance_type"].eq("abs_coefficient").all()

    by_feature = importance.set_index("feature")
    assert by_feature.loc["promo", "importance"] == pytest.approx(2.0)
    assert by_feature.loc["promo", "coefficient"] == pytest.approx(2.0)
    assert by_feature.loc["price", "coefficient"] == pytest.approx(-1.0)
    assert by_feature.loc["price", "importance"] == pytest.approx(1.0)
    assert by_feature.loc["promo", "pvalue"] == pytest.approx(0.001)


def test_compute_sarimax_coefficient_importance_no_exog() -> None:
    class _NoExog:
        param_names = ["ar.L1", "sigma2"]
        params = np.array([0.5, 1.0])
        bse = np.array([0.1, 0.2])
        pvalues = np.array([0.3, 0.0])

    importance = compute_sarimax_coefficient_importance(_NoExog(), [])
    assert importance.empty


# ── assemble_family_importance_table ─────────────────────────────────────────────


def test_assemble_family_importance_table_schema_and_ranking() -> None:
    per_family = {
        "sarimax": {
            "champion_id": "sarimax_trial_001",
            "importance": pd.DataFrame(
                {
                    "feature": ["a", "b"],
                    "importance": [1.0, 2.0],
                    "importance_type": ["abs_coefficient", "abs_coefficient"],
                    "coefficient": [1.0, -2.0],
                    "std_err": [0.1, 0.2],
                    "pvalue": [0.5, 0.01],
                }
            ),
        },
        "prophet": {
            "champion_id": "prophet_candidate_001",
            "importance": pd.DataFrame(
                {
                    "feature": ["[trend]"],
                    "importance": [5.0],
                    "importance_type": ["mean_abs_contribution"],
                }
            ),
        },
    }

    out = assemble_family_importance_table(per_family, "2026-06-07T00:00:00+00:00")

    assert list(out.columns) == IMPORTANCE_COLUMNS
    sarimax_rows = out[out["family"] == "sarimax"].sort_values("rank")
    assert list(sarimax_rows["feature"]) == ["b", "a"]  # ranked by importance desc
    assert list(sarimax_rows["rank"]) == [1, 2]
    # Prophet rows have no coefficient detail -> NaN-filled, not missing columns.
    assert out[out["family"] == "prophet"]["coefficient"].isna().all()


def test_assemble_family_importance_table_empty() -> None:
    out = assemble_family_importance_table({}, "2026-06-07T00:00:00+00:00")
    assert out.empty
    assert list(out.columns) == IMPORTANCE_COLUMNS


# ── generate_monthly_family_champion_explanations (node) ─────────────────────────


def _empty_frames() -> tuple[pd.DataFrame, ...]:
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def test_node_skips_when_disabled() -> None:
    summary = pd.DataFrame(
        {"family": ["sarimax"], "family_champion_id": ["sarimax_trial_001"]}
    )
    prophet_ft, sarimax_ft, catboost_ft = _empty_frames()

    importance, explainer, shap_values, metadata = (
        nodes.generate_monthly_family_champion_explanations(
            monthly_family_champion_summary=summary,
            monthly_prophet_candidate_models={},
            monthly_sarimax_candidate_models={},
            monthly_prophet_full_train=prophet_ft,
            monthly_sarimax_full_train=sarimax_ft,
            monthly_sarimax_training_metadata={},
            params_monthly={"explainability": {"enabled": False}},
            monthly_catboost_candidate_models={},
            monthly_catboost_full_train=catboost_ft,
            monthly_catboost_split_metadata={},
        )
    )

    assert importance.empty
    assert explainer == {}
    assert shap_values.empty
    assert metadata["enabled"] is False


def test_node_handles_empty_summary() -> None:
    prophet_ft, sarimax_ft, catboost_ft = _empty_frames()
    importance, explainer, shap_values, metadata = (
        nodes.generate_monthly_family_champion_explanations(
            monthly_family_champion_summary=pd.DataFrame(),
            monthly_prophet_candidate_models={},
            monthly_sarimax_candidate_models={},
            monthly_prophet_full_train=prophet_ft,
            monthly_sarimax_full_train=sarimax_ft,
            monthly_sarimax_training_metadata={},
            params_monthly={"explainability": {"enabled": True}},
            monthly_catboost_candidate_models={},
            monthly_catboost_full_train=catboost_ft,
            monthly_catboost_split_metadata={},
        )
    )
    assert importance.empty
    assert explainer == {}
    assert metadata["n_families_explained"] == 0


def test_node_builds_sarimax_importance() -> None:
    summary = pd.DataFrame(
        {"family": ["sarimax"], "family_champion_id": ["sarimax_trial_001"]}
    )
    prophet_ft, sarimax_ft, catboost_ft = _empty_frames()

    fake_champion = {"model": _FakeSarimaxResults()}
    fake_contract = {"active_regressors": ["promo", "price"]}

    with (
        patch.object(nodes, "_resolve_champion_model", return_value={"config": {}}),
        patch.object(
            nodes,
            "_build_production_champion_model",
            return_value=(fake_champion, fake_contract, {}),
        ),
    ):
        importance, explainer, shap_values, metadata = (
                nodes.generate_monthly_family_champion_explanations(
                    monthly_family_champion_summary=summary,
                    monthly_prophet_candidate_models={},
                    monthly_sarimax_candidate_models={
                        "sarimax_trial_001": {"config": {}}
                    },
                    monthly_prophet_full_train=prophet_ft,
                    monthly_sarimax_full_train=sarimax_ft,
                    monthly_sarimax_training_metadata={},
                    params_monthly={"explainability": {"enabled": True}},
                    monthly_catboost_candidate_models={},
                    monthly_catboost_full_train=catboost_ft,
                    monthly_catboost_split_metadata={},
                )
        )

    assert explainer == {}
    assert shap_values.empty
    sarimax_rows = importance[importance["family"] == "sarimax"]
    assert set(sarimax_rows["feature"]) == {"promo", "price"}
    assert metadata["families"]["sarimax"]["method"] == "sarimax_coefficients"
    assert metadata["n_families_explained"] == 1


def test_node_records_error_and_continues_on_family_failure() -> None:
    summary = pd.DataFrame(
        {"family": ["sarimax"], "family_champion_id": ["sarimax_trial_001"]}
    )
    prophet_ft, sarimax_ft, catboost_ft = _empty_frames()

    with patch.object(
        nodes, "_resolve_champion_model", side_effect=RuntimeError("not found")
    ):
        importance, explainer, shap_values, metadata = (
            nodes.generate_monthly_family_champion_explanations(
                monthly_family_champion_summary=summary,
                monthly_prophet_candidate_models={},
                monthly_sarimax_candidate_models={},
                monthly_prophet_full_train=prophet_ft,
                monthly_sarimax_full_train=sarimax_ft,
                monthly_sarimax_training_metadata={},
                params_monthly={"explainability": {"enabled": True}},
                monthly_catboost_candidate_models={},
                monthly_catboost_full_train=catboost_ft,
                monthly_catboost_split_metadata={},
            )
        )

    assert importance.empty  # the only family failed, so nothing was assembled
    assert metadata["families"]["sarimax"]["method"] is None
    assert "not found" in metadata["families"]["sarimax"]["error"]
    assert metadata["n_families_explained"] == 0
