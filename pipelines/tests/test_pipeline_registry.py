"""Smoke tests for the Kedro pipeline registry."""

from hdf_pipelines.pipeline_registry import register_pipelines


def test_all_expected_pipelines_registered():
    pipelines = register_pipelines()
    expected = {
        "__default__",
        "data_ingestion",
        "feature_engineering_monthly",
        "feature_engineering_weekly",
        "model_input_preparation",
        "train_monthly",
        "train_weekly",
        "model_selection",
        "reconciliation",
        "forecast_inference",
        "training",
        "inference",
        "full_experiment",
    }
    assert expected.issubset(set(pipelines.keys()))


def test_training_shortcut_contains_both_granularities():
    pipelines = register_pipelines()
    training_nodes = {n.name for n in pipelines["training"].nodes}
    assert any("monthly" in n for n in training_nodes)
    assert any("weekly" in n for n in training_nodes)


def test_default_pipeline_is_not_empty():
    pipelines = register_pipelines()
    assert len(pipelines["__default__"].nodes) > 0


def test_composed_shortcuts_are_subsets():
    pipelines = register_pipelines()
    training_nodes = {n.name for n in pipelines["training"].nodes}
    monthly_nodes = {n.name for n in pipelines["train_monthly"].nodes}
    weekly_nodes = {n.name for n in pipelines["train_weekly"].nodes}
    assert monthly_nodes.issubset(training_nodes)
    assert weekly_nodes.issubset(training_nodes)
