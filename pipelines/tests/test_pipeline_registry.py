"""Smoke tests for the Kedro pipeline registry."""

from hdf_pipelines.pipeline_registry import register_pipelines


def test_all_expected_pipelines_registered():
    pipelines = register_pipelines()
    expected = {
        "__default__",
        "monthly_mvp",
        "prophet_monthly_e2e",
        "data_ingestion",
        "feature_engineering_monthly",
        "feature_engineering_weekly",
        "model_input_preparation",
        "train_monthly",
        "train_weekly",
        "model_selection",
        "reconciliation",
        "forecast_inference",
        "experimental_training",
        "experimental_inference",
        "experimental_full_experiment",
    }
    assert expected.issubset(set(pipelines.keys()))


def test_monthly_mvp_is_registered():
    pipelines = register_pipelines()
    assert "monthly_mvp" in pipelines


def test_default_is_monthly_mvp():
    pipelines = register_pipelines()
    default_nodes = {n.name for n in pipelines["__default__"].nodes}
    mvp_nodes = {n.name for n in pipelines["monthly_mvp"].nodes}
    assert default_nodes == mvp_nodes


def test_default_has_no_scaffolded_weekly_nodes():
    """Ensure __default__ does not include the scaffolded weekly feature/training pipelines."""
    pipelines = register_pipelines()
    scaffolded_weekly_nodes = {n.name for n in pipelines["feature_engineering_weekly"].nodes} | {
        n.name for n in pipelines["train_weekly"].nodes
    }
    default_node_names = {n.name for n in pipelines["__default__"].nodes}
    assert scaffolded_weekly_nodes.isdisjoint(default_node_names)


def test_default_pipeline_is_not_empty():
    pipelines = register_pipelines()
    assert len(pipelines["__default__"].nodes) > 0


def test_experimental_training_contains_both_granularities():
    pipelines = register_pipelines()
    training_nodes = {n.name for n in pipelines["experimental_training"].nodes}
    assert any("monthly" in n for n in training_nodes)
    assert any("weekly" in n for n in training_nodes)


def test_experimental_composed_shortcuts_are_subsets():
    pipelines = register_pipelines()
    training_nodes = {n.name for n in pipelines["experimental_training"].nodes}
    monthly_nodes = {n.name for n in pipelines["train_monthly"].nodes}
    weekly_nodes = {n.name for n in pipelines["train_weekly"].nodes}
    assert monthly_nodes.issubset(training_nodes)
    assert weekly_nodes.issubset(training_nodes)
