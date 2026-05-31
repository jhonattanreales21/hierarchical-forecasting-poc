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


def test_train_monthly_prophet_is_namespaced():
    """Prophet sub-pipeline must carry the train_monthly.prophet namespace prefix."""
    pipelines = register_pipelines()
    train_monthly_node_names = {n.name for n in pipelines["train_monthly"].nodes}
    assert any(
        n.startswith("train_monthly.prophet.") for n in train_monthly_node_names
    ), "No node with 'train_monthly.prophet.' prefix found in train_monthly pipeline"


def test_monthly_mvp_excludes_sarimax_and_catboost_stub_nodes():
    """monthly_mvp must not include scaffolded CatBoost or SARIMAX training stub nodes.

    The SARIMAX model-input adapter (adapt_monthly_data_for_sarimax) is a legitimate
    data-preparation step added in Phase 3 and is allowed in monthly_mvp.  What must
    remain excluded are SARIMAX training stubs (tune_hyperparameters, train_best_candidate)
    and all CatBoost nodes.
    """
    pipelines = register_pipelines()
    mvp_node_names = {n.name for n in pipelines["monthly_mvp"].nodes}
    assert not any("catboost" in n for n in mvp_node_names), (
        "monthly_mvp contains CatBoost stub nodes"
    )
    # Adapter nodes are allowed; only exclude training-level SARIMAX stubs.
    sarimax_training_stubs = {
        n for n in mvp_node_names
        if "sarimax" in n and "adapt" not in n
    }
    assert not sarimax_training_stubs, (
        f"monthly_mvp contains SARIMAX training stub nodes: {sarimax_training_stubs}"
    )


def test_monthly_mvp_aliases_preserved_after_namespace_alignment():
    """__default__, monthly_mvp, and prophet_monthly_e2e must resolve to the same node set."""
    pipelines = register_pipelines()
    default_nodes = {n.name for n in pipelines["__default__"].nodes}
    mvp_nodes = {n.name for n in pipelines["monthly_mvp"].nodes}
    e2e_nodes = {n.name for n in pipelines["prophet_monthly_e2e"].nodes}
    assert default_nodes == mvp_nodes, "__default__ and monthly_mvp diverged"
    assert mvp_nodes == e2e_nodes, "monthly_mvp and prophet_monthly_e2e diverged"


def test_prophet_public_dataset_names_are_preserved():
    """The train_monthly pipeline must still wire to the original public Prophet catalog names."""
    pipelines = register_pipelines()
    prophet_node = next(
        n for n in pipelines["train_monthly"].nodes if "prophet" in n.name
    )
    expected_inputs = {
        "monthly_prophet_train",
        "monthly_prophet_validation",
        "monthly_prophet_split_metadata",
        "params:train_monthly.prophet",
    }
    expected_outputs = {
        "monthly_prophet_tuning_results",
        "monthly_prophet_validation_metrics",
        "monthly_prophet_prechampion_configs",
        "monthly_prophet_candidate_models",
        "monthly_prophet_training_metadata",
        "candidate_monthly_prophet",
    }
    assert expected_inputs == set(prophet_node.inputs), (
        f"Prophet inputs changed: {set(prophet_node.inputs)}"
    )
    assert expected_outputs == set(prophet_node.outputs), (
        f"Prophet outputs changed: {set(prophet_node.outputs)}"
    )
