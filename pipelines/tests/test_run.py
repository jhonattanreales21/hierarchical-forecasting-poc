"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

# Resolve to pipelines/ regardless of where pytest is invoked from
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestKedroRun:
    def test_model_input_preparation_run(self):
        bootstrap_project(_PROJECT_ROOT)

        with KedroSession.create(project_path=_PROJECT_ROOT) as session:
            session.run(pipeline_names=["model_input_preparation"])

        expected_outputs = [
            "monthly_prophet_modeling_data.parquet",
            "monthly_prophet_train.parquet",
            "monthly_prophet_validation.parquet",
            "monthly_prophet_test.parquet",
            "monthly_prophet_full_train.parquet",
            "monthly_prophet_future_3m.parquet",
            "monthly_prophet_future_6m.parquet",
            "monthly_prophet_split_metadata.json",
        ]
        output_dir = _PROJECT_ROOT / "data" / "05_model_input"
        for filename in expected_outputs:
            assert (output_dir / filename).exists()
