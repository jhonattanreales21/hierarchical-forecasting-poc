"""Tests for shared monthly training utilities."""

from __future__ import annotations

import logging

from hdf_pipelines.pipelines.train_monthly.nodes import log_trial_predictions


def test_log_trial_predictions_deduplicates_overlapping_target_months(caplog):
    trial_preds = [
        {
            "target_start": "2025-12",
            "target_end": "2026-02",
            "target_dates": ["2025-12", "2026-01", "2026-02"],
            "y_pred": [10.0, 20.0, 30.0],
        },
        {
            "target_start": "2026-01",
            "target_end": "2026-03",
            "target_dates": ["2026-01", "2026-02", "2026-03"],
            "y_pred": [21.0, 31.0, 40.0],
        },
        {
            "target_start": "2026-02",
            "target_end": "2026-04",
            "target_dates": ["2026-02", "2026-03", "2026-04"],
            "y_pred": [32.0, 41.0, 50.0],
        },
        {
            "target_start": "2026-03",
            "target_end": "2026-05",
            "target_dates": ["2026-03", "2026-04", "2026-05"],
            "y_pred": [42.0, 51.0, 60.0],
        },
    ]

    with caplog.at_level(logging.INFO):
        log_trial_predictions("prophet_candidate_001", trial_preds)

    assert "6 unique-month preds (12 cycle-step preds)" in caplog.text
    assert "2025-12=10.0" in caplog.text
    assert "2026-01=21.0" in caplog.text
    assert "2026-02=32.0" in caplog.text
    assert "2026-03=42.0" in caplog.text
    assert "2026-04=51.0" in caplog.text
    assert "2026-05=60.0" in caplog.text
