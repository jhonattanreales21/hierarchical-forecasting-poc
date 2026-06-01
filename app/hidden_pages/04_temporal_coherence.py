from ui.components import render_page_header
from ui.page_blocks.coherence_blocks import (
    render_expected_reconciliation_artifacts,
    render_reconciliation_status,
    render_temporal_coherence_intro,
)
from ui.styles import apply_global_styles

apply_global_styles()

render_page_header(
    title="Temporal Coherence",
    subtitle="Monthly ↔ Weekly ↔ Daily reconciliation concept and current pipeline status.",
    eyebrow="Reconciliation Layer",
)

render_temporal_coherence_intro()
render_reconciliation_status()
render_expected_reconciliation_artifacts()
