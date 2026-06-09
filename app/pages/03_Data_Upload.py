from ui.components import render_page_header
from ui.page_blocks.upload_blocks import render_data_upload_page
from ui.styles import apply_global_styles

apply_global_styles()
render_page_header(
    title="Data Upload",
    subtitle=(
        "Upload demand and exogenous CSV data, plus business-history documents for "
        "the assistant. Files are validated and stored; no pipeline is triggered."
    ),
    eyebrow="Inputs Intake",
)

render_data_upload_page()
