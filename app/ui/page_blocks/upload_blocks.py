"""Data Upload page blocks: demand/exogenous CSV intake and assistant documents."""

from __future__ import annotations

import io
import time

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from shared.rag import save_uploaded_file
from shared.upload_validation import summarize_csv, summarize_document
from ui.components import (
    render_kpi_card,
    render_section_header,
    render_success_banner,
    render_warning_banner,
)
from utils.paths import ASSISTANT_UPLOADS
from utils.uploads import (
    UPLOAD_CATEGORY_LABELS,
    UploadCategory,
    latest_uploads,
    read_upload_manifest,
    save_upload_bytes,
)

_DOC_TYPES = ["pdf", "docx", "md", "markdown", "txt"]

# Tune this value to control how long the simulated loading animation runs (in seconds).
_UPLOAD_SIMULATION_SECONDS: float = 20.0

# Pipeline phases shown during the loading simulation.
# Each entry: (display name, fraction of total time, list of step messages).
# Fractions sum to 1.0; the training phase carries the largest share to mirror reality.
_PIPELINE_PHASES: list[tuple[str, float, list[str]]] = [
    (
        "Data ingestion & validation",
        0.10,
        [
            "Validating file schema and encoding",
            "Parsing demand time series",
            "Parsing exogenous variables",
            "Detecting temporal granularity",
            "Checking date continuity and duplicates",
        ],
    ),
    (
        "Feature engineering",
        0.12,
        [
            "Aggregating demand to monthly level",
            "Building lag and rolling-window features",
            "Generating calendar and seasonal features",
            "Merging exogenous variables into feature matrix",
        ],
    ),
    (
        "Model input preparation",
        0.08,
        [
            "Building rolling-origin backtesting windows",
            "Preparing expanding-window train splits",
            "Generating horizon-shifted target arrays (h=1, 2, 3)",
        ],
    ),
    (
        "Model training",
        0.45,
        [
            "SARIMAX — Optuna hyperparameter search",
            "SARIMAX — refitting best configuration on full history",
            "Prophet — Optuna hyperparameter search",
            "Prophet — refitting best configuration on full history",
            "CatBoost (h=1) — direct multi-horizon training",
            "CatBoost (h=2) — direct multi-horizon training",
            "CatBoost (h=3) — direct multi-horizon training",
            "Running rolling-origin cross-validation on all candidates",
        ],
    ),
    (
        "Model selection",
        0.15,
        [
            "Pooling WMAPE across rolling-origin backtest cycles",
            "Computing MASE and RMSE per forecast horizon",
            "Electing family champions (SARIMAX, Prophet, CatBoost)",
            "Electing production champion across families",
        ],
    ),
    (
        "Forecast inference",
        0.10,
        [
            "Refitting production champion on full history",
            "Generating 3-month ahead point forecasts",
            "Persisting forecast artifacts to catalog",
        ],
    ),
]


def _format_bytes(num_bytes: int) -> str:
    """Return a human-readable file size (e.g. ``1.2 MB``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _read_uploaded_csv(file) -> pd.DataFrame:
    """Parse an uploaded CSV file object into a DataFrame."""
    return pd.read_csv(io.BytesIO(file.getvalue()))


def _render_csv_summary(label: str, summary: dict) -> None:
    """Render a CSV validation summary as KPI cards plus a date-range caption."""
    render_section_header(label)
    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Rows", f"{summary['n_rows']:,}")
    with cols[1]:
        render_kpi_card("Columns", str(summary["n_columns"]))
    with cols[2]:
        render_kpi_card("Granularity", str(summary["granularity"]).title())

    if summary["date_column"] and summary["date_min"]:
        st.caption(
            f"Date column: `{summary['date_column']}` · "
            f"Range: {summary['date_min']} → {summary['date_max']}"
        )
    else:
        render_warning_banner(
            "No date column was detected, so granularity could not be inferred.",
            title="Missing date column",
        )


def _simulate_upload_processing(
    duration_seconds: float = _UPLOAD_SIMULATION_SECONDS,
) -> None:
    """Show a staged loading animation that simulates the full forecasting pipeline.

    Phases and steps mirror the real Kedro pipeline order so the animation is
    convincing once live training is wired in behind it.

    Args:
        duration_seconds: Total wall-clock seconds distributed across phases
            according to each phase's weight in ``_PIPELINE_PHASES``.
    """
    total_steps = sum(len(steps) for _, _, steps in _PIPELINE_PHASES)
    completed = 0

    with st.status("Running forecasting pipeline…", expanded=True) as loader:
        progress_bar = st.progress(0)
        for phase_name, weight, steps in _PIPELINE_PHASES:
            st.write(f"**▸ {phase_name}**")
            per_step = (duration_seconds * weight) / len(steps)
            for step in steps:
                st.write(f"→ {step}")
                time.sleep(per_step)
                completed += 1
                progress_bar.progress(completed / total_steps)
        loader.update(
            label="Pipeline complete — production champion elected.",
            state="complete",
            expanded=False,
        )


def _process_csv_pair(demand_file, exogenous_file) -> None:
    """Validate, summarize, and persist the demand + exogenous CSV pair."""
    pairs: list[tuple[str, UploadedFile, UploadCategory]] = [
        ("Demand Summary", demand_file, "demand"),
        ("Exogenous Summary", exogenous_file, "exogenous_variables"),
    ]
    for label, file, category in pairs:
        try:
            summary = summarize_csv(_read_uploaded_csv(file))
        except (ValueError, pd.errors.ParserError) as exc:
            render_warning_banner(
                f"Could not read `{file.name}`: {exc}", title="Invalid CSV"
            )
            continue

        try:
            save_upload_bytes(
                content=file.getvalue(), filename=file.name, category=category
            )
        except ValueError as exc:
            render_warning_banner(str(exc), title="Upload rejected")
            continue

        _render_csv_summary(label, summary)

    render_success_banner(
        "Demand and exogenous files were saved. No pipeline or training was triggered.",
        title="Saved",
    )


def _process_document(document_file) -> None:
    """Persist an assistant knowledge document and render its summary."""
    saved_path = save_uploaded_file(
        document_file, ASSISTANT_UPLOADS, document_file.name
    )
    summary = summarize_document(saved_path)

    render_section_header("Document Summary")
    cols = st.columns(2)
    with cols[0]:
        render_kpi_card("Size", _format_bytes(summary["size_bytes"]))
    with cols[1]:
        if summary["kind"] == "pdf":
            render_kpi_card("Pages", str(summary["n_pages"]))
        else:
            render_kpi_card("Words", f"{summary['n_words']:,}")

    st.caption(f"Saved document: `{summary['filename']}`")
    render_success_banner(
        "Document saved. Build the RAG index from the Business Assistant page to use it.",
        title="Saved",
    )


def _render_latest_uploads() -> None:
    """Render a compact table of the most recent manifest uploads."""
    records = latest_uploads(read_upload_manifest(), limit=6)
    if not records:
        return
    st.caption("Latest saved uploads")
    display = [
        {
            "Category": UPLOAD_CATEGORY_LABELS.get(
                record["category"], record["category"]
            ),
            "File": record["filename"],
            "Status": record["status"],
            "Uploaded": record["uploaded_at"],
        }
        for record in records
    ]
    st.dataframe(display, width="stretch", hide_index=True)


def render_data_upload_page() -> None:
    """Render the three-uploader Data Upload workflow.

    Demand and exogenous CSV files share a single submit button that activates only
    when both are present; the assistant knowledge document has an independent submit
    button. Submitting validates and stores files and shows a summary — it does not
    trigger any pipeline or model training.
    """
    render_section_header(
        "Demand & Exogenous Data",
        description=(
            "Upload demand and exogenous-variable CSV files. Both are required "
            "before the data can be submitted."
        ),
    )
    cols = st.columns(2)
    with cols[0]:
        demand_file = st.file_uploader(
            "Demand data (CSV)", type=["csv"], key="upload_demand"
        )
    with cols[1]:
        exogenous_file = st.file_uploader(
            "Exogenous variables (CSV)", type=["csv"], key="upload_exogenous"
        )

    both_ready = demand_file is not None and exogenous_file is not None
    if not both_ready:
        st.caption("Load both demand and exogenous CSV files to enable submission.")
    if st.button(
        "Submit data",
        type="primary",
        width="stretch",
        disabled=not both_ready,
        key="submit_data",
    ):
        _simulate_upload_processing()
        _process_csv_pair(demand_file, exogenous_file)

    render_section_header(
        "Assistant Knowledge Document",
        description=(
            "Upload a business-history document (PDF, Word, or Markdown) for the "
            "Business Assistant. The RAG index is built later from the assistant page."
        ),
    )
    document_file = st.file_uploader(
        "Knowledge document",
        type=_DOC_TYPES,
        key="upload_document",
        help="PDF, DOCX, MD/Markdown, and TXT are accepted.",
    )
    if st.button(
        "Submit document",
        width="stretch",
        disabled=document_file is None,
        key="submit_document",
    ):
        _process_document(document_file)

    _render_latest_uploads()
