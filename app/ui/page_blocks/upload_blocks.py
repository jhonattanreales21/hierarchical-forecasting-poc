"""Reusable upload UI blocks for future RAG and pipeline inputs."""

from __future__ import annotations

import streamlit as st

from ui.components import render_section_header
from utils.uploads import (
    ALLOWED_UPLOAD_SUFFIXES,
    UPLOAD_CATEGORY_LABELS,
    UploadCategory,
    latest_uploads,
    read_upload_manifest,
    save_upload_bytes,
)

_CATEGORY_OPTIONS: list[UploadCategory] = [
    "queries",
    "exogenous_variables",
    "rag_documents",
]


def _allowed_type_list(category: UploadCategory) -> list[str]:
    """Return Streamlit file types without leading dots."""
    return sorted(suffix.lstrip(".") for suffix in ALLOWED_UPLOAD_SUFFIXES[category])


def render_upload_panel(key_prefix: str, compact: bool = False) -> None:
    """Render first-pass file upload controls without triggering downstream work."""
    title = "File Intake" if compact else "Upload Inputs For Future Workflows"
    description = (
        "Save query files, external-variable files, or RAG documents for a later "
        "pipeline/RAG task. This panel only stores files and records metadata."
    )
    render_section_header(title, description=None if compact else description)

    category = st.selectbox(
        "Upload category",
        options=_CATEGORY_OPTIONS,
        format_func=lambda value: UPLOAD_CATEGORY_LABELS[value],
        key=f"{key_prefix}_upload_category",
    )
    allowed_types = _allowed_type_list(category)
    uploaded_files = st.file_uploader(
        "Files",
        type=allowed_types,
        accept_multiple_files=True,
        key=f"{key_prefix}_uploads",
        help="Files are saved locally to app/.cache/user_uploads and added to a manifest.",
    )

    if st.button(
        "Save uploads",
        key=f"{key_prefix}_save_uploads",
        width="stretch",
        type="primary" if not compact else "secondary",
    ):
        if not uploaded_files:
            st.warning("Select at least one file to save.")
        else:
            saved = []
            for uploaded in uploaded_files:
                try:
                    saved.append(
                        save_upload_bytes(
                            content=uploaded.getvalue(),
                            filename=uploaded.name,
                            category=category,
                        )
                    )
                except ValueError as exc:
                    st.error(str(exc))
            if saved:
                st.success(f"Saved {len(saved)} file(s). No pipeline or RAG index was run.")

    records = latest_uploads(read_upload_manifest(), limit=3 if compact else 6)
    if records:
        st.caption("Latest saved uploads")
        display = [
            {
                "Category": UPLOAD_CATEGORY_LABELS.get(record["category"], record["category"]),
                "File": record["filename"],
                "Status": record["status"],
                "Uploaded": record["uploaded_at"],
            }
            for record in records
        ]
        st.dataframe(display, width="stretch", hide_index=True)
    elif not compact:
        st.info("No files have been saved through this intake panel yet.")


def render_sidebar_upload_panel(key_prefix: str) -> None:
    """Render the compact upload panel inside the Streamlit sidebar."""
    with st.sidebar:
        render_upload_panel(key_prefix=key_prefix, compact=True)
