"""Upload persistence helpers for first-pass app file intake."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from utils.paths import USER_UPLOAD_MANIFEST, USER_UPLOADS_ROOT

UploadCategory = Literal[
    "demand", "queries", "exogenous_variables", "rag_documents"
]

ALLOWED_UPLOAD_SUFFIXES: dict[UploadCategory, set[str]] = {
    "demand": {".csv"},
    "queries": {".csv", ".xlsx"},
    "exogenous_variables": {".csv"},
    "rag_documents": {".pdf", ".md", ".markdown", ".txt", ".docx"},
}

UPLOAD_CATEGORY_LABELS: dict[UploadCategory, str] = {
    "demand": "Demand",
    "queries": "Queries",
    "exogenous_variables": "External Variables",
    "rag_documents": "RAG Documents",
}


def is_allowed_upload(filename: str, category: UploadCategory) -> bool:
    """Return whether a filename is accepted for a category."""
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_UPLOAD_SUFFIXES[category]


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename while preserving the extension."""
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", path.suffix.lower())
    return f"{stem or 'upload'}{suffix}"


def read_upload_manifest(manifest_path: Path = USER_UPLOAD_MANIFEST) -> list[dict]:
    """Read the upload manifest, returning an empty list when it is absent."""
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        records = json.load(f)
    return records if isinstance(records, list) else []


def write_upload_manifest(
    records: list[dict],
    manifest_path: Path = USER_UPLOAD_MANIFEST,
) -> None:
    """Write manifest records as pretty JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)


def save_upload_bytes(
    content: bytes,
    filename: str,
    category: UploadCategory,
    upload_root: Path = USER_UPLOADS_ROOT,
    manifest_path: Path = USER_UPLOAD_MANIFEST,
) -> dict:
    """Persist upload bytes and append a status record to the manifest."""
    if not is_allowed_upload(filename, category):
        suffix = Path(filename).suffix.lower() or "no extension"
        msg = f"{suffix} files are not accepted for {category} uploads."
        raise ValueError(msg)

    uploaded_at = datetime.now(tz=UTC).isoformat()
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    safe_name = sanitize_filename(filename)
    category_dir = upload_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{timestamp}_{safe_name}"
    saved_path = category_dir / saved_filename
    saved_path.write_bytes(content)

    record = {
        "filename": filename,
        "saved_filename": saved_filename,
        "saved_path": str(saved_path),
        "category": category,
        "extension": Path(filename).suffix.lower(),
        "size_bytes": len(content),
        "uploaded_at": uploaded_at,
        "status": "saved",
    }
    records = read_upload_manifest(manifest_path)
    records.append(record)
    write_upload_manifest(records, manifest_path)
    return record


def latest_uploads(
    records: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Return the newest manifest records by upload timestamp."""
    return sorted(
        records,
        key=lambda item: str(item.get("uploaded_at", "")),
        reverse=True,
    )[:limit]
