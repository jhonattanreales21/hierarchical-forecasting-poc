"""Reusable validation and summary helpers for user-uploaded inputs.

These helpers keep upload validation logic out of the Streamlit layer so it can be
unit-tested and reused by pipelines. They operate on already-loaded DataFrames and
local file paths; they do not perform any file I/O beyond reading the document for
the document-summary helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from shared.rag import extract_document_text

Granularity = Literal["daily", "weekly", "monthly", "irregular", "unknown"]

# Median day-gap thresholds used to classify temporal granularity. The bands are
# wide enough to absorb calendar irregularities (e.g. 28–31 day months).
_DAILY_MAX_DAYS = 2.0
_WEEKLY_MIN_DAYS = 5.0
_WEEKLY_MAX_DAYS = 10.0
_MONTHLY_MIN_DAYS = 26.0
_MONTHLY_MAX_DAYS = 32.0


def find_date_column(df: pd.DataFrame) -> str | None:
    """Return the name of the first date-like column, if any.

    Performs a case-insensitive match for a column literally named ``date`` first,
    then falls back to the first column whose name contains ``date``.

    Args:
        df: DataFrame to inspect.

    Returns:
        The matching column name, or ``None`` when no date-like column exists.
    """
    lowered = {col.lower().strip(): col for col in df.columns}
    if "date" in lowered:
        return lowered["date"]
    for lower_name, original in lowered.items():
        if "date" in lower_name:
            return original
    return None


def detect_granularity(dates: pd.Series) -> Granularity:
    """Infer temporal granularity from a series of dates.

    Parses the values to datetimes, drops invalid/missing entries, sorts the unique
    timestamps, and classifies the median gap between consecutive dates into daily,
    weekly, or monthly bands. Values that do not fall in any band are ``irregular``;
    fewer than two valid dates yield ``unknown``.

    Args:
        dates: Series of date values (strings or datetimes). Both ``M/D/YYYY`` and
            ``YYYY-MM`` representations are handled by pandas inference.

    Returns:
        One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"irregular"``, or
        ``"unknown"``.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    unique_dates = pd.Series(parsed.dropna().unique()).sort_values()
    if len(unique_dates) < 2:
        return "unknown"

    median_gap_days = unique_dates.diff().dropna().dt.total_seconds().median() / 86400.0

    if median_gap_days <= _DAILY_MAX_DAYS:
        return "daily"
    if _WEEKLY_MIN_DAYS <= median_gap_days <= _WEEKLY_MAX_DAYS:
        return "weekly"
    if _MONTHLY_MIN_DAYS <= median_gap_days <= _MONTHLY_MAX_DAYS:
        return "monthly"
    return "irregular"


def summarize_csv(df: pd.DataFrame) -> dict:
    """Build a lightweight summary of an uploaded CSV DataFrame.

    Args:
        df: DataFrame parsed from an uploaded CSV file.

    Returns:
        Dictionary with ``n_rows``, ``n_columns``, ``columns``, ``date_column``,
        ``granularity``, ``date_min``, and ``date_max``. Date fields are ``None``
        when no date-like column is present.
    """
    date_column = find_date_column(df)
    granularity: Granularity = "unknown"
    date_min: str | None = None
    date_max: str | None = None

    if date_column is not None:
        parsed = pd.to_datetime(df[date_column], errors="coerce").dropna()
        granularity = detect_granularity(df[date_column])
        if not parsed.empty:
            date_min = parsed.min().date().isoformat()
            date_max = parsed.max().date().isoformat()

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "date_column": date_column,
        "granularity": granularity,
        "date_min": date_min,
        "date_max": date_max,
    }


def summarize_document(path: Path) -> dict:
    """Summarize a knowledge document for the assistant upload flow.

    Reports a type-appropriate size metric: page count for PDFs and word count for
    Word, Markdown, and text documents. Reuses ``extract_document_text`` so the set
    of supported types stays aligned with the RAG ingestion path.

    Args:
        path: Local path to the saved document.

    Returns:
        Dictionary with ``filename``, ``size_bytes``, ``kind`` (``"pdf"`` or
        ``"text"``), ``n_pages`` (``None`` for non-PDF), and ``n_words``.
    """
    records = extract_document_text(path)
    size_bytes = int(path.stat().st_size)
    is_pdf = path.suffix.lower() == ".pdf"
    n_words = sum(len(text.split()) for text, _ in records)

    return {
        "filename": path.name,
        "size_bytes": size_bytes,
        "kind": "pdf" if is_pdf else "text",
        "n_pages": len(records) if is_pdf else None,
        "n_words": int(n_words),
    }
