"""Display formatting helpers for Streamlit pages.

All functions return a string suitable for metric widgets and dataframe display.
None, NaN, and invalid inputs are handled gracefully with a configurable fallback.
"""

import math
from typing import Any


def format_percentage(value: Any, decimals: int = 1) -> str:
    """Format a ratio (0.0–1.0) as a percentage string.

    Args:
        value: Numeric ratio (e.g. 0.123 → "12.3%").
        decimals: Number of decimal places in the output.

    Returns:
        Formatted percentage string, or "N/A" for None, NaN, or invalid inputs.
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        return f"{v:.{decimals}%}"
    except (TypeError, ValueError):
        return "N/A"


def format_metric(value: Any, decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric metric with optional suffix.

    Args:
        value: Numeric value to format.
        decimals: Number of decimal places.
        suffix: Optional suffix appended after the number (e.g. " units").

    Returns:
        Formatted string, or "N/A" for None, NaN, or invalid inputs.
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def format_optional(value: Any, default: str = "Not available") -> str:
    """Return a string representation of value, or a default for empty/None values.

    Args:
        value: Any value to render as text.
        default: Fallback string for None or empty string.

    Returns:
        String representation, or the default.
    """
    if value is None or value == "":
        return default
    return str(value)


def format_date(value: Any, default: str = "N/A") -> str:
    """Extract the date portion (YYYY-MM-DD) from a datetime string or object.

    Args:
        value: String or object with optional time component, or None/empty.
        default: Fallback for missing or invalid values.

    Returns:
        Date string "YYYY-MM-DD", or the default.
    """
    if not value:
        return default
    return str(value)[:10]
