"""Pure helpers for the descriptive-analysis Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Granularity = Literal["daily", "weekly", "monthly"]

_DEMAND_CONTRACT: dict[Granularity, tuple[str, str]] = {
    "daily": ("date", "daily_demand"),
    "weekly": ("week_start_date", "weekly_demand"),
    "monthly": ("month_start_date", "monthly_demand"),
}


@dataclass(frozen=True)
class DemandSummary:
    """Compact descriptive demand metrics for a filtered frame."""

    total_demand: float
    average_demand: float
    peak_demand: float
    period_count: int


def normalize_demand_frame(df: pd.DataFrame, granularity: Granularity) -> pd.DataFrame:
    """Return a canonical demand frame for the requested temporal granularity."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "sku", "demand", "granularity"])
    if granularity not in _DEMAND_CONTRACT:
        msg = f"Unsupported granularity: {granularity}"
        raise ValueError(msg)

    date_col, demand_col = _DEMAND_CONTRACT[granularity]
    required = {date_col, "sku", demand_col}
    missing = required.difference(df.columns)
    if missing:
        msg = f"Demand frame missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    out = df[[date_col, "sku", demand_col]].copy()
    out = out.rename(columns={date_col: "date", demand_col: "demand"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sku"] = out["sku"].astype(str)
    out["demand"] = pd.to_numeric(out["demand"], errors="coerce")
    out["granularity"] = granularity
    out = out.dropna(subset=["date", "demand"])
    return out.sort_values(["sku", "date"]).reset_index(drop=True)


def normalize_exogenous_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly exogenous variables with a canonical ``date`` column."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    date_col = "month_start_date" if "month_start_date" in out.columns else "date"
    if date_col not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def filter_demand(
    df: pd.DataFrame,
    sku: str | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Filter canonical demand data by SKU and inclusive date range."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "sku", "demand", "granularity"])

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    if sku:
        mask &= df["sku"] == sku
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def filter_exogenous(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Filter canonical exogenous data by inclusive date range."""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def summarize_demand(df: pd.DataFrame) -> DemandSummary:
    """Compute simple descriptive metrics for filtered demand data."""
    if df is None or df.empty or "demand" not in df.columns:
        return DemandSummary(0.0, 0.0, 0.0, 0)

    demand = pd.to_numeric(df["demand"], errors="coerce").dropna()
    if demand.empty:
        return DemandSummary(0.0, 0.0, 0.0, 0)

    return DemandSummary(
        total_demand=float(demand.sum()),
        average_demand=float(demand.mean()),
        peak_demand=float(demand.max()),
        period_count=int(len(demand)),
    )


def exogenous_value_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric exogenous variable columns suitable for plotting."""
    if df is None or df.empty:
        return []
    excluded = {"date", "month_start_date"}
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols
