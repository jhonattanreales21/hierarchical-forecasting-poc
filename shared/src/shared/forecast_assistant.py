"""Forecast Assistant context, transformations, and LLM orchestration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MONTH_ALIASES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DATE_CANDIDATES = ("ds", "Date", "date", "Month", "month", "month_start_date")
DEMAND_CANDIDATES = ("y", "Demand", "demand", "Actual", "actual", "Monthly Demand")

SYSTEM_PROMPT = """You are an executive forecasting explainability assistant.
Only answer questions about forecast explainability, past demand behavior, model assumptions, and scenario variables.
If a question is outside that scope, respond in one short sentence saying you can only support forecast explainability questions.
Use only the provided forecast table, future scenario variables, model driver shares if available, historical demand/exogenous rows, model metadata, and retrieved RAG context.
Be concise: answer with no more than 5 bullets and no more than 120 words.
Write for executives and demand-planning managers, not data scientists.
For future months, prioritize exact scenario assumptions and forecast values.
For past spikes or drops, prioritize the historical demand/exogenous rows and retrieved RAG context.
Translate technical flags into business language.
Avoid raw variable syntax and internal feature names unless the user explicitly asks for technical details.
If the evidence is not available in the provided context, say that it is not available."""

SCOPE_TERMS = {
    "forecast",
    "demand",
    "month",
    "scenario",
    "assumption",
    "market share",
    "pfizer",
    "surgifoam",
    "rebate",
    "historical",
    "history",
    "spike",
    "drop",
    "increase",
    "decrease",
    "driver",
    "model",
    "actual",
}


@dataclass(frozen=True)
class DetectedPeriod:
    years: list[int]
    month: int | None


def is_in_scope(question: str) -> bool:
    """Return whether a question belongs to forecast explainability."""
    text = question.lower()
    return any(term in text for term in SCOPE_TERMS)


def detect_period(question: str) -> DetectedPeriod:
    """Detect years and month names/abbreviations from a user question."""
    years = sorted({int(y) for y in re.findall(r"\b(20[2-9]\d)\b", question)})
    month: int | None = None
    for token in re.findall(r"\b[a-zA-Z]{3,9}\b", question.lower()):
        clean = token.rstrip(".")
        if clean in MONTH_ALIASES:
            month = MONTH_ALIASES[clean]
            break
    return DetectedPeriod(years=years, month=month)


def identify_date_column(df: pd.DataFrame) -> str | None:
    """Find the likely monthly date column in a dataframe."""
    return next((col for col in DATE_CANDIDATES if col in df.columns), None)


def identify_demand_column(df: pd.DataFrame) -> str | None:
    """Find the likely demand/actual column in a dataframe."""
    return next((col for col in DEMAND_CANDIDATES if col in df.columns), None)


def normalize_monthly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized ``month`` column when possible."""
    if df.empty:
        return df.copy()
    out = df.copy()
    date_col = identify_date_column(out)
    if date_col:
        out["month"] = pd.to_datetime(out[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out


def select_historical_context(
    question: str,
    historical: pd.DataFrame,
    max_rows: int = 5,
) -> pd.DataFrame:
    """Select historical rows relevant to a question."""
    hist = normalize_monthly_frame(historical)
    if hist.empty or "month" not in hist.columns:
        return pd.DataFrame()

    detected = detect_period(question)
    demand_col = identify_demand_column(hist)

    if detected.years and detected.month:
        target = pd.Timestamp(detected.years[0], detected.month, 1)
        months = {target - pd.DateOffset(months=1), target, target + pd.DateOffset(months=1)}
        return hist[hist["month"].isin(months)].sort_values("month").head(max_rows)

    if detected.years:
        year_rows = hist[hist["month"].dt.year.isin(detected.years)]
        if not year_rows.empty and demand_col:
            return year_rows.sort_values(demand_col, ascending=False).head(max_rows).sort_values("month")
        return year_rows.sort_values("month").head(max_rows)

    if demand_col:
        return hist.sort_values(demand_col, ascending=False).head(max_rows).sort_values("month")
    return hist.sort_values("month").tail(max_rows)


def merge_historical_inputs(
    demand_df: pd.DataFrame,
    exogenous_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge historical demand and exogenous variables by month when both exist."""
    demand = normalize_monthly_frame(demand_df)
    if exogenous_df is None or exogenous_df.empty:
        return demand
    exog = normalize_monthly_frame(exogenous_df)
    if "month" not in demand.columns or "month" not in exog.columns:
        return demand
    return demand.merge(exog.drop_duplicates("month"), on="month", how="left", suffixes=("", "_scenario"))


def scenario_label(expected_market_share: Any) -> str:
    """Convert market-share assumptions into executive labels."""
    try:
        share = float(expected_market_share)
    except (TypeError, ValueError):
        return "not available"
    if share >= 1.0:
        return "severe competitor-stockout scenario"
    if share >= 0.8:
        return "high share-capture transition scenario"
    if share >= 0.65:
        return "elevated share-capture scenario"
    if share <= 0.45:
        return "constrained complementary-product scenario"
    return "normalized market-share assumption"


def _active(value: Any) -> str:
    try:
        return "active" if float(value) > 0 else "not active"
    except (TypeError, ValueError):
        return "not available"


def transform_scenario_rows(forecast: pd.DataFrame) -> pd.DataFrame:
    """Build business-readable future scenario rows from forecast output columns."""
    if forecast.empty:
        return pd.DataFrame()
    df = normalize_monthly_frame(forecast)
    out = pd.DataFrame()
    out["month"] = df["month"].dt.strftime("%Y-%m") if "month" in df else ""
    share = df["expected_market_share"] if "expected_market_share" in df else pd.Series([None] * len(df))
    out["expected market share"] = share.apply(_format_share)
    out["market-share scenario label"] = share.apply(scenario_label)
    out["competitor supply constraint effect"] = _first_available_status(
        df, ("pfizer_limited", "pfizer_limited_lag_1", "pfizer_limited_lag1")
    )
    out["complementary product constraint"] = _first_available_status(df, ("surgifoam_limited", "surgifoam_limited_lag_1"))
    out["rebate purchase acceleration"] = _first_available_status(df, ("rebate_target", "rebate_target_lag_1", "rebate_target_lag1"))
    if "yhat" in df:
        out["forecast demand units"] = df["yhat"].round(0).astype("Int64")
    return out


def rename_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common internal fields for assistant context."""
    names = {
        "month": "month",
        "ds": "month",
        "y": "actual demand units",
        "Monthly Demand": "actual demand units",
        "pfizer_limited": "competitor supply constraint",
        "pfizer_limited_lag_1": "lagged competitor supply constraint",
        "pfizer_limited_lag1": "lagged competitor supply constraint",
        "surgifoam_limited": "complementary product limitation",
        "rebate_target": "rebate purchase acceleration",
        "rebate_target_lag_1": "rebate payback lag",
        "rebate_payback_lag1": "rebate payback lag",
        "expected_market_share": "expected market share",
        "yhat": "forecast demand units",
        "yhat_lower": "forecast lower bound",
        "yhat_upper": "forecast upper bound",
    }
    out = df.copy()
    out = out.rename(columns={col: names[col] for col in out.columns if col in names})
    out = _coalesce_duplicate_columns(out)
    if "month" in out:
        out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.strftime("%Y-%m")
    if "expected market share" in out:
        out["expected market share"] = out["expected market share"].apply(_format_share)
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(2)
    return out


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate column names by taking the first non-null value per row."""
    if not df.columns.has_duplicates:
        return df
    merged: dict[str, pd.Series] = {}
    ordered_names = list(dict.fromkeys(df.columns.tolist()))
    for col in ordered_names:
        values = df.loc[:, df.columns == col]
        if isinstance(values, pd.Series):
            merged[col] = values
        else:
            merged[col] = values.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(merged, index=df.index)


def build_assistant_context(
    question: str,
    forecast: pd.DataFrame,
    historical: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    rag_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the evidence bundle sent to the LLM."""
    hist_rows = select_historical_context(question, historical)
    forecast_rows = rename_business_columns(forecast.head(24))
    scenario_rows = transform_scenario_rows(forecast.head(24))
    return {
        "user_question": question,
        "forecast_output_table": forecast_rows.to_dict(orient="records"),
        "future_scenario_variables": scenario_rows.to_dict(orient="records"),
        "historical_demand_exogenous_rows": rename_business_columns(hist_rows).to_dict(orient="records"),
        "retrieved_rag_context": rag_chunks or [],
        "model_metadata_and_assumptions": metadata or {},
    }


def answer_question(
    question: str,
    context: dict[str, Any],
    model: str = "gpt-5-mini",
) -> str:
    """Generate a concise answer with the OpenAI Responses API."""
    if not is_in_scope(question):
        return "I can only support forecast explainability questions."
    api_key = os.getenv("OPENAI_API_KEY") or _read_dotenv_value("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI is not configured. Set OPENAI_API_KEY to enable assistant answers."

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=f"Evidence bundle:\n{context}",
    )
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    return "The assistant did not return an answer."


def _read_dotenv_value(key: str) -> str | None:
    """Read a simple KEY=VALUE entry from a local .env file if present."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _format_share(value: Any) -> str:
    try:
        share = float(value)
    except (TypeError, ValueError):
        return "not available"
    if share <= 1.5:
        share *= 100
    return f"{share:.0f}%"


def _first_available_status(df: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    statuses: list[str] = []
    for _, row in df.iterrows():
        value = None
        for col in cols:
            if col in df.columns and pd.notna(row.get(col)):
                value = row.get(col)
                break
        statuses.append(_active(value))
    return statuses
