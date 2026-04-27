"""Data ingestion nodes: load, clean, validate, and aggregate raw CSV sources."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Raw CSV header names — used for contract validation before any renaming.
_DEMAND_RAW_COLUMNS = frozenset(
    ["SKU", "Year", "Month", "Month Name", "Date", "Monthly Demand", "Daily Demand"]
)

_DEMAND_RENAME = {
    "SKU": "sku",
    "Year": "year",
    "Month": "month",
    "Month Name": "month_name",
    "Date": "date",
    "Monthly Demand": "monthly_demand",
    "Daily Demand": "daily_demand",
}

# Validated after stripping whitespace; raw header has "surgifoam_limited " (trailing space).
_EXOGENOUS_STRIPPED_COLUMNS = frozenset(
    ["Date", "pfizer_limited", "surgifoam_limited", "rebate_target"]
)


def load_and_clean_demand(raw_demand: pd.DataFrame) -> pd.DataFrame:
    """Load and clean the raw daily demand CSV.

    Validates expected columns, renames to snake_case, parses dates,
    casts numeric types, removes duplicates, and sorts records.

    Args:
        raw_demand: Raw demand DataFrame loaded by the Kedro catalog.

    Returns:
        Cleaned demand DataFrame ready for primary dataset construction.
    """
    # Fail early on schema drift.
    missing = _DEMAND_RAW_COLUMNS - set(raw_demand.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in raw demand data: {sorted(missing)}"
        )

    df = raw_demand.rename(columns=_DEMAND_RENAME).copy()

    # Strip whitespace before type casting; SKU names often carry invisible spaces from Excel.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Source format is "M/D/YYYY" (e.g. "3/1/2023"); let pandas infer for flexibility.
    df["date"] = pd.to_datetime(df["date"])
    # Raw "Month" is "YYYY-MM", not a bare integer — parse then extract the month number.
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.month.astype(int)
    df["year"] = df["year"].astype(int)
    # errors="coerce" turns unparseable values into NaN instead of crashing.
    df["monthly_demand"] = pd.to_numeric(df["monthly_demand"], errors="coerce")
    df["daily_demand"] = pd.to_numeric(df["daily_demand"], errors="coerce")

    # NaNs after casting indicate corrupted source rows that would distort forecasts.
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if not null_cols.empty:
        logger.warning(
            "Null values detected in demand data:\n%s", null_cols.to_string()
        )

    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d duplicate rows from demand data.", n_removed)

    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    logger.info(
        "Demand data cleaned: %d rows, date range [%s, %s].",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def load_and_clean_exogenous(raw_exogenous: pd.DataFrame) -> pd.DataFrame:
    """Load and clean the raw exogenous variables CSV.

    Strips column name whitespace, validates structure, parses YYYY-MM dates,
    ensures numeric types, removes duplicates, and adds month_start_date.

    Args:
        raw_exogenous: Raw exogenous DataFrame loaded by the Kedro catalog.

    Returns:
        Cleaned exogenous DataFrame with a month_start_date column.
    """
    df = raw_exogenous.copy()
    # Strip before validation so "surgifoam_limited " matches the expected name.
    df.columns = [col.strip() for col in df.columns]

    missing = _EXOGENOUS_STRIPPED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in raw exogenous data: {sorted(missing)}"
        )

    df = df.rename(columns={"Date": "date"})
    df = df.dropna(how="all")  # remove trailing blank lines from CSV exports
    # Explicit format avoids ambiguous parsing of "YYYY-MM" strings.
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")

    for col in ["pfizer_limited", "surgifoam_limited", "rebate_target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if not null_cols.empty:
        logger.warning(
            "Null values detected in exogenous data:\n%s", null_cols.to_string()
        )

    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d duplicate rows from exogenous data.", n_removed)

    # Canonical month key: always the 1st of the month, aligns with demand joins.
    df["month_start_date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Exogenous data cleaned: %d rows, date range [%s, %s].",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def build_demand_daily(demand_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build the daily demand primary dataset.

    Args:
        demand_cleaned: Cleaned demand DataFrame from intermediate layer.

    Returns:
        Daily demand DataFrame at daily granularity.
    """
    cols = [
        "date",
        "sku",
        "daily_demand",
        "monthly_demand",  # kept for downstream cross-validation reference
        "year",
        "month",
        "month_name",
    ]
    return demand_cleaned[cols].copy()


def build_demand_weekly(demand_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build the weekly demand primary dataset.

    Aggregates daily_demand by SKU and ISO week using Monday as week start.

    Args:
        demand_cleaned: Cleaned demand DataFrame from intermediate layer.

    Returns:
        Weekly demand DataFrame with columns: week_start_date, sku, weekly_demand.
    """
    df = demand_cleaned[["date", "sku", "daily_demand"]].copy()
    # dayofweek is 0 (Mon) – 6 (Sun); subtracting it snaps every date to its Monday.
    df["week_start_date"] = df["date"] - pd.to_timedelta(
        df["date"].dt.dayofweek, unit="D"
    )

    weekly = df.groupby(["sku", "week_start_date"], as_index=False).agg(
        weekly_demand=("daily_demand", "sum")
    )
    weekly = weekly.sort_values(["sku", "week_start_date"]).reset_index(drop=True)
    return weekly[["week_start_date", "sku", "weekly_demand"]]


def build_demand_monthly(demand_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build the monthly demand primary dataset.

    Uses the reported monthly_demand as the authoritative monthly target.
    Logs inconsistencies between reported and summed daily demand, and warns
    if monthly_demand is not constant within a SKU-month.

    Args:
        demand_cleaned: Cleaned demand DataFrame from intermediate layer.

    Returns:
        Monthly demand DataFrame with columns: month_start_date, sku, monthly_demand.
    """
    df = demand_cleaned.copy()
    df["month_start_date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # --- Consistency validation (log-only, pipeline does not fail) ---
    daily_sum = (
        df.groupby(["sku", "month_start_date"])["daily_demand"]
        .sum()
        .reset_index()
        .rename(columns={"daily_demand": "daily_sum"})
    )

    # monthly_demand is expected to match the sum of daily_demand within each SKU-month.
    monthly_reported = (
        df.groupby(["sku", "month_start_date"])["monthly_demand"].first().reset_index()
    )

    # Merge on SKU and month to compare reported monthly_demand against summed daily_demand.
    validation = monthly_reported.merge(daily_sum, on=["sku", "month_start_date"])
    validation["abs_diff"] = (
        validation["monthly_demand"] - validation["daily_sum"]
    ).abs()

    # replace(0) guards against division by zero when reported demand is zero.
    validation["pct_diff"] = (
        validation["abs_diff"]
        / validation["monthly_demand"].replace(0, float("nan"))
        * 100
    )

    # Typical discrepancies are ±1–3 units — rounding artefacts in the source system.
    for _, row in validation[validation["abs_diff"] > 0].iterrows():
        logger.warning(
            "Monthly demand inconsistency — SKU=%s, month=%s: "
            "reported=%g, daily_sum=%g, abs_diff=%g, pct_diff=%.2f%%",
            row["sku"],
            row["month_start_date"].date(),
            row["monthly_demand"],
            row["daily_sum"],
            row["abs_diff"],
            row["pct_diff"],
        )

    # monthly_demand should be constant within a SKU-month; multiple values would
    # silently corrupt the monthly target if undetected.
    monthly_unique = df.groupby(["sku", "month_start_date"])["monthly_demand"].nunique()
    multi_valued = monthly_unique[monthly_unique > 1]
    if not multi_valued.empty:
        logger.warning(
            "Multiple monthly_demand values found for %d SKU-month combinations. "
            "Using first value after sort by date.",
            len(multi_valued),
        )

    # Sort before groupby so "first" is deterministic when duplicate values exist.
    monthly = (
        df.sort_values(["sku", "month_start_date", "date"])
        .groupby(["sku", "month_start_date"], as_index=False)
        .agg(monthly_demand=("monthly_demand", "first"))
    )
    monthly = monthly.sort_values(["sku", "month_start_date"]).reset_index(drop=True)
    return monthly[["month_start_date", "sku", "monthly_demand"]]


def build_exogenous_monthly(exogenous_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build the monthly exogenous primary dataset.

    Aligns dates to month start and keeps one row per month.

    Args:
        exogenous_cleaned: Cleaned exogenous DataFrame from intermediate layer.

    Returns:
        Monthly exogenous DataFrame with one row per month.
    """
    # Sort first so keep="first" on duplicates always picks the earliest row.
    df = exogenous_cleaned.copy().sort_values("month_start_date").reset_index(drop=True)

    # keep=False marks all rows of a duplicated month so we can log before dropping.
    duplicated = df["month_start_date"].duplicated(keep=False)
    if duplicated.any():
        dup_months = df.loc[duplicated, "month_start_date"].unique().tolist()
        logger.warning(
            "Duplicate months found in exogenous data (%d months). "
            "Keeping first sorted row per month: %s",
            len(dup_months),
            dup_months,
        )
        df = df.drop_duplicates(subset=["month_start_date"], keep="first")

    # Drop raw `date`; month_start_date is the canonical key for downstream joins.
    cols = ["month_start_date", "pfizer_limited", "surgifoam_limited", "rebate_target"]
    return df[cols].sort_values("month_start_date").reset_index(drop=True)
