"""Monthly model-input preparation nodes.

Generic monthly layer (build_monthly_modeling_data → split_monthly_modeling_data →
build_monthly_split_metadata) produces model-family-agnostic datasets with canonical
column names (month_start_date, monthly_demand).

Prophet compatibility adapter (adapt_monthly_data_for_prophet) renames those columns
to ds / y and feeds the existing build_monthly_prophet_future_regressors and
build_monthly_prophet_split_metadata functions unchanged.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import holidays
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)

_WEEKDAY_TO_NUMBER = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}

_CALENDAR_FEATURE_COLUMNS = [
    "business_days",
    "total_tuesdays",
    "total_thursdays",
    "working_tuesdays",
    "working_thursdays",
    "has_5_working_tuesdays",
    "has_5_working_thursdays",
    "tuesday_holidays",
    "thursday_holidays",
    "total_holidays",
]


# ── Parameter extraction helpers ──────────────────────────────────────────────

def _get_monthly_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the generic monthly configuration block."""
    if "monthly" in parameters:
        return dict(parameters["monthly"])
    return dict(parameters)


def _get_monthly_prophet_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Monthly Prophet configuration block."""
    if "monthly_prophet" in parameters:
        return dict(parameters["monthly_prophet"])
    return dict(parameters)


def _get_monthly_sarimax_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Monthly SARIMAX configuration block."""
    if "monthly_sarimax" in parameters:
        return dict(parameters["monthly_sarimax"])
    return dict(parameters)


# ── Shared validation and utility helpers ─────────────────────────────────────

def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when a required schema column is missing."""
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )


def _ensure_datetime_column(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Convert a column to datetime and fail clearly when parsing fails."""
    normalized_df = df.copy()
    parsed_dates = pd.to_datetime(normalized_df[column], errors="coerce")
    invalid_rows = normalized_df.loc[parsed_dates.isna(), [column]]
    if not invalid_rows.empty:
        raise ValueError(
            f"Column {column!r} in {dataset_name} contains invalid dates:\n"
            f"{invalid_rows.head(10).to_string(index=False)}"
        )

    # Snap any intra-month date to the first of the month for consistent joins.
    normalized_df[column] = parsed_dates.dt.to_period("M").dt.to_timestamp()
    return normalized_df


def _validate_unique_rows(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Ensure the dataset is unique on the expected business key."""
    duplicated = df.duplicated(subset=list(key_columns), keep=False)
    if duplicated.any():
        duplicate_rows = df.loc[duplicated, list(key_columns)].sort_values(
            list(key_columns)
        )
        raise ValueError(
            f"{dataset_name} must be unique by {list(key_columns)}. "
            f"Found duplicated keys:\n{duplicate_rows.to_string(index=False)}"
        )


def _validate_datetime_and_numeric(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
) -> None:
    """Validate that the date column is datetime and the target column is numeric."""
    if not is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"{date_column!r} must be a datetime column.")
    if not is_numeric_dtype(df[target_column]):
        raise ValueError(f"{target_column!r} must be numeric.")


def _summarize_date_range(df: pd.DataFrame, date_column: str) -> dict[str, Any]:
    """Return a compact date-range summary for logs and metadata."""
    if df.empty:
        return {"start_date": None, "end_date": None, "rows": 0}

    return {
        "start_date": pd.Timestamp(df[date_column].min()).date().isoformat(),
        "end_date": pd.Timestamp(df[date_column].max()).date().isoformat(),
        "rows": int(len(df)),
    }


def _validate_no_nulls(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Raise an error when required modeling columns still contain nulls."""
    null_counts = df[list(required_columns)].isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    if not remaining_nulls.empty:
        raise ValueError(
            f"Null values remain in required columns for {dataset_name}:\n"
            f"{remaining_nulls.to_string()}"
        )


def _get_active_regressors(
    df: pd.DataFrame,
    active_regressors: Sequence[str],
    dataset_name: str,
) -> list[str]:
    """Validate and return the configured active regressor columns."""
    regressors = list(active_regressors)
    _validate_required_columns(df, regressors, dataset_name)
    return regressors


def _drop_null_modeling_rows(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    active_regressors: Sequence[str],
    missing_value_params: Mapping[str, Any],
    dataset_name: str = "monthly_modeling_data",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply configured null handling: drop rows with null target or null active regressors."""
    cleaned_df = df.copy()
    dropped_rows = {
        "null_target": 0,
        "null_active_regressors": 0,
        "null_active_regressor_columns": {},
    }

    null_target_mask = cleaned_df[target_column].isna()
    if bool(missing_value_params["drop_rows_with_null_target"]):
        dropped_rows["null_target"] = int(null_target_mask.sum())
        cleaned_df = cleaned_df.loc[~null_target_mask].copy()
    elif null_target_mask.any():
        raise ValueError(
            f"Found null values in target column {target_column!r} while "
            "drop_rows_with_null_target is disabled."
        )

    if active_regressors:
        regressor_null_counts = cleaned_df[list(active_regressors)].isnull().sum()
        regressor_null_counts = regressor_null_counts[regressor_null_counts > 0]
        if bool(missing_value_params["drop_rows_with_null_active_regressors"]):
            null_regressor_mask = (
                cleaned_df[list(active_regressors)].isnull().any(axis=1)
            )
            dropped_rows["null_active_regressors"] = int(null_regressor_mask.sum())
            dropped_rows["null_active_regressor_columns"] = {
                column: int(count) for column, count in regressor_null_counts.items()
            }
            # Early lagged features can be null by design; drop leading months instead of imputing.
            cleaned_df = cleaned_df.loc[~null_regressor_mask].copy()
        elif not regressor_null_counts.empty:
            raise ValueError(
                "Found null values in active regressor columns while "
                "drop_rows_with_null_active_regressors is disabled:\n"
                f"{regressor_null_counts.to_string()}"
            )

    cleaned_df = cleaned_df.sort_values(["sku", date_column]).reset_index(drop=True)
    _validate_no_nulls(
        cleaned_df,
        [date_column, target_column, *active_regressors],
        dataset_name,
    )
    return cleaned_df, dropped_rows


def _parse_month_start(date_value: Any, field_name: str) -> pd.Timestamp:
    """Parse and normalize a configured date boundary to month start."""
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError(f"Invalid {field_name!r}: {date_value!r}")
    return pd.Timestamp(parsed_date).to_period("M").to_timestamp()


def _split_by_dates(
    df: pd.DataFrame,
    date_column: str,
    split_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split historical data using explicit date boundaries."""
    train_end_date = _parse_month_start(
        split_params["train_end_date"], "train_end_date"
    )
    validation_end_date = _parse_month_start(
        split_params["validation_end_date"], "validation_end_date"
    )
    test_end_date = _parse_month_start(split_params["test_end_date"], "test_end_date")

    if not (train_end_date < validation_end_date < test_end_date):
        raise ValueError(
            "Date split boundaries must satisfy "
            "train_end_date < validation_end_date < test_end_date."
        )

    train = df.loc[df[date_column] <= train_end_date].copy()
    validation = df.loc[
        (df[date_column] > train_end_date) & (df[date_column] <= validation_end_date)
    ].copy()
    test = df.loc[
        (df[date_column] > validation_end_date) & (df[date_column] <= test_end_date)
    ].copy()
    return train, validation, test


def _split_by_month_counts(
    df: pd.DataFrame,
    date_column: str,
    split_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split historical data using trailing validation and test month counts."""
    validation_months = int(split_params["validation_months"])
    test_months = int(split_params["test_months"])
    if validation_months <= 0 or test_months <= 0:
        raise ValueError("validation_months and test_months must be positive integers.")

    # Build an ordered list of unique month timestamps to use as positional index for slicing.
    unique_months = pd.Series(
        pd.DatetimeIndex(df[date_column].drop_duplicates().sort_values())
    ).reset_index(drop=True)
    required_months = validation_months + test_months
    if len(unique_months) <= required_months:
        raise ValueError(
            "Not enough historical months to create non-empty train, validation, "
            f"and test splits. Found {len(unique_months)} unique months and need "
            f"more than {required_months}."
        )

    # Slice from the tail: last N months = test, preceding M months = validation, rest = train.
    test_month_index = unique_months.iloc[-test_months:]
    validation_month_index = unique_months.iloc[
        -(validation_months + test_months) : -test_months
    ]
    train_month_index = unique_months.iloc[: -(validation_months + test_months)]

    train = df.loc[df[date_column].isin(train_month_index)].copy()
    validation = df.loc[df[date_column].isin(validation_month_index)].copy()
    test = df.loc[df[date_column].isin(test_month_index)].copy()
    return train, validation, test


def _validate_split_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    date_column: str,
) -> None:
    """Validate split emptiness, overlap, and chronological ordering."""
    split_frames = {"train": train, "validation": validation, "test": test}
    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"{split_name.title()} split is empty.")

    if not (
        train[date_column].max() < validation[date_column].min()
        and validation[date_column].max() < test[date_column].min()
    ):
        raise ValueError(
            "Chronological splits overlap or are out of order. Expected "
            "train < validation < test."
        )


def _parse_weekmask(weekmask: str) -> set[int]:
    """Translate a NumPy-style weekmask into weekday numbers."""
    weekday_names = weekmask.split()
    invalid_days = sorted(set(weekday_names) - set(_WEEKDAY_TO_NUMBER))
    if invalid_days:
        raise ValueError(
            f"Invalid weekday names in weekmask {weekmask!r}: {invalid_days}"
        )

    return {
        day_number
        for day, day_number in _WEEKDAY_TO_NUMBER.items()
        if day in weekday_names
    }


def _get_country_holidays(
    country_code: str,
    years: Sequence[int],
    observed: bool = True,
) -> pd.DatetimeIndex:
    """Return observed holiday dates for the configured country."""
    holiday_calendar = holidays.country_holidays(
        country=country_code,
        years=sorted(set(years)),
        observed=observed,
    )
    holiday_dates = [
        pd.Timestamp(holiday_date).normalize()
        for holiday_date in holiday_calendar.keys()
    ]
    return pd.DatetimeIndex(sorted(holiday_dates))


def _count_monthly_calendar_features(
    month_start_date: pd.Timestamp,
    holiday_dates: pd.DatetimeIndex,
    business_weekdays: set[int],
    min_working_weekday_count: int,
) -> dict[str, int | pd.Timestamp]:
    """Count deterministic calendar features for a single month."""
    month_start = pd.Timestamp(month_start_date).normalize()
    month_end = month_start + pd.offsets.MonthEnd(1)
    month_days = pd.date_range(start=month_start, end=month_end, freq="D")

    is_holiday = month_days.isin(holiday_dates)
    is_business_weekday = month_days.dayofweek.isin(sorted(business_weekdays))
    is_tuesday = month_days.dayofweek == _WEEKDAY_TO_NUMBER["Tue"]
    is_thursday = month_days.dayofweek == _WEEKDAY_TO_NUMBER["Thu"]

    working_tuesdays = int((is_tuesday & ~is_holiday).sum())
    working_thursdays = int((is_thursday & ~is_holiday).sum())

    return {
        "month_start_date": month_start,
        "business_days": int((is_business_weekday & ~is_holiday).sum()),
        "total_tuesdays": int(is_tuesday.sum()),
        "total_thursdays": int(is_thursday.sum()),
        "working_tuesdays": working_tuesdays,
        "working_thursdays": working_thursdays,
        "has_5_working_tuesdays": int(working_tuesdays >= min_working_weekday_count),
        "has_5_working_thursdays": int(working_thursdays >= min_working_weekday_count),
        "tuesday_holidays": int((is_tuesday & is_holiday).sum()),
        "thursday_holidays": int((is_thursday & is_holiday).sum()),
        "total_holidays": int(is_holiday.sum()),
    }


def _generate_future_months(last_ds: pd.Timestamp, horizon_months: int) -> pd.DataFrame:
    """Generate a month-start index that begins after the last historical month."""
    future_index = pd.date_range(
        start=pd.Timestamp(last_ds) + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq="MS",
    )
    return pd.DataFrame({"month_start_date": future_index})


def _build_future_calendar_features(
    future_months: pd.DataFrame,
    monthly_calendar_features: pd.DataFrame,
    calendar_parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Reuse existing calendar features and generate missing future months."""
    calendar_df = _ensure_datetime_column(
        monthly_calendar_features,
        "month_start_date",
        "monthly_calendar_features",
    )
    _validate_required_columns(
        calendar_df,
        ["month_start_date", *_CALENDAR_FEATURE_COLUMNS],
        "monthly_calendar_features",
    )
    _validate_unique_rows(
        calendar_df,
        ["month_start_date"],
        "monthly_calendar_features",
    )

    future_dates = pd.DatetimeIndex(future_months["month_start_date"])
    existing_future_calendar = calendar_df.loc[
        calendar_df["month_start_date"].isin(future_dates),
        ["month_start_date", *_CALENDAR_FEATURE_COLUMNS],
    ].copy()

    missing_calendar_dates = sorted(
        set(future_dates) - set(existing_future_calendar["month_start_date"])
    )
    if missing_calendar_dates:
        logger.info(
            "Generating deterministic calendar features for %d future months not "
            "present in monthly_calendar_features.",
            len(missing_calendar_dates),
        )
        holiday_dates = _get_country_holidays(
            country_code=calendar_parameters["calendar_features"]["country_holidays"],
            years=future_dates.year.unique().tolist(),
            observed=calendar_parameters["calendar_features"]["observed_holidays"],
        )
        business_weekdays = _parse_weekmask(
            calendar_parameters["calendar_features"]["weekmask"]
        )
        min_working_count: int = int(
            calendar_parameters["calendar_features"].get(
                "min_working_weekday_count_for_flag", 5
            )
        )
        generated_calendar = pd.DataFrame(
            [
                _count_monthly_calendar_features(
                    month_start, holiday_dates, business_weekdays, min_working_count
                )
                for month_start in missing_calendar_dates
            ]
        )
        existing_future_calendar = pd.concat(
            [existing_future_calendar, generated_calendar],
            ignore_index=True,
        )

    return existing_future_calendar.sort_values("month_start_date").reset_index(
        drop=True
    )


def _build_future_exogenous_features(
    future_months: pd.DataFrame,
    monthly_exogenous_features: pd.DataFrame,
    required_exogenous_columns: Sequence[str],
) -> pd.DataFrame:
    """Select future exogenous rows and fail when required months are unavailable."""
    exogenous_df = _ensure_datetime_column(
        monthly_exogenous_features,
        "month_start_date",
        "monthly_exogenous_features",
    )
    _validate_required_columns(
        exogenous_df,
        ["month_start_date", *required_exogenous_columns],
        "monthly_exogenous_features",
    )
    _validate_unique_rows(
        exogenous_df,
        ["month_start_date"],
        "monthly_exogenous_features",
    )

    future_exogenous = future_months.merge(
        exogenous_df[["month_start_date", *required_exogenous_columns]],
        on="month_start_date",
        how="left",
        validate="one_to_one",
    )
    missing_exogenous_months = future_exogenous.loc[
        future_exogenous[required_exogenous_columns].isnull().any(axis=1),
        "month_start_date",
    ]
    if not missing_exogenous_months.empty:
        formatted_months = [
            pd.Timestamp(month).date().isoformat()
            for month in pd.DatetimeIndex(missing_exogenous_months.unique())
        ]
        raise ValueError(
            "Missing required future exogenous values for months: "
            f"{formatted_months}. Future exogenous regressors must be available "
            "before saving Prophet future datasets."
        )

    return future_exogenous.sort_values("month_start_date").reset_index(drop=True)


# ── Generic monthly layer ─────────────────────────────────────────────────────

def build_monthly_modeling_data(
    monthly_prophet_features: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the generic monthly modeling table with canonical (non-Prophet) column names.

    Selects the configured date, target, SKU, and active regressor columns; validates
    schema and uniqueness; and drops rows with null targets or null active regressors
    according to the ``missing_values`` parameter block.  The output retains the
    original generic column names (``month_start_date``, ``monthly_demand``) — Prophet-
    specific renaming is handled downstream by ``adapt_monthly_data_for_prophet``.

    Args:
        monthly_prophet_features: Feature-engineered monthly DataFrame from the feature
            engineering pipeline.
        parameters: Pipeline parameters. Reads the ``monthly`` block.

    Returns:
        Tuple of:
            - modeling_df: Cleaned DataFrame with columns
              ``[month_start_date, monthly_demand, sku, *active_regressors]`` sorted by
              SKU and date.
            - preparation_metadata: Dict with granularity, column names, active
              regressors, dropped-row counts, and date range summary.
    """
    monthly_params = _get_monthly_params(parameters)
    date_column = monthly_params["date_column"]
    target_column = monthly_params["target_column"]
    sku_column = monthly_params["sku_column"]
    active_regressors = _get_active_regressors(
        monthly_prophet_features,
        monthly_params["active_regressors"],
        "monthly_prophet_features",
    )

    _validate_required_columns(
        monthly_prophet_features,
        [date_column, target_column, sku_column, *active_regressors],
        "monthly_prophet_features",
    )

    feature_df = _ensure_datetime_column(
        monthly_prophet_features,
        date_column,
        "monthly_prophet_features",
    )
    _validate_unique_rows(
        feature_df, [sku_column, date_column], "monthly_prophet_features"
    )

    logger.info(
        "Building generic monthly modeling data from shape=%s.", feature_df.shape
    )
    logger.info("Selected active regressors: %s", active_regressors)

    modeling_df = (
        feature_df[[date_column, target_column, sku_column, *active_regressors]]
        .sort_values([sku_column, date_column])
        .reset_index(drop=True)
    )
    _validate_datetime_and_numeric(modeling_df, date_column, target_column)

    modeling_df, dropped_rows = _drop_null_modeling_rows(
        modeling_df,
        date_column=date_column,
        target_column=target_column,
        active_regressors=active_regressors,
        missing_value_params=monthly_params["missing_values"],
        dataset_name="monthly_modeling_data",
    )

    logger.info(
        "Final monthly_modeling_data shape=%s, range=[%s, %s].",
        modeling_df.shape,
        modeling_df[date_column].min().date(),
        modeling_df[date_column].max().date(),
    )

    preparation_metadata: dict[str, Any] = {
        "granularity": "monthly",
        "date_column": date_column,
        "target_column": target_column,
        "sku_column": sku_column,
        "active_regressors": active_regressors,
        "dropped_rows": {
            "null_target": dropped_rows["null_target"],
            "null_active_regressors": dropped_rows["null_active_regressors"],
        },
        "modeling_data": _summarize_date_range(modeling_df, date_column),
    }
    return modeling_df, preparation_metadata


def split_monthly_modeling_data(
    monthly_modeling_data: pd.DataFrame,
    preparation_metadata: dict[str, Any],
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition the generic monthly modeling table into temporal splits.

    Supports two split modes controlled by ``parameters["monthly"]["split"]["mode"]``:

    - ``"date"`` — explicit cutoff dates.
    - ``"months"`` — trailing month counts.

    ``full_train`` concatenates all three partitions and is intended for the final
    retrain before production inference.

    Args:
        monthly_modeling_data: Generic modeling DataFrame from ``build_monthly_modeling_data``.
        preparation_metadata: Metadata dict produced by the preparation node.
        parameters: Pipeline parameters with a ``monthly.split`` block.

    Returns:
        Tuple of five items:
            - monthly_train: Historical rows reserved for model fitting.
            - monthly_validation: Held-out rows for hyperparameter selection.
            - monthly_test: Held-out rows for final evaluation.
            - monthly_full_train: All rows combined, sorted by SKU and date.
            - updated_metadata: Metadata dict extended with split-mode and date-range summaries.
    """
    monthly_params = _get_monthly_params(parameters)
    date_column = monthly_params["date_column"]
    sku_column = monthly_params["sku_column"]
    split_params = monthly_params["split"]
    split_mode = split_params["mode"]

    logger.info(
        "Splitting monthly modeling data shape=%s using split mode=%s.",
        monthly_modeling_data.shape,
        split_mode,
    )

    if split_mode == "date":
        train, validation, test = _split_by_dates(
            monthly_modeling_data, date_column, split_params
        )
    elif split_mode == "months":
        train, validation, test = _split_by_month_counts(
            monthly_modeling_data, date_column, split_params
        )
    else:
        raise ValueError(
            f"Unsupported split mode {split_mode!r}. Expected 'date' or 'months'."
        )

    _validate_split_frames(train, validation, test, date_column)
    full_train = (
        pd.concat([train, validation, test], ignore_index=True)
        .sort_values([sku_column, date_column])
        .reset_index(drop=True)
    )

    logger.info("Train split: %s", _summarize_date_range(train, date_column))
    logger.info("Validation split: %s", _summarize_date_range(validation, date_column))
    logger.info("Test split: %s", _summarize_date_range(test, date_column))

    updated_metadata = dict(preparation_metadata)
    updated_metadata["split_mode"] = split_mode
    updated_metadata["train"] = _summarize_date_range(train, date_column)
    updated_metadata["validation"] = _summarize_date_range(validation, date_column)
    updated_metadata["test"] = _summarize_date_range(test, date_column)
    updated_metadata["full_train"] = _summarize_date_range(full_train, date_column)

    return train, validation, test, full_train, updated_metadata


def build_monthly_split_metadata(
    monthly_train: pd.DataFrame,
    monthly_validation: pd.DataFrame,
    monthly_test: pd.DataFrame,
    monthly_full_train: pd.DataFrame,
    preparation_metadata: dict[str, Any],
    parameters: dict,
) -> dict[str, Any]:
    """Compile a generic monthly split metadata artifact for catalog persistence.

    Args:
        monthly_train: Train split DataFrame.
        monthly_validation: Validation split DataFrame.
        monthly_test: Test split DataFrame.
        monthly_full_train: Full-train split DataFrame.
        preparation_metadata: Extended metadata dict from ``split_monthly_modeling_data``.
        parameters: Pipeline parameters with a ``monthly`` block.

    Returns:
        Dict with granularity, column names, split boundaries, row counts, active
        features, and dropped-row counts.
    """
    monthly_params = _get_monthly_params(parameters)
    date_column = monthly_params["date_column"]

    metadata: dict[str, Any] = {
        "granularity": preparation_metadata["granularity"],
        "date_column": preparation_metadata["date_column"],
        "target_column": preparation_metadata["target_column"],
        "sku_column": preparation_metadata["sku_column"],
        "split_mode": preparation_metadata["split_mode"],
        "active_features": preparation_metadata["active_regressors"],
        "train": _summarize_date_range(monthly_train, date_column),
        "validation": _summarize_date_range(monthly_validation, date_column),
        "test": _summarize_date_range(monthly_test, date_column),
        "full_train": _summarize_date_range(monthly_full_train, date_column),
        "row_counts": {
            "train": len(monthly_train),
            "validation": len(monthly_validation),
            "test": len(monthly_test),
            "full_train": len(monthly_full_train),
        },
        "dropped_rows": preparation_metadata["dropped_rows"],
        "created_by": "model_input_preparation",
    }
    logger.info("Built monthly_split_metadata: %s", metadata)
    return metadata


# ── Prophet compatibility adapter ─────────────────────────────────────────────

def adapt_monthly_data_for_prophet(
    monthly_modeling_data: pd.DataFrame,
    monthly_train: pd.DataFrame,
    monthly_validation: pd.DataFrame,
    monthly_test: pd.DataFrame,
    monthly_full_train: pd.DataFrame,
    monthly_split_metadata: dict[str, Any],
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rename generic monthly columns to Prophet's ds / y convention.

    Consumes the generic monthly datasets and produces Prophet-compatible equivalents
    by renaming ``month_start_date`` → ``ds`` and ``monthly_demand`` → ``y``.  All
    other columns (SKU, active regressors) are preserved without modification.

    Args:
        monthly_modeling_data: Generic modeling DataFrame.
        monthly_train: Generic train split.
        monthly_validation: Generic validation split.
        monthly_test: Generic test split.
        monthly_full_train: Generic full-train split.
        monthly_split_metadata: Generic split metadata dict.
        parameters: Pipeline parameters. Reads ``monthly`` (source column names) and
            ``monthly_prophet`` (Prophet column names and active_regressors).

    Returns:
        Tuple of six items:
            - monthly_prophet_modeling_data: Prophet-format modeling DataFrame.
            - monthly_prophet_train: Prophet-format train split.
            - monthly_prophet_validation: Prophet-format validation split.
            - monthly_prophet_test: Prophet-format test split.
            - monthly_prophet_full_train: Prophet-format full-train split.
            - prophet_adapter_metadata: Internal metadata dict for
              ``build_monthly_prophet_split_metadata``.
    """
    monthly_params = _get_monthly_params(parameters)
    prophet_params = _get_monthly_prophet_params(parameters)

    date_column = monthly_params["date_column"]
    target_column = monthly_params["target_column"]
    prophet_date_column = prophet_params["prophet_date_column"]
    prophet_target_column = prophet_params["prophet_target_column"]

    rename_map = {date_column: prophet_date_column, target_column: prophet_target_column}

    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=rename_map).copy()

    prophet_modeling_data = _rename(monthly_modeling_data)
    prophet_train = _rename(monthly_train)
    prophet_validation = _rename(monthly_validation)
    prophet_test = _rename(monthly_test)
    prophet_full_train = _rename(monthly_full_train)

    _validate_datetime_and_numeric(
        prophet_modeling_data, prophet_date_column, prophet_target_column
    )

    logger.info(
        "Adapted generic monthly data to Prophet format: %s → %s, %s → %s.",
        date_column, prophet_date_column,
        target_column, prophet_target_column,
    )

    # Internal metadata consumed by build_monthly_prophet_split_metadata.
    prophet_adapter_metadata: dict[str, Any] = {
        "model_family": "prophet",
        "granularity": monthly_split_metadata["granularity"],
        "split_mode": monthly_split_metadata["split_mode"],
        "active_regressors": list(prophet_params["active_regressors"]),
        "dropped_rows": monthly_split_metadata["dropped_rows"],
        "modeling_data": _summarize_date_range(
            prophet_modeling_data, prophet_date_column
        ),
    }

    return (
        prophet_modeling_data,
        prophet_train,
        prophet_validation,
        prophet_test,
        prophet_full_train,
        prophet_adapter_metadata,
    )


# ── SARIMAX adapter ───────────────────────────────────────────────────────────

def _validate_monthly_frequency(
    date_series: pd.Series,
    split_name: str,
    require_regular_frequency: bool,
) -> list[str]:
    """Check for missing monthly periods and return a list of ISO date strings."""
    if len(date_series) < 2:
        return []

    sorted_dates = pd.DatetimeIndex(date_series.sort_values())
    expected_range = pd.date_range(
        start=sorted_dates.min(),
        end=sorted_dates.max(),
        freq="MS",
    )
    missing_periods = sorted(set(expected_range) - set(sorted_dates))
    missing_strs = [pd.Timestamp(p).date().isoformat() for p in missing_periods]

    if missing_strs and require_regular_frequency:
        raise ValueError(
            f"Missing monthly periods in {split_name!r} split while "
            f"require_regular_frequency is enabled: {missing_strs}"
        )
    if missing_strs:
        logger.warning(
            "Missing monthly periods in %r split: %s", split_name, missing_strs
        )
    return missing_strs


def _prepare_sarimax_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    exogenous_columns: list[str],
    sarimax_params: Mapping[str, Any],
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and prepare one split DataFrame for SARIMAX consumption.

    Returns a tabular DataFrame [date_column, target_column, *exogenous_columns]
    and a split-level info dict with row counts, date range, and diagnostics.
    """
    prepared = df.copy().sort_values(date_column).reset_index(drop=True)

    if not is_datetime64_any_dtype(prepared[date_column]):
        prepared[date_column] = pd.to_datetime(prepared[date_column])

    # Duplicate dates in a single-SKU monthly series indicate upstream data quality issues.
    duplicate_mask = prepared[date_column].duplicated(keep=False)
    if duplicate_mask.any():
        duped = prepared.loc[duplicate_mask, [date_column]].drop_duplicates()
        duped_dates = duped[date_column].dt.date.tolist()
        raise ValueError(
            f"Duplicate dates found in {split_name!r} split after SARIMAX adapter: "
            f"{duped_dates}. The SARIMAX adapter expects a unique monthly date index."
        )

    missing_periods = _validate_monthly_frequency(
        prepared[date_column],
        split_name,
        bool(sarimax_params.get("require_regular_frequency", True)),
    )

    # Null target handling
    null_target_mask = prepared[target_column].isna()
    null_target_rows_dropped = 0
    if bool(sarimax_params.get("drop_rows_with_null_target", True)):
        null_target_rows_dropped = int(null_target_mask.sum())
        prepared = prepared.loc[~null_target_mask].copy()
    elif null_target_mask.any():
        raise ValueError(
            f"Null values in target column {target_column!r} in {split_name!r} split "
            "while drop_rows_with_null_target is disabled."
        )

    # Null exogenous handling
    if exogenous_columns:
        null_exog_mask = prepared[exogenous_columns].isna().any(axis=1)
        if bool(sarimax_params.get("drop_rows_with_null_exog", False)):
            prepared = prepared.loc[~null_exog_mask].copy()
        elif null_exog_mask.any() and not bool(sarimax_params.get("impute_exog", False)):
            null_exog_counts = (
                prepared[exogenous_columns].isna().sum()
            )
            raise ValueError(
                f"Null values in exogenous columns in {split_name!r} split. "
                "Set drop_rows_with_null_exog or impute_exog to handle nulls:\n"
                f"{null_exog_counts[null_exog_counts > 0].to_string()}"
            )

    # Guard against accidental duplication when target is also listed in exogenous_columns.
    unique_exog = [c for c in exogenous_columns if c not in {date_column, target_column}]
    output_columns = [date_column, target_column, *unique_exog]
    output_df = prepared[output_columns].reset_index(drop=True)

    split_info: dict[str, Any] = {
        "rows": len(output_df),
        "start": output_df[date_column].min().date().isoformat() if not output_df.empty else None,
        "end": output_df[date_column].max().date().isoformat() if not output_df.empty else None,
        "missing_periods": missing_periods,
        "null_target_rows_dropped": null_target_rows_dropped,
    }
    return output_df, split_info


def adapt_monthly_data_for_sarimax(
    monthly_train: pd.DataFrame,
    monthly_validation: pd.DataFrame,
    monthly_test: pd.DataFrame,
    monthly_full_train: pd.DataFrame,
    monthly_split_metadata: dict[str, Any],
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Transform generic monthly splits into SARIMAX-ready tabular datasets.

    Produces tabular DataFrames with ``[date_column, target_column, *exogenous_columns]``
    for each split.  Validates temporal index quality, enforces monthly frequency,
    separates target from exogenous features, and emits SARIMAX-specific split metadata.

    Consumes generic ``monthly_*`` datasets; never reads Prophet-renamed columns.

    Args:
        monthly_train: Generic train split.
        monthly_validation: Generic validation split.
        monthly_test: Generic test split.
        monthly_full_train: Generic full-train split.
        monthly_split_metadata: Generic split metadata dict.
        parameters: Pipeline parameters. Reads the ``monthly_sarimax`` block.

    Returns:
        Tuple of five items:
            - sarimax_train: SARIMAX-ready train split DataFrame.
            - sarimax_validation: SARIMAX-ready validation split DataFrame.
            - sarimax_test: SARIMAX-ready test split DataFrame.
            - sarimax_full_train: SARIMAX-ready full-train split DataFrame.
            - sarimax_split_metadata: Metadata dict for the SARIMAX splits.
    """
    sarimax_params = _get_monthly_sarimax_params(parameters)
    date_column: str = sarimax_params.get("date_column", "month_start_date")
    target_column: str = sarimax_params.get("target_column", "monthly_demand")
    sku_column: str = sarimax_params.get("sku_column", "sku")
    exogenous_columns: list[str] = list(sarimax_params.get("exogenous_columns", []))

    splits_raw = {
        "train": monthly_train,
        "validation": monthly_validation,
        "test": monthly_test,
        "full_train": monthly_full_train,
    }

    # Fail early when required columns are missing in any split.
    required_cols = [date_column, target_column, *exogenous_columns]
    for split_name, split_df in splits_raw.items():
        _validate_required_columns(
            split_df,
            required_cols,
            f"monthly_{split_name} (sarimax adapter)",
        )

    logger.info(
        "Adapting generic monthly splits for SARIMAX: date=%s, target=%s, exog=%s.",
        date_column,
        target_column,
        exogenous_columns if exogenous_columns else "(none)",
    )

    prepared_splits: dict[str, pd.DataFrame] = {}
    split_infos: dict[str, Any] = {}
    for split_name, split_df in splits_raw.items():
        prepared, info = _prepare_sarimax_split(
            split_df,
            date_column=date_column,
            target_column=target_column,
            exogenous_columns=exogenous_columns,
            sarimax_params=sarimax_params,
            split_name=split_name,
        )
        prepared_splits[split_name] = prepared
        split_infos[split_name] = info
        logger.info("SARIMAX %s split prepared: %s", split_name, info)

    sarimax_metadata: dict[str, Any] = {
        "granularity": "monthly",
        "model_family": "sarimax",
        "date_column": date_column,
        "target_column": target_column,
        "sku_column": sku_column,
        "frequency": sarimax_params.get("frequency", "MS"),
        "exogenous_columns": exogenous_columns,
        "allow_empty_exog": bool(sarimax_params.get("allow_empty_exog", True)),
        "splits": split_infos,
        "source_metadata": {"from": "monthly_split_metadata"},
        "created_by": "model_input_preparation.sarimax_adapter",
    }
    logger.info("Built monthly_sarimax_split_metadata: %s", sarimax_metadata)

    return (
        prepared_splits["train"],
        prepared_splits["validation"],
        prepared_splits["test"],
        prepared_splits["full_train"],
        sarimax_metadata,
    )


# ── Prophet future regressors and split metadata ──────────────────────────────

def build_monthly_prophet_future_regressors(
    monthly_prophet_modeling_data: pd.DataFrame,
    monthly_calendar_features: pd.DataFrame,
    monthly_exogenous_features: pd.DataFrame,
    parameters: dict,
    calendar_parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct future regressor DataFrames for each configured forecast horizon.

    For each horizon in ``future.horizons_months`` (must include 3, 6, and 12), generates
    a monthly date index starting the month after the last historical observation.
    Calendar features are reused from ``monthly_calendar_features`` where available, and
    computed deterministically for any missing future months.  Exogenous regressors are
    looked up from ``monthly_exogenous_features`` and must be fully populated for all
    future months.

    Args:
        monthly_prophet_modeling_data: Prophet-format historical modeling DataFrame
            (produced by ``adapt_monthly_data_for_prophet``).
        monthly_calendar_features: Monthly calendar feature table.
        monthly_exogenous_features: Monthly exogenous feature table.
        parameters: Pipeline parameters with a ``monthly_prophet`` block.
        calendar_parameters: Calendar feature parameters.

    Returns:
        Tuple of three DataFrames (future_3m, future_6m, future_12m) with schema
        ``[ds, sku, *active_regressors]``.
    """
    monthly_params = _get_monthly_params(parameters)
    supported_horizons: list[int] = list(
        monthly_params.get("supported_future_horizons", [3, 6, 12])
    )
    prophet_params = _get_monthly_prophet_params(parameters)
    prophet_date_column = prophet_params["prophet_date_column"]
    sku_column = prophet_params["sku_column"]
    active_regressors = list(prophet_params["active_regressors"])
    required_horizons = [
        horizon
        for horizon in supported_horizons
        if horizon in prophet_params["future"]["horizons_months"]
    ]
    if required_horizons != supported_horizons:
        raise ValueError(
            "model_input_preparation.monthly_prophet.future.horizons_months must "
            f"include {supported_horizons}."
        )

    logger.info(
        "Building future Prophet regressors from modeling shape=%s, calendar shape=%s, "
        "exogenous shape=%s.",
        monthly_prophet_modeling_data.shape,
        monthly_calendar_features.shape,
        monthly_exogenous_features.shape,
    )

    last_historical_ds = pd.Timestamp(
        monthly_prophet_modeling_data[prophet_date_column].max()
    )
    sku_values = (
        monthly_prophet_modeling_data[sku_column]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if len(sku_values) > 1:
        logger.info(
            "Detected %d SKUs in historical modeling data.", len(sku_values)
        )

    exogenous_regressors = [
        column
        for column in active_regressors
        if column in monthly_exogenous_features.columns
    ]
    missing_regressor_sources = sorted(
        set(active_regressors)
        - set(_CALENDAR_FEATURE_COLUMNS)
        - set(monthly_exogenous_features.columns)
    )
    if missing_regressor_sources:
        raise ValueError(
            "Active regressors are not available in calendar or exogenous sources: "
            f"{missing_regressor_sources}"
        )

    future_datasets: dict[int, pd.DataFrame] = {}
    for horizon in supported_horizons:
        future_months = _generate_future_months(last_historical_ds, horizon)
        future_calendar = _build_future_calendar_features(
            future_months, monthly_calendar_features, calendar_parameters
        )
        future_exogenous = _build_future_exogenous_features(
            future_months, monthly_exogenous_features, exogenous_regressors
        )

        future_regressors = future_calendar.merge(
            future_exogenous, on="month_start_date", how="left", validate="one_to_one"
        )
        future_regressors = future_regressors[
            ["month_start_date", *active_regressors]
        ].copy()

        sku_frame = pd.DataFrame({sku_column: sku_values.tolist()})
        future_dataset = sku_frame.assign(_join_key=1).merge(
            future_regressors.assign(_join_key=1), on="_join_key", how="inner"
        )
        future_dataset = future_dataset.drop(columns="_join_key").rename(
            columns={"month_start_date": prophet_date_column}
        )
        future_dataset = (
            future_dataset[[prophet_date_column, sku_column, *active_regressors]]
            .sort_values([sku_column, prophet_date_column])
            .reset_index(drop=True)
        )

        if prophet_params["prophet_target_column"] in future_dataset.columns:
            raise ValueError("Future Prophet regressor datasets must not include y.")
        _validate_no_nulls(
            future_dataset,
            [prophet_date_column, *active_regressors],
            f"monthly_prophet_future_{horizon}m",
        )
        future_datasets[horizon] = future_dataset
        logger.info(
            "Future horizon %sm: %s",
            horizon,
            _summarize_date_range(future_dataset, prophet_date_column),
        )

    return future_datasets[3], future_datasets[6], future_datasets[12]


def build_monthly_generic_future_frames(
    monthly_modeling_data: pd.DataFrame,
    monthly_calendar_features: pd.DataFrame,
    monthly_exogenous_features: pd.DataFrame,
    parameters: dict,
    calendar_parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build family-agnostic future feature frames for each supported forecast horizon.

    Produces three DataFrames (3m, 6m, 12m) with schema
    ``[month_start_date, sku, *active_regressors]``.  These are the canonical inputs
    for metadata-driven champion inference regardless of the elected model family.
    Calendar features are reused from ``monthly_calendar_features`` where available
    and computed deterministically for missing future months. Exogenous regressors are
    looked up from ``monthly_exogenous_features`` and must be fully available for all
    future months.

    Args:
        monthly_modeling_data: Generic modeling DataFrame with columns
            ``[month_start_date, monthly_demand, sku, *active_regressors]``.
        monthly_calendar_features: Monthly calendar feature table.
        monthly_exogenous_features: Monthly exogenous feature table.
        parameters: Pipeline parameters. Reads the ``monthly`` block.
        calendar_parameters: Calendar feature parameters used to generate calendar
            features for future months not present in ``monthly_calendar_features``.

    Returns:
        Tuple ``(monthly_future_3m, monthly_future_6m, monthly_future_12m)`` with
        schema ``[month_start_date, sku, *active_regressors]``.
    """
    monthly_params = _get_monthly_params(parameters)
    date_column: str = monthly_params["date_column"]
    target_column: str = monthly_params["target_column"]
    sku_column: str = monthly_params["sku_column"]
    active_regressors: list[str] = list(monthly_params["active_regressors"])
    supported_horizons: list[int] = list(
        monthly_params.get("supported_future_horizons", [3, 6, 12])
    )

    last_historical_date = pd.Timestamp(monthly_modeling_data[date_column].max())
    sku_values = (
        monthly_modeling_data[sku_column]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    exogenous_regressors = [
        col for col in active_regressors
        if col in monthly_exogenous_features.columns
    ]
    missing_regressor_sources = sorted(
        set(active_regressors)
        - set(_CALENDAR_FEATURE_COLUMNS)
        - set(monthly_exogenous_features.columns)
    )
    if missing_regressor_sources:
        raise ValueError(
            "Active regressors are not available in calendar or exogenous sources: "
            f"{missing_regressor_sources}"
        )

    logger.info(
        "Building generic monthly future frames — horizons=%s, last_history=%s.",
        supported_horizons,
        last_historical_date.date(),
    )

    future_datasets: dict[int, pd.DataFrame] = {}
    for horizon in supported_horizons:
        future_months = _generate_future_months(last_historical_date, horizon)
        future_calendar = _build_future_calendar_features(
            future_months, monthly_calendar_features, calendar_parameters
        )
        future_exogenous = _build_future_exogenous_features(
            future_months, monthly_exogenous_features, exogenous_regressors
        )

        future_regressors = future_calendar.merge(
            future_exogenous, on="month_start_date", how="left", validate="one_to_one"
        )
        future_regressors = future_regressors[
            ["month_start_date", *active_regressors]
        ].copy()

        sku_frame = pd.DataFrame({sku_column: sku_values.tolist()})
        future_dataset = sku_frame.assign(_join_key=1).merge(
            future_regressors.assign(_join_key=1), on="_join_key", how="inner"
        )
        future_dataset = (
            future_dataset.drop(columns="_join_key")
            [[date_column, sku_column, *active_regressors]]
            .sort_values([sku_column, date_column])
            .reset_index(drop=True)
        )

        if target_column in future_dataset.columns:
            raise ValueError(
                f"Generic monthly future frames must not include the target column "
                f"'{target_column}'."
            )
        _validate_no_nulls(
            future_dataset,
            [date_column, *active_regressors],
            f"monthly_future_{horizon}m",
        )
        future_datasets[horizon] = future_dataset
        logger.info(
            "Generic future horizon %sm: %s",
            horizon,
            _summarize_date_range(future_dataset, date_column),
        )

    return future_datasets[3], future_datasets[6], future_datasets[12]


# ── CatBoost adapter ─────────────────────────────────────────────────────────

def _get_monthly_catboost_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Monthly CatBoost configuration block."""
    if "monthly_catboost" in parameters:
        return dict(parameters["monthly_catboost"])
    return dict(parameters)


def _compute_catboost_target_features(
    df: pd.DataFrame,
    target_column: str,
    sku_column: str,
    date_column: str,
    target_lags: list[int],
    rolling_windows: list[int],
    include_rolling_std: bool,
    include_rolling_min_max: bool,
    trend_diffs: list[int],
    trend_pct_changes: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Compute target-derived lag, rolling, and trend features on the full sorted dataset.

    All features are computed within each SKU group after sorting by date so that
    split boundaries do not introduce null lags in non-leading rows. Rolling aggregations
    are applied to the once-shifted demand series so the current period's demand is never
    used as input for the same period's prediction.

    Args:
        df: Combined monthly dataset sorted by (sku_column, date_column).
        target_column: Name of the demand target column.
        sku_column: Name of the SKU identifier column.
        date_column: Name of the month-start date column.
        target_lags: Lag integers for direct demand lags.
        rolling_windows: Window sizes for rolling aggregations.
        include_rolling_std: Whether to compute rolling standard deviation.
        include_rolling_min_max: Whether to compute rolling min and max.
        trend_diffs: Difference periods (e.g. [1, 12]).
        trend_pct_changes: Percent-change periods (e.g. [1, 12]).

    Returns:
        Tuple of (DataFrame with new feature columns, list of added column names).
    """
    result = df.sort_values([sku_column, date_column]).reset_index(drop=True).copy()
    added: list[str] = []

    # Direct target lags within each SKU group.
    for lag in sorted(target_lags):
        col = f"demand_lag_{lag}"
        result[col] = result.groupby(sku_column, sort=False)[target_column].transform(
            lambda s, n=lag: s.shift(n)
        )
        added.append(col)

    # Rolling aggregations on the once-shifted series (leakage-safe).
    for window in sorted(rolling_windows):
        result[f"rolling_mean_{window}"] = result.groupby(sku_column, sort=False)[target_column].transform(
            lambda s, w=window: s.shift(1).rolling(w).mean()
        )
        added.append(f"rolling_mean_{window}")

        if include_rolling_std:
            result[f"rolling_std_{window}"] = result.groupby(sku_column, sort=False)[target_column].transform(
                lambda s, w=window: s.shift(1).rolling(w).std()
            )
            added.append(f"rolling_std_{window}")

        if include_rolling_min_max:
            result[f"rolling_min_{window}"] = result.groupby(sku_column, sort=False)[target_column].transform(
                lambda s, w=window: s.shift(1).rolling(w).min()
            )
            result[f"rolling_max_{window}"] = result.groupby(sku_column, sort=False)[target_column].transform(
                lambda s, w=window: s.shift(1).rolling(w).max()
            )
            added.extend([f"rolling_min_{window}", f"rolling_max_{window}"])

    # Trend difference features (on once-shifted demand to avoid same-period leakage).
    for period in sorted(trend_diffs):
        col = f"demand_diff_{period}"
        result[col] = result.groupby(sku_column, sort=False)[target_column].transform(
            lambda s, p=period: s.shift(1).diff(p)
        )
        added.append(col)

    # Percent-change features (on once-shifted demand).
    for period in sorted(trend_pct_changes):
        col = f"demand_pct_change_{period}"
        result[col] = result.groupby(sku_column, sort=False)[target_column].transform(
            lambda s, p=period: s.shift(1).pct_change(p)
        )
        added.append(col)

    # Ratio of short vs long rolling mean — only when both are available.
    if "rolling_mean_3" in added and "rolling_mean_12" in added:
        denom = result["rolling_mean_12"].replace(0.0, float("nan"))
        result["rolling_mean_3_vs_12"] = result["rolling_mean_3"] / denom
        added.append("rolling_mean_3_vs_12")

    return result, added


def adapt_monthly_data_for_catboost(
    monthly_train: pd.DataFrame,
    monthly_validation: pd.DataFrame,
    monthly_test: pd.DataFrame,
    monthly_full_train: pd.DataFrame,
    monthly_split_metadata: dict[str, Any],
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build CatBoost-ready monthly datasets with target-derived lag and rolling features.

    Computes target-derived lag, rolling, trend, and calendar enrichment features on the
    full chronological dataset before re-splitting, so that split boundaries do not create
    null lags at the first rows of the validation or test period.  Leading rows with null
    target lag features are dropped from the train split only.

    Args:
        monthly_train: Generic monthly train split.
        monthly_validation: Generic monthly validation split.
        monthly_test: Generic monthly test split.
        monthly_full_train: Generic monthly full-train dataset (all splits combined).
        monthly_split_metadata: Generic split metadata with train/validation/test date ranges.
        parameters: Pipeline parameters. Reads the ``monthly_catboost`` block.

    Returns:
        Tuple of five items:
            - monthly_catboost_train: CatBoost-ready train DataFrame.
            - monthly_catboost_validation: CatBoost-ready validation DataFrame.
            - monthly_catboost_test: CatBoost-ready test DataFrame.
            - monthly_catboost_full_train: CatBoost-ready full-train DataFrame.
            - monthly_catboost_split_metadata: Metadata dict for the CatBoost splits.
    """
    catboost_params = _get_monthly_catboost_params(parameters)
    date_column: str = catboost_params.get("date_column", "month_start_date")
    target_column: str = catboost_params.get("target_column", "monthly_demand")
    sku_column: str = catboost_params.get("sku_column", "sku")
    target_lags: list[int] = list(catboost_params.get("target_lags", [1, 2, 3, 6, 12]))
    rolling_windows: list[int] = list(catboost_params.get("rolling_windows", [3, 6, 12]))
    include_rolling_std: bool = bool(catboost_params.get("include_rolling_std", True))
    include_rolling_min_max: bool = bool(catboost_params.get("include_rolling_min_max", True))
    trend_diffs: list[int] = list(catboost_params.get("trend_diffs", [1, 12]))
    trend_pct_changes: list[int] = list(catboost_params.get("trend_pct_changes", [1, 12]))
    drop_null: bool = bool(catboost_params.get("drop_rows_with_null_target_features", True))
    include_missingness_flags: bool = bool(catboost_params.get("include_missingness_flags", False))
    missingness_flag_columns: list[str] = list(catboost_params.get("missingness_flag_columns", []))
    catboost_active_regressors: list[str] = list(catboost_params.get("active_regressors", []))

    _validate_required_columns(
        monthly_full_train,
        [date_column, target_column, sku_column],
        "monthly_full_train",
    )

    if catboost_active_regressors:
        missing_regressors = [
            c for c in catboost_active_regressors if c not in monthly_full_train.columns
        ]
        if missing_regressors:
            raise ValueError(
                f"monthly_catboost.active_regressors contains columns not found in the dataset: "
                f"{missing_regressors}. Available columns: {sorted(monthly_full_train.columns)}"
            )

    logger.info(
        "Building CatBoost monthly feature dataset from monthly_full_train shape=%s.",
        monthly_full_train.shape,
    )

    combined = _ensure_datetime_column(monthly_full_train.copy(), date_column, "monthly_full_train")

    # Add month calendar feature — deterministic and always known at inference time.
    combined["month"] = combined[date_column].dt.month.astype("int64")
    new_columns: list[str] = ["month"]

    # Compute all target-derived features on the full sorted dataset.
    combined, target_feature_columns = _compute_catboost_target_features(
        combined,
        target_column=target_column,
        sku_column=sku_column,
        date_column=date_column,
        target_lags=target_lags,
        rolling_windows=rolling_windows,
        include_rolling_std=include_rolling_std,
        include_rolling_min_max=include_rolling_min_max,
        trend_diffs=trend_diffs,
        trend_pct_changes=trend_pct_changes,
    )
    new_columns.extend(target_feature_columns)

    # Optionally add missingness flags for specified exogenous columns.
    if include_missingness_flags and missingness_flag_columns:
        for flag_source in missingness_flag_columns:
            if flag_source in combined.columns:
                combined[f"is_missing_{flag_source}"] = combined[flag_source].isna().astype("int8")
                new_columns.append(f"is_missing_{flag_source}")
            else:
                logger.warning(
                    "missingness_flag_columns entry %r not found in dataset — skipped.",
                    flag_source,
                )

    logger.info(
        "Generated %d new CatBoost feature columns: %s.",
        len(new_columns),
        new_columns,
    )

    # Re-split using date boundaries from the generic split metadata.
    train_end = pd.Timestamp(monthly_split_metadata["train"]["end_date"])
    val_end = pd.Timestamp(monthly_split_metadata["validation"]["end_date"])
    test_end = pd.Timestamp(monthly_split_metadata["test"]["end_date"])

    catboost_train = combined.loc[combined[date_column] <= train_end].copy()
    catboost_validation = combined.loc[
        (combined[date_column] > train_end) & (combined[date_column] <= val_end)
    ].copy()
    catboost_test = combined.loc[
        (combined[date_column] > val_end) & (combined[date_column] <= test_end)
    ].copy()

    # Drop leading train rows where target lag features are null (lag warmup period).
    target_lag_cols = [
        f"demand_lag_{lag}" for lag in target_lags if f"demand_lag_{lag}" in combined.columns
    ]
    dropped_count = 0
    if target_lag_cols and drop_null:
        null_mask = catboost_train[target_lag_cols].isnull().any(axis=1)
        dropped_count = int(null_mask.sum())
        catboost_train = catboost_train.loc[~null_mask].copy()
        logger.info(
            "Dropped %d leading train rows with null target lag features (lag warmup).",
            dropped_count,
        )

    # Warn when validation or test rows still carry null target lag features.
    for split_name, split_df in [("validation", catboost_validation), ("test", catboost_test)]:
        if target_lag_cols and split_df[target_lag_cols].isnull().any().any():
            null_counts = split_df[target_lag_cols].isnull().sum()
            logger.warning(
                "Null target lag features detected in %s split — training data may be too short: %s",
                split_name,
                null_counts[null_counts > 0].to_dict(),
            )

    for split_name, split_df in [
        ("train", catboost_train),
        ("validation", catboost_validation),
        ("test", catboost_test),
    ]:
        if split_df.empty:
            raise ValueError(
                f"CatBoost {split_name} split is empty after feature generation. "
                "Check that training data is long enough to survive the lag warmup period."
            )

    catboost_full_train = (
        pd.concat([catboost_train, catboost_validation, catboost_test], ignore_index=True)
        .sort_values([sku_column, date_column])
        .reset_index(drop=True)
    )

    logger.info("CatBoost train: %s", _summarize_date_range(catboost_train, date_column))
    logger.info("CatBoost validation: %s", _summarize_date_range(catboost_validation, date_column))
    logger.info("CatBoost test: %s", _summarize_date_range(catboost_test, date_column))

    # When catboost_active_regressors is non-empty, restrict exogenous columns to that
    # subset only (keeps target-derived features intact). Falls back to all shared columns
    # when the list is empty.
    if catboost_active_regressors:
        keep_cols = [date_column, target_column, sku_column, *catboost_active_regressors, *new_columns]
        catboost_train = catboost_train[[c for c in keep_cols if c in catboost_train.columns]].copy()
        catboost_validation = catboost_validation[
            [c for c in keep_cols if c in catboost_validation.columns]
        ].copy()
        catboost_test = catboost_test[[c for c in keep_cols if c in catboost_test.columns]].copy()
        catboost_full_train = catboost_full_train[
            [c for c in keep_cols if c in catboost_full_train.columns]
        ].copy()
        logger.info(
            "CatBoost-specific active_regressors applied: %d exogenous columns selected: %s.",
            len(catboost_active_regressors),
            catboost_active_regressors,
        )
    else:
        logger.info("No catboost-specific active_regressors configured — using all shared exogenous columns.")

    all_feature_columns = [
        col for col in catboost_full_train.columns
        if col not in {date_column, target_column, sku_column}
    ]
    base_feature_columns = [
        col for col in catboost_full_train.columns
        if col not in {date_column, target_column, sku_column} and col not in new_columns
    ]

    # Target-derived columns that cannot be known at future inference time without
    # re-computing them from arriving actuals (lags, rolling, diffs, pct_changes).
    _target_derived_prefixes = (
        "demand_lag_",
        "rolling_mean_",
        "rolling_std_",
        "rolling_min_",
        "rolling_max_",
        "rolling_mean_3_vs_12",
        "demand_diff_",
        "demand_pct_change_",
    )
    future_required_columns = [
        col for col in all_feature_columns
        if not any(col.startswith(prefix) for prefix in _target_derived_prefixes)
        and col != "rolling_mean_3_vs_12"
    ]

    # Columns expected to be structurally null at the boundary of the lag warmup
    # window (e.g. demand_diff_12 requires 13 + 1 prior observations).
    structural_null_columns = [
        col for col in all_feature_columns
        if col.startswith("demand_diff_") or col.startswith("demand_pct_change_")
    ]

    catboost_metadata: dict[str, Any] = {
        "granularity": "monthly",
        "model_family": "catboost",
        "date_column": date_column,
        "target_column": target_column,
        "sku_column": sku_column,
        "all_feature_columns": all_feature_columns,
        "categorical_feature_columns": [],
        "null_handling_policy": "catboost_native",
        "structural_null_columns": structural_null_columns,
        "base_feature_columns": base_feature_columns,
        "new_feature_columns": new_columns,
        "target_lag_columns": target_lag_cols,
        "rolling_feature_columns": [c for c in new_columns if c.startswith("rolling_")],
        "trend_feature_columns": [
            c for c in new_columns
            if c.startswith("demand_diff_") or c.startswith("demand_pct_change_")
        ],
        "future_required_columns": future_required_columns,
        "lag_settings": {"target_lags": target_lags},
        "rolling_settings": {
            "windows": rolling_windows,
            "include_std": include_rolling_std,
            "include_min_max": include_rolling_min_max,
        },
        "splits": {
            "train": _summarize_date_range(catboost_train, date_column),
            "validation": _summarize_date_range(catboost_validation, date_column),
            "test": _summarize_date_range(catboost_test, date_column),
            "full_train": _summarize_date_range(catboost_full_train, date_column),
        },
        "dropped_rows": {"null_target_features_in_train": dropped_count},
        "source_metadata": {"from": "monthly_split_metadata"},
        "created_by": "model_input_preparation.catboost_adapter",
    }

    logger.info("Built monthly_catboost_split_metadata.")
    return (
        catboost_train.reset_index(drop=True),
        catboost_validation.reset_index(drop=True),
        catboost_test.reset_index(drop=True),
        catboost_full_train,
        catboost_metadata,
    )


# ── Prophet future regressors and split metadata ──────────────────────────────

def build_monthly_prophet_split_metadata(  # noqa: PLR0912, PLR0913
    monthly_prophet_train: pd.DataFrame,
    monthly_prophet_validation: pd.DataFrame,
    monthly_prophet_test: pd.DataFrame,
    monthly_prophet_full_train: pd.DataFrame,
    monthly_prophet_future_3m: pd.DataFrame,
    monthly_prophet_future_6m: pd.DataFrame,
    monthly_prophet_future_12m: pd.DataFrame,
    preparation_metadata: dict[str, Any],
    parameters: dict,
) -> dict[str, Any]:
    """Compile a single metadata dict covering all Prophet split partitions and future horizons.

    Args:
        monthly_prophet_train: Prophet train split.
        monthly_prophet_validation: Prophet validation split.
        monthly_prophet_test: Prophet test split.
        monthly_prophet_full_train: Prophet full-train split.
        monthly_prophet_future_3m: Future regressor DataFrame for the 3-month horizon.
        monthly_prophet_future_6m: Future regressor DataFrame for the 6-month horizon.
        monthly_prophet_future_12m: Future regressor DataFrame for the 12-month horizon.
        preparation_metadata: Internal metadata dict produced by ``adapt_monthly_data_for_prophet``.
            Must contain ``model_family``, ``granularity``, ``split_mode``,
            ``active_regressors``, and ``dropped_rows``.
        parameters: Pipeline parameters with a ``monthly_prophet`` block.

    Returns:
        Dict with model_family, granularity, split_mode, active_regressors, train,
        validation, test, full_train, future_horizons, and dropped_rows.
    """
    prophet_params = _get_monthly_prophet_params(parameters)
    prophet_date_column = prophet_params["prophet_date_column"]

    metadata = {
        "model_family": preparation_metadata["model_family"],
        "granularity": preparation_metadata["granularity"],
        "split_mode": preparation_metadata["split_mode"],
        "active_regressors": preparation_metadata["active_regressors"],
        "train": _summarize_date_range(monthly_prophet_train, prophet_date_column),
        "validation": _summarize_date_range(
            monthly_prophet_validation, prophet_date_column
        ),
        "test": _summarize_date_range(monthly_prophet_test, prophet_date_column),
        "full_train": _summarize_date_range(
            monthly_prophet_full_train, prophet_date_column
        ),
        "future_horizons": {
            3: _summarize_date_range(monthly_prophet_future_3m, prophet_date_column),
            6: _summarize_date_range(monthly_prophet_future_6m, prophet_date_column),
            12: _summarize_date_range(monthly_prophet_future_12m, prophet_date_column),
        },
        "dropped_rows": preparation_metadata["dropped_rows"],
    }
    logger.info("Prepared monthly Prophet split metadata: %s", metadata)
    return metadata
