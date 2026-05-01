"""Monthly Prophet model-input preparation nodes."""

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
_SUPPORTED_FUTURE_HORIZONS = (3, 6, 12)
_MIN_WORKING_WEEKDAY_COUNT_FOR_FLAG = 5


def _get_monthly_prophet_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Monthly Prophet configuration block."""
    if "monthly_prophet" in parameters:
        return dict(parameters["monthly_prophet"])
    return dict(parameters)


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
    """Validate Prophet's required schema types."""
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply conservative null handling required by Prophet."""
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
            # Early lagged features can be null by design; Prophet requires complete
            # regressor rows, so those leading months are removed instead of imputed.
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
        "monthly_prophet_modeling_data",
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
    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }
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
) -> dict[str, int | pd.Timestamp]:
    """Count deterministic calendar features for a single month."""
    month_start = pd.Timestamp(month_start_date).normalize()
    month_end = month_start + pd.offsets.MonthEnd(1)
    month_days = pd.date_range(start=month_start, end=month_end, freq="D")

    # Boolean masks per day; combine them with & / ~ to count each feature.
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
        "has_5_working_tuesdays": int(
            working_tuesdays >= _MIN_WORKING_WEEKDAY_COUNT_FOR_FLAG
        ),
        "has_5_working_thursdays": int(
            working_thursdays >= _MIN_WORKING_WEEKDAY_COUNT_FOR_FLAG
        ),
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

    # Set difference: future months not already covered by the catalog dataset.
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
        generated_calendar = pd.DataFrame(
            [
                _count_monthly_calendar_features(
                    month_start, holiday_dates, business_weekdays
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


def prepare_monthly_prophet_modeling_data(
    monthly_prophet_features: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rename, filter, and clean the monthly feature table for use as Prophet historical input.

    Selects the configured date, target, SKU, and active regressor columns; renames them
    to Prophet's expected schema (``ds``, ``y``); and drops rows with null targets or null
    active regressors according to the ``missing_values`` parameter block.

    Args:
        monthly_prophet_features: Feature-engineered monthly DataFrame produced by the
            feature engineering pipeline. Must contain the columns declared in
            ``parameters["monthly_prophet"]``.
        parameters: Pipeline parameters. Reads the ``monthly_prophet`` block or falls back
            to the top-level dict. Expected keys: ``date_column``, ``target_column``,
            ``sku_column``, ``prophet_date_column``, ``prophet_target_column``,
            ``active_regressors``, ``missing_values``.

    Returns:
        Tuple of:
            - modeling_df: Cleaned DataFrame with columns ``[ds, y, sku, *active_regressors]``
              sorted by SKU and date.
            - preparation_metadata: Dict summarising model family, granularity, active
              regressors, dropped-row counts, and the date range of the output.

    Raises:
        ValueError: If required columns are missing, dates cannot be parsed, the dataset
            is not unique by (SKU, date), or nulls remain after the configured drop policy.
    """
    prophet_params = _get_monthly_prophet_params(parameters)
    date_column = prophet_params["date_column"]
    target_column = prophet_params["target_column"]
    sku_column = prophet_params["sku_column"]
    prophet_date_column = prophet_params["prophet_date_column"]
    prophet_target_column = prophet_params["prophet_target_column"]
    active_regressors = _get_active_regressors(
        monthly_prophet_features,
        prophet_params["active_regressors"],
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
        "Preparing monthly Prophet modeling data from shape=%s.",
        feature_df.shape,
    )
    logger.info("Selected active regressors: %s", active_regressors)
    logger.info(
        "Input null counts for target and active regressors:\n%s",
        feature_df[[target_column, *active_regressors]].isnull().sum().to_string(),
    )

    modeling_df = (
        feature_df[[date_column, target_column, sku_column, *active_regressors]]
        .rename(
            columns={
                date_column: prophet_date_column,
                target_column: prophet_target_column,
            }
        )
        .sort_values([sku_column, prophet_date_column])
        .reset_index(drop=True)
    )
    _validate_datetime_and_numeric(
        modeling_df,
        prophet_date_column,
        prophet_target_column,
    )

    modeling_df, dropped_rows = _drop_null_modeling_rows(
        modeling_df,
        date_column=prophet_date_column,
        target_column=prophet_target_column,
        active_regressors=active_regressors,
        missing_value_params=prophet_params["missing_values"],
    )

    logger.info(
        "Dropped %d rows with null target values.",
        dropped_rows["null_target"],
    )
    logger.info(
        "Dropped %d rows with null active regressors. Columns with nulls=%s",
        dropped_rows["null_active_regressors"],
        dropped_rows["null_active_regressor_columns"],
    )
    logger.info(
        "Final monthly_prophet_modeling_data shape=%s, range=[%s, %s], columns=%s.",
        modeling_df.shape,
        modeling_df[prophet_date_column].min().date(),
        modeling_df[prophet_date_column].max().date(),
        list(modeling_df.columns),
    )

    preparation_metadata: dict[str, Any] = {
        "model_family": "prophet",
        "granularity": "monthly",
        "active_regressors": active_regressors,
        "dropped_rows": {
            "null_target": dropped_rows["null_target"],
            "null_active_regressors": dropped_rows["null_active_regressors"],
        },
        "modeling_data": _summarize_date_range(modeling_df, prophet_date_column),
    }
    return modeling_df, preparation_metadata


def split_monthly_prophet_data(
    monthly_prophet_modeling_data: pd.DataFrame,
    preparation_metadata: dict[str, Any],
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition the Prophet modeling table into train, validation, test, and full-train splits.

    Supports two split modes controlled by ``parameters["monthly_prophet"]["split"]["mode"]``:

    - ``"date"`` — explicit cutoff dates (``train_end_date``, ``validation_end_date``,
      ``test_end_date``).
    - ``"months"`` — trailing month counts (``validation_months``, ``test_months``).

    ``full_train`` always concatenates all three partitions and is intended for the final
    retrain before production inference.

    Args:
        monthly_prophet_modeling_data: Cleaned modeling DataFrame from
            ``prepare_monthly_prophet_modeling_data``.
        preparation_metadata: Metadata dict produced by the preparation node; extended
            with split summaries before being returned.
        parameters: Pipeline parameters with a ``monthly_prophet.split`` block.

    Returns:
        Tuple of five items:
            - train: Historical rows reserved for model fitting.
            - validation: Held-out rows for hyperparameter selection.
            - test: Held-out rows for final evaluation.
            - full_train: All rows combined (train + validation + test), sorted by SKU and date.
            - updated_metadata: Metadata dict extended with date-range summaries per partition.

    Raises:
        ValueError: If ``split_mode`` is not ``"date"`` or ``"months"``, if date boundaries
            are in the wrong order, if there are insufficient months for a non-empty train
            set, or if any split partition is empty or overlaps with another.
    """
    prophet_params = _get_monthly_prophet_params(parameters)
    prophet_date_column = prophet_params["prophet_date_column"]
    split_params = prophet_params["split"]
    split_mode = split_params["mode"]

    logger.info(
        "Splitting monthly Prophet modeling data shape=%s using split mode=%s.",
        monthly_prophet_modeling_data.shape,
        split_mode,
    )

    if split_mode == "date":
        train, validation, test = _split_by_dates(
            monthly_prophet_modeling_data,
            prophet_date_column,
            split_params,
        )
    elif split_mode == "months":
        train, validation, test = _split_by_month_counts(
            monthly_prophet_modeling_data,
            prophet_date_column,
            split_params,
        )
    else:
        raise ValueError(
            f"Unsupported split mode {split_mode!r}. Expected 'date' or 'months'."
        )

    _validate_split_frames(train, validation, test, prophet_date_column)
    full_train = (
        pd.concat([train, validation, test], ignore_index=True)
        .sort_values(["sku", prophet_date_column])
        .reset_index(drop=True)
    )

    logger.info(
        "Train split summary: %s", _summarize_date_range(train, prophet_date_column)
    )
    logger.info(
        "Validation split summary: %s",
        _summarize_date_range(validation, prophet_date_column),
    )
    logger.info(
        "Test split summary: %s", _summarize_date_range(test, prophet_date_column)
    )

    updated_metadata = dict(preparation_metadata)
    updated_metadata["split_mode"] = split_mode
    updated_metadata["train"] = _summarize_date_range(train, prophet_date_column)
    updated_metadata["validation"] = _summarize_date_range(
        validation,
        prophet_date_column,
    )
    updated_metadata["test"] = _summarize_date_range(test, prophet_date_column)
    updated_metadata["full_train"] = _summarize_date_range(
        full_train,
        prophet_date_column,
    )

    return train, validation, test, full_train, updated_metadata


def build_monthly_prophet_future_regressors(
    monthly_prophet_modeling_data: pd.DataFrame,
    monthly_calendar_features: pd.DataFrame,
    monthly_exogenous_features: pd.DataFrame,
    parameters: dict,
    calendar_parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct future regressor DataFrames for each configured forecast horizon.

    For each horizon in ``future.horizons_months`` (must include both 3 and 6), generates a
    monthly date index starting the month after the last historical observation. Calendar
    features are reused from ``monthly_calendar_features`` where available, and computed
    deterministically for any missing future months. Exogenous regressors are looked up
    from ``monthly_exogenous_features`` and must be fully populated for all future months.

    The output DataFrames contain only ``[ds, sku, *active_regressors]`` — no target
    column ``y`` — as required by Prophet's ``predict`` API.

    Args:
        monthly_prophet_modeling_data: Cleaned historical modeling DataFrame. Used to
            determine the last historical date and the distinct set of SKUs.
        monthly_calendar_features: Monthly calendar feature table; rows covering the future
            horizon are reused directly; any missing months are computed on the fly.
        monthly_exogenous_features: Monthly exogenous feature table. Must cover all future
            months for every active exogenous regressor.
        parameters: Pipeline parameters with a ``monthly_prophet`` block. Reads
            ``prophet_date_column``, ``sku_column``, ``active_regressors``, and
            ``future.horizons_months``.
        calendar_parameters: Calendar feature parameters used when future months must be
            generated. Reads ``calendar_features.country_holidays``,
            ``observed_holidays``, and ``weekmask``.

    Returns:
        Tuple of three DataFrames:
            - future_3m: Future regressor table for the 3-month horizon.
            - future_6m: Future regressor table for the 6-month horizon.
            - future_12m: Future regressor table for the 12-month horizon.
        All share the schema ``[ds, sku, *active_regressors]``, cross-joined over all SKUs.

    Raises:
        ValueError: If ``horizons_months`` does not include 3, 6, and 12, if an active
            regressor cannot be sourced from either the calendar or exogenous datasets,
            if required future exogenous values are missing, or if the output accidentally
            contains the target column ``y``.
    """
    prophet_params = _get_monthly_prophet_params(parameters)
    prophet_date_column = prophet_params["prophet_date_column"]
    sku_column = prophet_params["sku_column"]
    active_regressors = list(prophet_params["active_regressors"])
    required_horizons = [
        horizon
        for horizon in _SUPPORTED_FUTURE_HORIZONS
        if horizon in prophet_params["future"]["horizons_months"]
    ]
    if required_horizons != list(_SUPPORTED_FUTURE_HORIZONS):
        raise ValueError(
            "model_input_preparation.monthly_prophet.future.horizons_months must "
            f"include {_SUPPORTED_FUTURE_HORIZONS}."
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
            "Detected %d SKUs in historical modeling data. Future regressors will be "
            "generated for each SKU.",
            len(sku_values),
        )

    # Split active regressors by source: those in the exogenous dataset vs. calendar-derived.
    exogenous_regressors = [
        column
        for column in active_regressors
        if column in monthly_exogenous_features.columns
    ]
    # Any regressor not found in either source cannot be materialized for future horizons.
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
    for horizon in _SUPPORTED_FUTURE_HORIZONS:
        future_months = _generate_future_months(last_historical_ds, horizon)
        future_calendar = _build_future_calendar_features(
            future_months,
            monthly_calendar_features,
            calendar_parameters,
        )
        future_exogenous = _build_future_exogenous_features(
            future_months,
            monthly_exogenous_features,
            exogenous_regressors,
        )

        future_regressors = future_calendar.merge(
            future_exogenous,
            on="month_start_date",
            how="left",
            validate="one_to_one",
        )
        future_regressors = future_regressors[
            ["month_start_date", *active_regressors]
        ].copy()

        # Cross-join: replicate future months for every SKU via a constant join key.
        sku_frame = pd.DataFrame({sku_column: sku_values.tolist()})
        future_dataset = sku_frame.assign(_join_key=1).merge(
            future_regressors.assign(_join_key=1),
            on="_join_key",
            how="inner",
        )
        future_dataset = future_dataset.drop(columns="_join_key").rename(
            columns={"month_start_date": prophet_date_column}
        )
        # Future inference only needs dates, SKU metadata, and known regressors.
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
            "Future horizon %sm summary: %s",
            horizon,
            _summarize_date_range(future_dataset, prophet_date_column),
        )

    logger.info(
        "Built future Prophet datasets with columns=%s.",
        list(future_datasets[3].columns),
    )
    return future_datasets[3], future_datasets[6], future_datasets[12]


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

    Aggregates date-range summaries for each partition (train, validation, test, full_train)
    and each future horizon (3 m, 6 m) into one artifact for downstream traceability and
    experiment logging in MLflow.

    Args:
        monthly_prophet_train: Train split DataFrame.
        monthly_prophet_validation: Validation split DataFrame.
        monthly_prophet_test: Test split DataFrame.
        monthly_prophet_full_train: Full-train split DataFrame (all partitions combined).
        monthly_prophet_future_3m: Future regressor DataFrame for the 3-month horizon.
        monthly_prophet_future_6m: Future regressor DataFrame for the 6-month horizon.
        monthly_prophet_future_12m: Future regressor DataFrame for the 12-month horizon.
        preparation_metadata: Metadata produced by the preparation and split nodes.
            Must contain ``model_family``, ``granularity``, ``split_mode``,
            ``active_regressors``, and ``dropped_rows``.
        parameters: Pipeline parameters with a ``monthly_prophet`` block. Reads
            ``prophet_date_column``.

    Returns:
        Dict with keys: ``model_family``, ``granularity``, ``split_mode``,
        ``active_regressors``, ``train``, ``validation``, ``test``, ``full_train``,
        ``future_horizons`` (keyed by horizon integer 3 and 6), and ``dropped_rows``.
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
            monthly_prophet_validation,
            prophet_date_column,
        ),
        "test": _summarize_date_range(monthly_prophet_test, prophet_date_column),
        "full_train": _summarize_date_range(
            monthly_prophet_full_train,
            prophet_date_column,
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
