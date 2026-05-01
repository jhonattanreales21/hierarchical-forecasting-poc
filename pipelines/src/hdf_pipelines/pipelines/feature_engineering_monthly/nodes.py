"""Monthly feature engineering nodes for the Prophet MVP."""

import logging
from collections.abc import Sequence

import holidays
import pandas as pd
from pandas.tseries.offsets import MonthEnd

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
_MIN_WORKING_WEEKDAY_COUNT_FOR_FLAG = 5


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


def _ensure_month_start_dates(
    df: pd.DataFrame,
    date_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Normalize the configured date column to month-start timestamps."""
    normalized_df = df.copy()
    normalized_df[date_column] = pd.to_datetime(normalized_df[date_column])

    normalized_dates = normalized_df[date_column].dt.to_period("M").dt.to_timestamp()
    changed_rows = int((normalized_df[date_column] != normalized_dates).sum())
    if changed_rows > 0:
        logger.warning(
            "%s contains %d non-month-start dates. Normalizing them to month start.",
            dataset_name,
            changed_rows,
        )

    normalized_df[date_column] = normalized_dates
    return normalized_df


def _validate_unique_rows(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Ensure the dataset is unique on the expected business key."""
    # keep=False marks every copy of a duplicate so the error report shows the full conflicting set.
    duplicated = df.duplicated(subset=list(key_columns), keep=False)
    if duplicated.any():
        duplicated_rows = df.loc[duplicated, list(key_columns)].sort_values(
            list(key_columns)
        )
        raise ValueError(
            f"{dataset_name} must be unique by {list(key_columns)}. "
            f"Found duplicated keys:\n{duplicated_rows.to_string(index=False)}"
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


def _get_month_date_range(month_start_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Return every calendar day that belongs to the given month."""
    month_start = pd.Timestamp(month_start_date).normalize()
    month_end = month_start + MonthEnd(1)
    return pd.date_range(start=month_start, end=month_end, freq="D")


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
    # Expand the month to daily dates so weekday and holiday counts remain explicit.
    month_days = _get_month_date_range(month_start_date)
    is_holiday = month_days.isin(holiday_dates)
    is_business_weekday = month_days.dayofweek.isin(sorted(business_weekdays))

    # Tuesdays and Thursdays matter because downstream business logic uses those cycles.
    is_tuesday = month_days.dayofweek == _WEEKDAY_TO_NUMBER["Tue"]
    is_thursday = month_days.dayofweek == _WEEKDAY_TO_NUMBER["Thu"]

    working_tuesdays = int((is_tuesday & ~is_holiday).sum())
    working_thursdays = int((is_thursday & ~is_holiday).sum())

    return {
        "month_start_date": pd.Timestamp(month_start_date).normalize(),
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


def _add_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    date_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create chronological lag features without dropping the leading null rows."""
    lagged_df = df.copy().sort_values(date_column).reset_index(drop=True)
    lagged_columns: list[str] = []

    for lag in lags:
        for column in columns:
            lag_column = f"{column}_lag_{lag}"
            # Leading nulls are expected because early months have no historical values yet.
            lagged_df[lag_column] = lagged_df[column].shift(lag)
            lagged_columns.append(lag_column)

    return lagged_df, lagged_columns


def _format_month_list(months: Sequence[pd.Timestamp]) -> list[str]:
    """Convert timestamps to ISO date strings for compact logging."""
    return [pd.Timestamp(month).date().isoformat() for month in months]


def build_monthly_calendar_features(
    demand_monthly: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Compute deterministic calendar features for every month present in the demand table.

    For each unique month, counts business days, Tuesdays, Thursdays, working Tuesdays,
    working Thursdays, and public holidays using the configured country calendar and
    weekmask. Results are independent of demand values, so the same table can be reused
    for future forecast horizons without recomputation.

    Args:
        demand_monthly: Monthly demand DataFrame used only to determine which months to
            compute calendar features for. Must contain ``date_column`` and ``sku_column``
            as declared in ``parameters``.
        parameters: Feature engineering parameters. Expected keys: ``date_column``,
            ``sku_column``, ``target_column``, and a ``calendar_features`` block with
            ``enabled`` (bool), ``country_holidays`` (ISO country code),
            ``observed_holidays`` (bool), and ``weekmask`` (NumPy-style string, e.g.
            ``"Mon Tue Wed Thu Fri"``).

    Returns:
        Calendar feature DataFrame with one row per unique month and columns:
        ``month_start_date``, ``business_days``, ``total_tuesdays``, ``total_thursdays``,
        ``working_tuesdays``, ``working_thursdays``, ``has_5_working_tuesdays``,
        ``has_5_working_thursdays``, ``tuesday_holidays``, ``thursday_holidays``,
        ``total_holidays``. All feature columns are cast to ``int64``.

    Raises:
        ValueError: If ``calendar_features.enabled`` is ``False``, required columns are
            missing from ``demand_monthly``, or the weekmask contains unrecognized day names.
    """
    if not parameters["calendar_features"]["enabled"]:
        raise ValueError(
            "feature_engineering_monthly.calendar_features.enabled must be true."
        )

    date_column = parameters["date_column"]
    sku_column = parameters["sku_column"]
    target_column = parameters["target_column"]

    _validate_required_columns(
        demand_monthly,
        [date_column, sku_column, target_column],
        "demand_monthly",
    )

    demand_df = _ensure_month_start_dates(demand_monthly, date_column, "demand_monthly")
    _validate_unique_rows(demand_df, [sku_column, date_column], "demand_monthly")

    logger.info(
        "Building monthly calendar features from demand_monthly shape=%s, date range=[%s, %s].",
        demand_df.shape,
        demand_df[date_column].min().date(),
        demand_df[date_column].max().date(),
    )

    month_index = pd.DatetimeIndex(
        demand_df[date_column].drop_duplicates().sort_values().tolist()
    )
    calendar_params = parameters["calendar_features"]
    holiday_dates = _get_country_holidays(
        country_code=calendar_params["country_holidays"],
        years=month_index.year.unique().tolist(),
        observed=calendar_params["observed_holidays"],
    )
    business_weekdays = _parse_weekmask(calendar_params["weekmask"])

    calendar_rows = [
        _count_monthly_calendar_features(month_start, holiday_dates, business_weekdays)
        for month_start in month_index
    ]
    calendar_features = (
        pd.DataFrame(calendar_rows).sort_values(date_column).reset_index(drop=True)
    )
    calendar_features[_CALENDAR_FEATURE_COLUMNS] = calendar_features[
        _CALENDAR_FEATURE_COLUMNS
    ].astype("int64")

    logger.info(
        "Built monthly calendar features shape=%s, date range=[%s, %s], columns=%s.",
        calendar_features.shape,
        calendar_features[date_column].min().date(),
        calendar_features[date_column].max().date(),
        _CALENDAR_FEATURE_COLUMNS,
    )
    return calendar_features


def build_monthly_exogenous_features(
    exogenous_monthly: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Build monthly exogenous features and append chronological lag columns.

    Selects the declared ``base_columns`` from the exogenous primary dataset and creates
    lag columns (e.g., ``pfizer_limited_lag_1``) for every configured lag value. Leading
    null rows introduced by lagging are kept intentionally — the downstream
    model-input preparation node drops them after all sources are joined.

    Args:
        exogenous_monthly: Monthly exogenous DataFrame from the primary layer. Must
            contain ``date_column`` and all ``base_columns`` declared in ``parameters``.
        parameters: Feature engineering parameters. Expected keys: ``date_column``, and
            an ``exogenous_features`` block with ``enabled`` (bool), ``base_columns``
            (list of column names to include and lag), and ``lags`` (list of positive
            integers, e.g. ``[1, 2]``).

    Returns:
        Monthly exogenous feature DataFrame with columns: ``date_column``,
        ``*base_columns``, and one ``{col}_lag_{n}`` column per (base column, lag)
        combination. Sorted by date.

    Raises:
        ValueError: If ``exogenous_features.enabled`` is ``False`` or required columns
            are missing from ``exogenous_monthly``.
    """
    if not parameters["exogenous_features"]["enabled"]:
        raise ValueError(
            "feature_engineering_monthly.exogenous_features.enabled must be true."
        )

    date_column = parameters["date_column"]
    exogenous_params = parameters["exogenous_features"]
    base_columns = exogenous_params["base_columns"]
    lags = exogenous_params["lags"]

    _validate_required_columns(
        exogenous_monthly,
        [date_column, *base_columns],
        "exogenous_monthly",
    )

    exogenous_df = _ensure_month_start_dates(
        exogenous_monthly,
        date_column,
        "exogenous_monthly",
    )
    _validate_unique_rows(exogenous_df, [date_column], "exogenous_monthly")
    # Restrict to declared base_columns before lagging to avoid carrying unintended regressors downstream.
    exogenous_df = (
        exogenous_df[[date_column, *base_columns]]
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    logger.info(
        "Building monthly exogenous features from exogenous_monthly shape=%s, date range=[%s, %s].",
        exogenous_df.shape,
        exogenous_df[date_column].min().date(),
        exogenous_df[date_column].max().date(),
    )

    # Add lagged features based on the configured base columns and lags.
    exogenous_features, lagged_columns = _add_lag_features(
        exogenous_df,
        columns=base_columns,
        lags=lags,
        date_column=date_column,
    )
    # Log the number of nulls introduced by lagging.
    null_counts = exogenous_features[lagged_columns].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    logger.info(
        "Generated monthly exogenous lag columns=%s.",
        lagged_columns,
    )
    logger.info(
        "Null values introduced by lagged monthly exogenous features:\n%s",
        null_counts.to_string() if not null_counts.empty else "No lag nulls detected.",
    )

    exogenous_features = exogenous_features.sort_values(date_column).reset_index(
        drop=True
    )
    logger.info(
        "Built monthly exogenous features shape=%s, date range=[%s, %s].",
        exogenous_features.shape,
        exogenous_features[date_column].min().date(),
        exogenous_features[date_column].max().date(),
    )
    return exogenous_features


def build_monthly_prophet_features(
    demand_monthly: pd.DataFrame,
    monthly_calendar_features: pd.DataFrame,
    monthly_exogenous_features: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Join monthly demand, calendar, and exogenous features into the final Prophet feature table.

    Performs two sequential left joins on ``date_column``:

    1. demand → calendar (many-to-one): attaches deterministic calendar features to each
       demand row.
    2. result → exogenous (many-to-one): attaches exogenous regressors and their lags.

    Months with demand but no matching exogenous or calendar rows produce null regressor
    values and are logged as warnings. The downstream model-input preparation node
    controls whether those rows are dropped.

    Args:
        demand_monthly: Monthly demand DataFrame. Must contain ``date_column``,
            ``sku_column``, and ``target_column``.
        monthly_calendar_features: Calendar feature DataFrame produced by
            ``build_monthly_calendar_features``. Must contain ``date_column`` and all
            columns in ``_CALENDAR_FEATURE_COLUMNS``. Must be unique by ``date_column``.
        monthly_exogenous_features: Exogenous feature DataFrame produced by
            ``build_monthly_exogenous_features``. Must contain ``date_column`` and all
            ``base_columns`` declared in ``parameters``. Must be unique by ``date_column``.
        parameters: Feature engineering parameters. Expected keys: ``date_column``,
            ``sku_column``, ``target_column``, and ``exogenous_features.base_columns``.

    Returns:
        Prophet feature DataFrame with one row per (SKU, month) and columns from all
        three input sources merged together. Sorted by ``(sku_column, date_column)``.

    Raises:
        ValueError: If required columns are missing from any input DataFrame, or if
            ``monthly_calendar_features`` or ``monthly_exogenous_features`` contain
            duplicate month keys, which would cause row fan-out in the join.
    """
    date_column = parameters["date_column"]
    sku_column = parameters["sku_column"]
    target_column = parameters["target_column"]
    base_exogenous_columns = parameters["exogenous_features"]["base_columns"]

    # Validate all required columns are present before joining to fail fast with clear errors.
    _validate_required_columns(
        demand_monthly,
        [date_column, sku_column, target_column],
        "demand_monthly",
    )
    _validate_required_columns(
        monthly_calendar_features,
        [date_column, *_CALENDAR_FEATURE_COLUMNS],
        "monthly_calendar_features",
    )
    _validate_required_columns(
        monthly_exogenous_features,
        [date_column, *base_exogenous_columns],
        "monthly_exogenous_features",
    )

    demand_df = _ensure_month_start_dates(demand_monthly, date_column, "demand_monthly")
    calendar_df = _ensure_month_start_dates(
        monthly_calendar_features,
        date_column,
        "monthly_calendar_features",
    )
    exogenous_df = _ensure_month_start_dates(
        monthly_exogenous_features,
        date_column,
        "monthly_exogenous_features",
    )

    _validate_unique_rows(demand_df, [sku_column, date_column], "demand_monthly")
    _validate_unique_rows(calendar_df, [date_column], "monthly_calendar_features")
    _validate_unique_rows(exogenous_df, [date_column], "monthly_exogenous_features")

    logger.info(
        "Joining monthly Prophet features with demand shape=%s, calendar shape=%s, exogenous shape=%s.",
        demand_df.shape,
        calendar_df.shape,
        exogenous_df.shape,
    )

    # Drop duplicates before the join to avoid unexpected fan-out.
    # The date columns are expected to be unique after validation.
    demand_months = pd.DatetimeIndex(
        demand_df[date_column].drop_duplicates().sort_values()
    )
    exogenous_months = pd.DatetimeIndex(
        exogenous_df[date_column].drop_duplicates().sort_values()
    )

    # Identify months with demand but missing exogenous variables, and vice versa.
    missing_exogenous_months = sorted(set(demand_months) - set(exogenous_months))
    extra_exogenous_months = sorted(set(exogenous_months) - set(demand_months))

    if missing_exogenous_months:
        logger.warning(
            "Months with demand but missing exogenous variables (%d): %s",
            len(missing_exogenous_months),
            _format_month_list(missing_exogenous_months),
        )
    if extra_exogenous_months:
        logger.warning(
            "Months with exogenous variables but no demand records (%d): %s",
            len(extra_exogenous_months),
            _format_month_list(extra_exogenous_months),
        )

    # Left joins preserve every demand observation while attaching deterministic features.
    # validate="many_to_one" guards against accidental fan-out if calendar/exogenous have duplicate months.
    prophet_features = demand_df.merge(
        calendar_df,
        on=date_column,
        how="left",
        validate="many_to_one",
    ).merge(
        exogenous_df,
        on=date_column,
        how="left",
        validate="many_to_one",
    )

    prophet_features = prophet_features.sort_values(
        [sku_column, date_column]
    ).reset_index(drop=True)
    generated_columns = [
        column for column in prophet_features.columns if column not in demand_df.columns
    ]

    logger.info(
        "Built monthly_prophet_features shape=%s, date range=[%s, %s], generated columns=%s.",
        prophet_features.shape,
        prophet_features[date_column].min().date(),
        prophet_features[date_column].max().date(),
        generated_columns,
    )
    return prophet_features
