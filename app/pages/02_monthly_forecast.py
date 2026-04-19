import pandas as pd
import streamlit as st

from shared.schemas import ForecastRecord  # noqa: F401

st.title("Monthly Forecast — Primary Layer")

st.info("""
    **Primary analytical layer.** This is the model that aligns with the main business decision
    horizon and the primary target for accuracy improvement from the current baseline of 82.3%
    to ≥ 85% forecast accuracy. All evaluation and stakeholder reporting is centred on this layer.
    """)

model_family = st.selectbox(
    "Model family",
    options=["SARIMAX", "Prophet", "CatBoost"],
)

forecast_type = st.radio(
    "Forecast type",
    options=["Raw forecast", "Reconciled forecast"],
    horizontal=True,
)

_parquet_map = {
    "Raw forecast": "data/07_model_output/monthly_forecast_raw.parquet",
    "Reconciled forecast": "data/07_model_output/monthly_forecast_reconciled.parquet",
}
st.info(
    f"Outputs load from `{_parquet_map[forecast_type]}` once the Kedro pipeline has been run."
)

_empty: pd.DataFrame = pd.DataFrame(
    columns=[
        "ds",
        "y_actual",
        "y_forecast",
        "y_lower",
        "y_upper",
        "model_name",
        "horizon_step",
    ]
)
st.dataframe(_empty, use_container_width=True)
