import streamlit as st

st.title("Weekly Anchor Forecast — Secondary Layer")

st.info("""
    **Secondary operational layer.** 14-week horizon, useful for short-term interpretability and
    as an operational complement to the monthly layer. Results here should be read alongside the
    monthly forecast — not in isolation.
    """)

forecast_type = st.radio(
    "Forecast type",
    options=["Raw forecast", "Reconciled forecast"],
    horizontal=True,
)

_parquet_map = {
    "Raw forecast": "data/07_model_output/weekly_forecast_raw.parquet",
    "Reconciled forecast": "data/07_model_output/weekly_forecast_reconciled.parquet",
}
st.info(
    f"Outputs load from `{_parquet_map[forecast_type]}` once the Kedro pipeline has been run."
)

st.metric(label="Horizon", value="14 weeks")

model_family = st.selectbox(
    "Model family",
    options=["SARIMAX", "Prophet", "CatBoost"],
)

_chart_placeholder = st.empty()
st.caption(
    "A Plotly line chart showing the weekly point forecast and prediction interval "
    "will be rendered here once forecast outputs are available."
)
