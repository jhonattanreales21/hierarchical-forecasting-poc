import streamlit as st

st.set_page_config(
    page_title="Demand Forecast POC",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Demand Forecast POC")
st.markdown(
    "Temporal hierarchical forecasting for a critical SKU. "
    "Select a page from the sidebar to explore the results."
)
