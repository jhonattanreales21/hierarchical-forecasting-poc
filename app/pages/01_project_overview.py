import streamlit as st

from shared.schemas import TemporalGranularity

st.title("Project Overview")

st.info("""
    **Temporal Hierarchy**

    This PoC uses a strictly temporal forecasting hierarchy:

    - **Monthly** — Primary analytical and decision layer. Main output for stakeholder decisions.
      Central benchmark aligned with the business decision horizon.
    - **Weekly** — Secondary enhancement layer. Valuable operational complement, but must not
      compromise the quality of the monthly layer.
    - **Daily** — Low-priority exploratory extension. Currently **disabled by default**
      (`daily_allocation.enabled = false`). Not a core deliverable.
    """)

st.subheader("Model Candidates")

st.markdown("""
| Model      | Description                                                              | Primary Layer      |
|------------|--------------------------------------------------------------------------|--------------------|
| SARIMAX    | Structured statistical baseline with seasonal and exogenous components   | Monthly            |
| Prophet    | Existing benchmark; handles seasonality and trend changes robustly       | Monthly & Weekly   |
| CatBoost   | Main tabular candidate with full exogenous variable support              | Monthly & Weekly   |
""")

st.subheader("Granularity Levels")
st.write([g.value for g in TemporalGranularity])

st.caption("Forecast horizons — Monthly: 1 / 2 / 3 months | Weekly: 4 / 9 / 14 weeks")
