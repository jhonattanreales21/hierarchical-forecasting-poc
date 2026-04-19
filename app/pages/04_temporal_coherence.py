import streamlit as st

st.title("Temporal Coherence")

st.markdown("""
    ### Monthly ↔ Weekly ↔ Daily Reconciliation

    Temporal coherence ensures that forecasts at different granularities are internally consistent:
    the sum of weekly forecasts within a month aligns with the monthly forecast, and daily
    allocations (when enabled) sum to the weekly total.

    The **primary reconciliation target** for this PoC is **monthly ↔ weekly**. Daily allocation
    is an optional exploratory extension and is currently disabled by parameter.
    """)

st.code(
    """\
Monthly  <->  Weekly  <->  Daily (disabled)
[primary]     [anchor]     [optional extension]""",
    language=None,
)

st.warning(
    "Daily allocation is currently **disabled**. "
    "It can be re-enabled by setting `daily_allocation.enabled: true` "
    "in `conf/base/parameters/forecast_inference.yml`."
)

st.caption(
    "The reconciliation method is controlled by the `reconciliation.yml` parameter "
    "(default: `mint_shrink`)."
)
