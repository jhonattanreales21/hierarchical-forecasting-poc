import pandas as pd
import streamlit as st

from shared.schemas import BacktestResult  # noqa: F401

st.title("Evaluation Report")

_columns = ["model_name", "granularity", "mape", "rmse", "mase", "n_folds", "horizon"]
_empty: pd.DataFrame = pd.DataFrame(columns=_columns)

st.subheader("Monthly Models")
st.dataframe(_empty, use_container_width=True)
st.caption("Primary evaluation focus — accuracy target: ≥ 85% (MAPE improvement from 82.3% baseline).")

st.subheader("Weekly Models")
st.dataframe(_empty, use_container_width=True)

st.info(
    "These tables will populate after the evaluation pipeline runs and outputs results to "
    "`data/08_reporting/evaluation_report.parquet`."
)
