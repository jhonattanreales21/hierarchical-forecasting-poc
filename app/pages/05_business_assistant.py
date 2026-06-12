import pandas as pd
import streamlit as st

from shared.forecast_assistant import (
    answer_question,
    build_assistant_context,
    is_in_scope,
    merge_historical_inputs,
    rename_business_columns,
    transform_scenario_rows,
)
from shared.rag import FaissVectorStore, build_chunks_from_path
from shared.viz import add_event_lines, plot_forecast
from ui.components import render_empty_state, render_hero, render_section_header
from ui.styles import apply_global_styles
from utils.champion import extract_champion_identity, forecast_has_intervals
from utils.data_loaders import (
    load_champion_metadata,
    load_inference_metadata,
    load_model_selection_summary,
    load_monthly_forecast,
    load_monthly_modeling_data,
    load_monthly_modeling_data_full,
    load_raw_exogenous_data,
)
from utils.paths import ASSISTANT_UPLOADS, ASSISTANT_VECTORSTORE, CHAMPION_META
from utils.paths import DEFAULT_RAG_DOCUMENT

_RAG_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _available_rag_sources() -> list:
    """Return default and uploaded RAG documents available to the assistant."""
    sources = []
    if DEFAULT_RAG_DOCUMENT.exists():
        sources.append(DEFAULT_RAG_DOCUMENT)
    if ASSISTANT_UPLOADS.exists():
        sources.extend(
            path
            for path in ASSISTANT_UPLOADS.iterdir()
            if path.is_file() and path.suffix.lower() in _RAG_SUFFIXES
        )
    return sorted(sources, key=lambda path: path.stat().st_mtime, reverse=True)


def _latest_rag_source():
    """Return the newest available RAG source, preferring fresh uploads."""
    sources = _available_rag_sources()
    return sources[0] if sources else None


apply_global_styles()

render_hero(
    title="Forecast Assistant",
    subtitle=(
        "Ask in plain language across forecast outputs, scenario assumptions, "
        "and historical demand."
    ),
    eyebrow="Demand Forecasting Platform",
)

if not CHAMPION_META.exists():
    render_empty_state(
        title="Champion Model Not Found",
        message=(
            "No champion metadata was found. Run the monthly training and inference "
            "pipelines before using the assistant."
        ),
        status="warning",
    )
    st.stop()

vectorstore = FaissVectorStore(ASSISTANT_VECTORSTORE)
latest_rag_source = _latest_rag_source()

meta = load_champion_metadata()
inference_meta = load_inference_metadata()
identity = extract_champion_identity(
    meta, inference_meta, load_model_selection_summary()
)
actuals = load_monthly_modeling_data()
raw_exogenous = load_raw_exogenous_data()
historical_full = merge_historical_inputs(
    load_monthly_modeling_data_full(),
    raw_exogenous,
)

champion_id = identity.get("champion_id") or ""
future_fc = load_monthly_forecast(12)
if future_fc.empty:
    future_fc = load_monthly_forecast(6)
if future_fc.empty:
    future_fc = load_monthly_forecast(3)
if not future_fc.empty and not raw_exogenous.empty:
    future_fc = merge_historical_inputs(future_fc, raw_exogenous)

# ---------------------------------------------------------------------------
# Assistant interaction — wide chat with the knowledge panel on the right
# ---------------------------------------------------------------------------
ask_col, knowledge_col = st.columns([2.0, 1.0], gap="large")

with knowledge_col:
    render_section_header(
        "Assistant Knowledge",
        description=(
            "Documents uploaded on the Data Upload page ground historical "
            "explanations. Build the index from the latest uploaded document below."
        ),
    )

    latest_rag_source = _latest_rag_source()
    if latest_rag_source:
        source_label = latest_rag_source.name
        if vectorstore.is_stale_for(latest_rag_source):
            st.warning(f"Latest RAG source is not indexed yet: {source_label}")
            if st.button("Build latest RAG source", width="stretch"):
                with st.spinner(f"Indexing {source_label}..."):
                    chunks = build_chunks_from_path(latest_rag_source)
                    count = vectorstore.build(chunks, source_path=latest_rag_source)
                st.success(f"Indexed {count} chunks from {source_label}.")
        else:
            st.caption(f"Active RAG source: {source_label}")

    if vectorstore.exists():
        st.success("RAG index ready.")
    else:
        st.info("Build the default Recothrom RAG document or upload a newer document.")

with ask_col:
    render_section_header(
        "Ask The Assistant",
        description="Ask about a future month, a historical spike/drop, or active scenario assumptions.",
    )
    question = st.text_area(
        "Question",
        value="Why does the forecast change in June 2026?",
        height=160,
    )
    ask = st.button("Ask", type="primary", width="stretch")

    if not vectorstore.exists():
        st.info(
            "RAG index not built. Answers can use forecast and data artifacts, but historical business-document evidence is unavailable."
        )

    if ask:
        if not question.strip():
            st.warning("Enter a forecast explainability question first.")
        elif not is_in_scope(question):
            st.warning("I can only support forecast explainability questions.")
        else:
            with st.spinner("Retrieving context and preparing answer..."):
                rag_chunks = vectorstore.search(question, top_k=5)
                context = build_assistant_context(
                    question=question,
                    forecast=future_fc,
                    historical=historical_full,
                    metadata=meta,
                    rag_chunks=rag_chunks,
                )
                answer = answer_question(question, context)
            st.markdown(answer)

            with st.expander("Evidence sent to assistant"):
                st.write("Forecast output")
                st.dataframe(
                    rename_business_columns(future_fc.head(12)),
                    width="stretch",
                    hide_index=True,
                )
                st.write("Historical context")
                st.dataframe(
                    pd.DataFrame(context["historical_demand_exogenous_rows"]),
                    width="stretch",
                    hide_index=True,
                )
                st.write("Retrieved RAG chunks")
                for chunk in rag_chunks:
                    page = chunk.get("page_number")
                    source = chunk.get("source", "document")
                    st.caption(f"{source}" + (f", page {page}" if page else ""))
                    st.write(chunk.get("text", "")[:800])

st.divider()

# ---------------------------------------------------------------------------
# Forecast visuals — one selectable chart (avoids cramped side-by-side panels)
# ---------------------------------------------------------------------------
render_section_header(
    "Forecast Visuals",
    description="Pick a view: available history, the 12-month outlook, or the planned driver mix.",
)

_CHART_HISTORY = "Historical Demand"
_CHART_FORECAST = "12-Month Forecast"
_CHART_DRIVERS = "Driver Mix"
_chart_options = [_CHART_HISTORY, _CHART_FORECAST, _CHART_DRIVERS]

if hasattr(st, "segmented_control"):
    selected_chart = st.segmented_control(
        "Chart view",
        options=_chart_options,
        default=_CHART_FORECAST,
        selection_mode="single",
        key="assistant_chart_view",
        label_visibility="collapsed",
        width="stretch",
    )
else:
    selected_chart = st.radio(
        "Chart view",
        options=_chart_options,
        index=1,
        horizontal=True,
        key="assistant_chart_view",
        label_visibility="collapsed",
    )

selected_chart = selected_chart or _CHART_FORECAST
_empty_band = pd.DataFrame({"ds": [], "yhat": [], "yhat_lower": [], "yhat_upper": []})

if selected_chart == _CHART_HISTORY:
    if actuals.empty:
        st.warning("No historical demand is available yet.")
    else:
        fig = plot_forecast(
            actuals=actuals,
            test_forecast=_empty_band,
            future_forecast=_empty_band,
            title="Available Historical Monthly Demand",
        )
        # Mark months flagged by "limited"-type exogenous events, clipped to the
        # historical window so future flags do not stretch the axis.
        if not raw_exogenous.empty and "Date" in raw_exogenous.columns:
            hist_events = raw_exogenous[raw_exogenous["Date"] <= actuals["ds"].max()]
            fig = add_event_lines(
                fig,
                hist_events,
                event_columns={
                    "pfizer_limited": "Pfizer Limited",
                    "surgifoam_limited": "Surgifoam Limited",
                },
                date_col="Date",
            )
        st.plotly_chart(fig, width="stretch")

elif selected_chart == _CHART_FORECAST:
    if future_fc.empty:
        st.warning("No forecast is available yet.")
    else:
        recent_actuals = (
            actuals.sort_values("ds").tail(4)
            if not actuals.empty
            else pd.DataFrame({"ds": [], "y": []})
        )
        fig = plot_forecast(
            actuals=recent_actuals,
            test_forecast=_empty_band,
            future_forecast=future_fc,
            title="12-Month Forecast with Recent Demand",
            champion_id=champion_id,
            show_future_intervals=forecast_has_intervals(future_fc, identity),
        )
        for trace in fig.data:
            if trace.name == "Forecast (future)":
                trace.line.dash = "dash"
        st.plotly_chart(fig, width="stretch")

else:  # Driver Mix
    render_empty_state(
        title="Coming soon",
        message=(
            "Driver Mix will show how exogenous drivers and model signals contribute "
            "to each forecast period. This view is planned for an upcoming release."
        ),
        status="info",
    )

# ---------------------------------------------------------------------------
# Scenario assumptions sent to the assistant
# ---------------------------------------------------------------------------
scenario_rows = transform_scenario_rows(future_fc)
if not scenario_rows.empty:
    render_section_header(
        "Scenario Variables",
        description="Business-readable assumptions sent to the assistant.",
    )
    st.dataframe(scenario_rows, width="stretch", hide_index=True)
