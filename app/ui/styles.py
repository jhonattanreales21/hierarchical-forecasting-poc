"""Global CSS styles for the Hierarchical Demand Forecasting Streamlit app.

Call apply_global_styles() once at the top of each page to inject shared CSS.
All custom component classes used in components.py are defined here.
"""

import streamlit as st

from ui.theme import COLORS


def apply_global_styles() -> None:
    """Inject global CSS overrides and custom component classes into the Streamlit app."""
    bg = COLORS["page_bg"]
    sidebar_bg = COLORS["sidebar_bg"]
    border = COLORS["border"]
    navy = COLORS["primary_navy"]
    blue = COLORS["corporate_blue"]
    deep_blue = COLORS["deep_blue"]
    light_blue_bg = COLORS["light_blue_bg"]
    accent_cyan = COLORS["accent_cyan"]
    text_dark = COLORS["text_dark"]
    text_muted = COLORS["text_muted"]
    text_light = COLORS["text_light"]
    card_bg = COLORS["card_bg"]
    warning_bg = COLORS["warning_bg"]
    warning_border = COLORS["warning_border"]
    warning_text = COLORS["warning_text"]
    success_bg = COLORS["success_bg"]
    success_border = COLORS["success_border"]
    success_text = COLORS["success_text"]
    danger_bg = COLORS["danger_bg"]
    danger_text = COLORS["danger_text"]

    css = f"""
    <style>
    /* ── Page & sidebar background ─────────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        border-right: 1px solid {border};
    }}

    /* ── Main container ────────────────────────────────────────────────────── */
    .block-container {{
        padding-top: 1.75rem !important;
        padding-bottom: 2.5rem !important;
    }}

    /* ── Streamlit native headings override ────────────────────────────────── */
    h1 {{
        color: {navy} !important;
        font-weight: 700 !important;
    }}
    h2, h3 {{
        color: {deep_blue} !important;
        font-weight: 600 !important;
    }}

    /* ── Page header ───────────────────────────────────────────────────────── */
    .app-page-header {{
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid {blue};
    }}
    .app-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {blue};
        margin-bottom: 0.25rem;
    }}
    .app-page-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {navy};
        margin: 0 0 0.2rem 0;
        line-height: 1.2;
    }}
    .app-page-subtitle {{
        font-size: 0.95rem;
        color: {text_muted};
        margin: 0;
        line-height: 1.5;
    }}

    /* ── Hero block ────────────────────────────────────────────────────────── */
    .app-hero {{
        background: linear-gradient(135deg, {navy} 0%, {deep_blue} 100%);
        color: #FFFFFF;
        padding: 1.75rem 2.1rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 31, 91, 0.16);
    }}
    .app-hero .app-eyebrow,
    .app-hero-eyebrow {{
        color: {accent_cyan};
    }}
    .app-hero .app-page-title,
    .app-hero-title {{
        color: #FFFFFF !important;
        font-size: 2rem;
        margin-bottom: 0.35rem;
    }}
    .app-hero .app-page-subtitle,
    .app-hero-subtitle {{
        color: rgba(255, 255, 255, 0.82);
        font-size: 1.05rem;
    }}

    /* ── Card ──────────────────────────────────────────────────────────────── */
    .app-card {{
        background: {card_bg};
        border-radius: 10px;
        border: 1px solid {border};
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }}

    /* ── Info banner ───────────────────────────────────────────────────────── */
    .app-info-banner {{
        background: {light_blue_bg};
        border-left: 4px solid {blue};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
    }}
    .app-info-banner .app-banner-title {{
        font-weight: 700;
        color: {deep_blue};
        margin-bottom: 0.15rem;
        font-size: 0.92rem;
    }}
    .app-info-banner .app-banner-body {{
        color: {text_dark};
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.55;
    }}

    /* ── Warning banner ────────────────────────────────────────────────────── */
    .app-warning-banner {{
        background: {warning_bg};
        border-left: 4px solid {warning_border};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
    }}
    .app-warning-banner .app-banner-title {{
        font-weight: 700;
        color: {warning_text};
        margin-bottom: 0.15rem;
        font-size: 0.92rem;
    }}
    .app-warning-banner .app-banner-body {{
        color: {text_dark};
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.55;
    }}

    /* ── Success banner ────────────────────────────────────────────────────── */
    .app-success-banner {{
        background: {success_bg};
        border-left: 4px solid {success_border};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
    }}
    .app-success-banner .app-banner-title {{
        font-weight: 700;
        color: {success_text};
        margin-bottom: 0.15rem;
        font-size: 0.92rem;
    }}
    .app-success-banner .app-banner-body {{
        color: {text_dark};
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.55;
    }}

    /* ── KPI card ──────────────────────────────────────────────────────────── */
    .app-kpi-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 100%;
    }}
    .app-kpi-label {{
        font-size: 0.76rem;
        font-weight: 600;
        color: {text_muted};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }}
    .app-kpi-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {navy};
        line-height: 1.15;
    }}
    .app-kpi-help {{
        font-size: 0.76rem;
        color: {text_light};
        margin-top: 0.3rem;
    }}
    .app-kpi-card.status-success .app-kpi-value {{
        color: {success_text};
    }}
    .app-kpi-card.status-warning .app-kpi-value {{
        color: {warning_text};
    }}
    .app-kpi-card.status-danger .app-kpi-value {{
        color: {danger_text};
    }}

    /* ── Status badge ──────────────────────────────────────────────────────── */
    .app-status-badge {{
        display: inline-block;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        text-transform: uppercase;
        line-height: 1.4;
    }}
    .app-status-badge.info {{
        background: {light_blue_bg};
        color: {deep_blue};
    }}
    .app-status-badge.success {{
        background: {success_bg};
        color: {success_text};
    }}
    .app-status-badge.warning {{
        background: {warning_bg};
        color: {warning_text};
    }}
    .app-status-badge.danger {{
        background: {danger_bg};
        color: {danger_text};
    }}

    /* ── Section header ────────────────────────────────────────────────────── */
    .app-section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {navy};
        border-bottom: 1px solid {border};
        padding-bottom: 0.45rem;
        margin-bottom: 0.35rem;
    }}
    .app-section-description {{
        font-size: 0.88rem;
        color: {text_muted};
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }}

    /* ── Monthly Forecast page polish ─────────────────────────────────────── */
    .monthly-section-gap {{
        height: 0.55rem;
    }}
    .monthly-toolbar-header {{
        margin-bottom: 0.65rem;
    }}
    .monthly-toolbar-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {blue};
        margin-bottom: 0.2rem;
    }}
    .monthly-toolbar-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {navy};
        margin: 0 0 0.2rem 0;
    }}
    .monthly-toolbar-copy {{
        font-size: 0.88rem;
        color: {text_muted};
        margin: 0;
        line-height: 1.45;
    }}
    .monthly-toolbar-note {{
        font-size: 0.8rem;
        color: {text_muted};
        margin: 0.55rem 0 0 0;
        line-height: 1.4;
    }}
    .monthly-toolbar-note strong {{
        color: {deep_blue};
    }}
    .monthly-chart-panel-header {{
        margin-bottom: 0.75rem;
    }}
    .monthly-chart-panel-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {blue};
        margin-bottom: 0.2rem;
    }}
    .monthly-chart-panel-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {navy};
        margin: 0 0 0.25rem 0;
    }}
    .monthly-chart-panel-copy {{
        font-size: 0.9rem;
        color: {text_muted};
        margin: 0;
        line-height: 1.5;
    }}
    .monthly-chart-panel-meta {{
        font-size: 0.8rem;
        color: {text_muted};
        margin-top: 0.45rem;
        line-height: 1.45;
    }}
    .st-key-monthly-horizon-toolbar,
    .st-key-monthly-chart-panel,
    .st-key-descriptive-controls {{
        background: {card_bg};
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }}
    .st-key-monthly-horizon-toolbar [data-testid="stSegmentedControl"] {{
        margin-top: 0.35rem;
    }}
    .st-key-monthly-horizon-toolbar [data-testid="stRadio"] {{
        margin-top: 0.25rem;
    }}
    .st-key-monthly-chart-panel [data-testid="stPlotlyChart"] {{
        margin-top: 0.25rem;
    }}

    /* ── Muted text ────────────────────────────────────────────────────────── */
    .app-muted-text {{
        color: {text_muted};
        font-size: 0.875rem;
        line-height: 1.5;
    }}

    /* ── Empty state ───────────────────────────────────────────────────────── */
    .app-empty-state {{
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: {card_bg};
        border: 1px dashed {border};
        border-radius: 10px;
        margin: 1rem 0;
    }}
    .app-empty-state-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {text_dark};
        margin-bottom: 0.45rem;
    }}
    .app-empty-state-body {{
        font-size: 0.88rem;
        color: {text_muted};
        line-height: 1.55;
    }}

    /* ── Capability block (placeholder pages) ──────────────────────────────── */
    .app-capability-block {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    .app-capability-block-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {navy};
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .app-capability-block-description {{
        font-size: 0.88rem;
        color: {text_muted};
        margin-bottom: 0.75rem;
        line-height: 1.55;
    }}
    .app-capability-block-list {{
        font-size: 0.84rem;
        color: {text_dark};
        margin: 0;
        padding-left: 1.2rem;
    }}
    .app-capability-block-list li {{
        margin-bottom: 0.2rem;
        line-height: 1.5;
    }}
    .app-capability-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {text_muted};
        margin-bottom: 0.3rem;
    }}

    /* ── Landing page nav cards ─────────────────────────────────────────────── */
    .app-nav-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 100%;
        margin-bottom: 0.75rem;
    }}
    .app-nav-card-title {{
        font-size: 0.9rem;
        font-weight: 700;
        color: {navy};
        margin-bottom: 0.25rem;
    }}
    .app-nav-card-description {{
        font-size: 0.82rem;
        color: {text_muted};
        line-height: 1.45;
        margin-bottom: 0.5rem;
    }}

    /* ── Chat preview (Business Assistant page) ─────────────────────────────── */
    .app-chat-shell {{
        padding: 1rem 1.1rem;
        border-radius: 10px;
        background: {card_bg};
        border: 1px solid {border};
        margin-bottom: 1rem;
    }}
    .app-chat-bubble {{
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        line-height: 1.55;
        font-size: 0.88rem;
    }}
    .app-chat-bubble.user {{
        background: {light_blue_bg};
        border: 1px solid rgba(0, 87, 184, 0.18);
        margin-left: 12%;
        color: {text_dark};
    }}
    .app-chat-bubble.assistant {{
        background: {card_bg};
        border: 1px solid {border};
        margin-right: 12%;
        color: {text_dark};
    }}
    .app-chat-label {{
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_muted};
        margin-bottom: 0.35rem;
    }}

    /* ── Readiness map card (Business Assistant page) ────────────────────────── */
    .app-readiness-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .app-readiness-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }}
    .app-readiness-title {{
        font-weight: 700;
        color: {navy};
        font-size: 0.9rem;
    }}
    .app-readiness-copy {{
        color: {text_muted};
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
