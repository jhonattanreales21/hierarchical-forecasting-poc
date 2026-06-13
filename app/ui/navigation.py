"""Top horizontal navigation bar for the Demand Forecasting app.

Call render_top_navbar(current_page) at the top of each functional page,
immediately after apply_global_styles(). The current_page argument must
match one of the keys in _PAGES exactly.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from ui.theme import COLORS

_PAGES: dict[str, str] = {
    "Data Upload": "pages/01_Data_Upload.py",
    "Descriptive Analysis": "pages/02_Descriptive_Analysis.py",
    "Monthly Forecast": "pages/03_Monthly_Forecast.py",
    "Evaluation Report": "pages/04_Evaluation_Report.py",
    "Business Assistant": "pages/05_business_assistant.py",
}

_ICONS: list[str] = [
    "cloud-upload",
    "bar-chart-line",
    "graph-up",
    "clipboard-data",
    "robot",
]

_PAGE_NAMES: list[str] = list(_PAGES.keys())


def render_top_navbar(current_page: str) -> None:
    """Render the horizontal top navigation bar and handle page switching.

    Args:
        current_page: Display name of the currently active page.
            Must be one of: "Data Upload", "Descriptive Analysis",
            "Monthly Forecast", "Evaluation Report", "Business Assistant".
    """
    default_index = (
        _PAGE_NAMES.index(current_page) if current_page in _PAGE_NAMES else 0
    )

    selected = option_menu(
        menu_title=None,
        options=_PAGE_NAMES,
        icons=_ICONS,
        default_index=default_index,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0.4rem 0.5rem",
                "background-color": COLORS["card_bg"],
                "border-bottom": f"1px solid {COLORS['border']}",
                "border-radius": "0",
                "margin-bottom": "0.75rem",
                "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
            },
            "icon": {
                "color": COLORS["corporate_blue"],
                "font-size": "0.88rem",
            },
            "nav-link": {
                "font-size": "0.875rem",
                "font-weight": "500",
                "color": COLORS["text_dark"],
                "padding": "0.45rem 0.9rem",
                "border-radius": "6px",
                "--hover-color": COLORS["light_blue_bg"],
            },
            "nav-link-selected": {
                "background-color": COLORS["primary_navy"],
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    if selected != current_page:
        st.switch_page(_PAGES[selected])
