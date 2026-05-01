"""Reusable UI components for the Hierarchical Demand Forecasting app.

All components render HTML via st.markdown. They receive already-formatted values
and have no knowledge of business logic, data files, or artifact paths.

Usage:
    from ui.components import render_page_header, render_info_banner
"""

from html import escape

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """Render a styled page header with optional eyebrow label and subtitle.

    Args:
        title: Primary page title.
        subtitle: Supporting description shown below the title.
        eyebrow: Small uppercase label shown above the title (e.g. "Monthly Layer").
    """
    eyebrow_html = (
        f'<div class="app-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    )
    subtitle_html = (
        f'<p class="app-page-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="app-page-header">
            {eyebrow_html}
            <h1 class="app-page-title">{escape(title)}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    title: str,
    subtitle: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """Render a dark navy hero banner for prominent page introductions.

    Args:
        title: Primary headline shown in white.
        subtitle: Supporting text shown in muted white.
        eyebrow: Small accent-colored label shown above the title.
    """
    eyebrow_html = (
        f'<div class="app-eyebrow app-hero-eyebrow">{escape(eyebrow)}</div>'
        if eyebrow
        else ""
    )
    subtitle_html = (
        f'<p class="app-page-subtitle app-hero-subtitle">{escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="app-hero">
            {eyebrow_html}
            <h1 class="app-page-title app-hero-title">{escape(title)}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_banner(message: str, title: str | None = None) -> None:
    """Render a blue informational banner.

    Args:
        message: Body text of the banner.
        title: Optional bold title shown above the message.
    """
    title_html = f'<div class="app-banner-title">{escape(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="app-info-banner">
            {title_html}
            <p class="app-banner-body">{escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_banner(message: str, title: str | None = None) -> None:
    """Render an amber warning banner.

    Args:
        message: Body text of the banner.
        title: Optional bold title shown above the message.
    """
    title_html = f'<div class="app-banner-title">{escape(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="app-warning-banner">
            {title_html}
            <p class="app-banner-body">{escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success_banner(message: str, title: str | None = None) -> None:
    """Render a green success banner.

    Args:
        message: Body text of the banner.
        title: Optional bold title shown above the message.
    """
    title_html = f'<div class="app-banner-title">{escape(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="app-success-banner">
            {title_html}
            <p class="app-banner-body">{escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    help_text: str | None = None,
    status: str | None = None,
) -> None:
    """Render a standalone KPI card with a label, large value, and optional details.

    Args:
        label: Short uppercase metric label (e.g. "Test MAPE").
        value: Pre-formatted metric value (e.g. "12.3%").
        help_text: Optional small text shown below the value.
        status: Optional visual modifier — "success", "warning", or "danger".
    """
    status_class = (
        f" status-{status}" if status in ("success", "warning", "danger") else ""
    )
    help_html = (
        f'<div class="app-kpi-help">{escape(help_text)}</div>' if help_text else ""
    )
    st.markdown(
        f"""
        <div class="app-kpi-card{status_class}">
            <div class="app-kpi-label">{escape(label)}</div>
            <div class="app-kpi-value">{escape(value)}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, status: str = "info") -> None:
    """Render a small inline status badge/pill.

    Args:
        label: Text displayed inside the badge.
        status: Visual style — "info" (default), "success", "warning", or "danger".
    """
    valid_statuses = {"info", "success", "warning", "danger"}
    badge_status = status if status in valid_statuses else "info"
    st.markdown(
        f'<span class="app-status-badge {badge_status}">{escape(label)}</span>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None) -> None:
    """Render a styled section separator with title and optional description.

    Args:
        title: Section title shown with a bottom border.
        description: Optional muted description shown below the title.
    """
    description_html = (
        f'<div class="app-section-description">{escape(description)}</div>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <div class="app-section-title">{escape(title)}</div>
        {description_html}
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    message: str,
    status: str = "info",
) -> None:
    """Render a centered empty state card with a title and descriptive message.

    Args:
        title: Bold headline for the empty state.
        message: Supporting explanation or next-step instruction.
        status: Visual accent — "info", "warning", or "danger" (affects border color).
    """
    border_colors = {
        "info": "#0057B8",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "success": "#10B981",
    }
    border_color = border_colors.get(status, border_colors["info"])
    st.markdown(
        f"""
        <div class="app-empty-state" style="border-color: {border_color};">
            <div class="app-empty-state-title">{escape(title)}</div>
            <div class="app-empty-state-body">{escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
