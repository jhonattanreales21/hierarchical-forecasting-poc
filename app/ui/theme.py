"""Central design tokens for the Hierarchical Demand Forecasting app.

Import these constants in other UI modules to ensure consistency.
Do not hardcode color or spacing values in components or styles — use tokens here.
"""

COLORS: dict[str, str] = {
    "primary_navy": "#001F5B",
    "corporate_blue": "#0057B8",
    "deep_blue": "#003B7A",
    "light_blue_bg": "#EAF2FF",
    "accent_cyan": "#00A3E0",
    "text_dark": "#1F2937",
    "text_muted": "#6B7280",
    "text_light": "#9CA3AF",
    "page_bg": "#F5F7FA",
    "card_bg": "#FFFFFF",
    "border": "#E5E7EB",
    "warning_bg": "#FFF7D6",
    "warning_border": "#F59E0B",
    "warning_text": "#92400E",
    "success_bg": "#E8F7EF",
    "success_border": "#10B981",
    "success_text": "#065F46",
    "danger_bg": "#FDECEC",
    "danger_border": "#EF4444",
    "danger_text": "#991B1B",
    "sidebar_bg": "#FFFFFF",
}

TYPOGRAPHY: dict[str, str] = {
    "size_xs": "0.72rem",
    "size_sm": "0.875rem",
    "size_base": "1rem",
    "size_lg": "1.15rem",
    "size_xl": "1.4rem",
    "size_2xl": "1.8rem",
    "size_3xl": "2.2rem",
    "weight_normal": "400",
    "weight_medium": "500",
    "weight_semibold": "600",
    "weight_bold": "700",
}

SPACING: dict[str, str] = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
    "2xl": "3rem",
}

RADIUS: dict[str, str] = {
    "sm": "6px",
    "md": "10px",
    "lg": "14px",
    "pill": "999px",
}

SHADOWS: dict[str, str] = {
    "subtle": "0 1px 3px rgba(0, 0, 0, 0.06)",
    "card": "0 2px 6px rgba(0, 0, 0, 0.08)",
    "elevated": "0 4px 12px rgba(0, 0, 0, 0.10)",
}
