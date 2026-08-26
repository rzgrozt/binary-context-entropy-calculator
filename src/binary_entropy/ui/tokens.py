"""Python-visible design tokens and display-only numeric formatting."""

import math
from typing import Final

COLOR_SURFACE: Final = "#171c21"
COLOR_SURFACE_RAISED: Final = "#20272e"
COLOR_TEXT_PRIMARY: Final = "#f4f7fa"
COLOR_TEXT_SECONDARY: Final = "#c1cbd4"
COLOR_BORDER_SUBTLE: Final = "#3a4652"
COLOR_INTERACTIVE: Final = "#4cc9f0"
COLOR_SECONDARY_SERIES: Final = "#b7c4d1"
FONT_DATA: Final = '"IBM Plex Sans", "Inter", "Source Sans 3", system-ui, sans-serif'
CHART_LINE_WIDTH: Final = 2
CHART_MARKER_SIZE: Final = 8
CHART_ENTROPY_MIN: Final = 0.0
CHART_ENTROPY_MAX: Final = 1.0
MOTION_DURATION: Final = 0
UI_DECIMALS: Final = 3
UI_NUMBER_FORMAT: Final = "%.3f"
PLOTLY_NUMBER_FORMAT: Final = ".3f"


def format_ui_decimal(value: float) -> str:
    """Format a visible value without changing calculations or exports."""
    if math.isinf(value):
        return "infinity" if value > 0.0 else "-infinity"
    if math.isnan(value):
        return "unavailable"
    return f"{value:.{UI_DECIMALS}f}"
