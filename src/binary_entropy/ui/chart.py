"""Static Plotly construction for prefix predictive entropy."""

from dataclasses import dataclass

import plotly.graph_objects as go

from binary_entropy.domain import SequenceAnalysis
from binary_entropy.ui.tokens import (
    CHART_ENTROPY_MAX,
    CHART_ENTROPY_MIN,
    CHART_LINE_WIDTH,
    CHART_MARKER_SIZE,
    COLOR_BORDER_SUBTLE,
    COLOR_INTERACTIVE,
    COLOR_SURFACE,
    COLOR_TEXT_SECONDARY,
    FONT_DATA,
    MOTION_DURATION,
)


@dataclass(frozen=True, slots=True)
class EntropyChartSpec:
    """Exact chart data and interaction contract independent of rendering."""

    depths: tuple[int, ...]
    entropies: tuple[float, ...]
    hover_template: str
    entropy_range: tuple[float, float]
    transition_duration: int


def entropy_chart_spec(analysis: SequenceAnalysis) -> EntropyChartSpec:
    """Derive ordered chart values from prefix results."""
    return EntropyChartSpec(
        depths=tuple(row.depth for row in analysis.rows),
        entropies=tuple(row.entropy_bits for row in analysis.rows),
        hover_template=(
            "Depth %{x}<br>Predictive entropy %{y:.12f} bits<extra></extra>"
        ),
        entropy_range=(CHART_ENTROPY_MIN, CHART_ENTROPY_MAX),
        transition_duration=MOTION_DURATION,
    )


def entropy_figure(analysis: SequenceAnalysis) -> go.Figure:
    """Build an ordered line-and-marker figure with exact hover values."""
    spec = entropy_chart_spec(analysis)
    trace = go.Scatter(
        x=spec.depths,
        y=spec.entropies,
        mode="lines+markers",
        hovertemplate=spec.hover_template,
        line={"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        marker={
            "color": COLOR_SURFACE,
            "size": CHART_MARKER_SIZE,
            "line": {"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        },
        showlegend=False,
    )
    layout = go.Layout(
        autosize=True,
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"family": FONT_DATA, "color": COLOR_TEXT_SECONDARY},
        hovermode="closest",
        transition={"duration": spec.transition_duration},
        xaxis={
            "title": {"text": "Context depth"},
            "dtick": 1,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Predictive entropy (bits)"},
            "range": spec.entropy_range,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        },
    )
    return go.Figure(data=(trace,), layout=layout, frames=())
