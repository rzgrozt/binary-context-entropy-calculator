"""Static context-depth entropy chart construction for VMM evidence."""

from dataclasses import dataclass
from typing import Literal, Protocol

import plotly.graph_objects as go
import streamlit as st
from plotly.basedatatypes import BaseFigure
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.plotly_chart import PlotlyState

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
    PLOTLY_NUMBER_FORMAT,
)
from binary_entropy.vmm_types import VMMRecordAnalysis


class _PlotlyChartRenderer(Protocol):
    def plotly_chart[ConfigValue](  # noqa: PLR0913
        self,
        figure_or_data: BaseFigure,
        *,
        key: str,
        width: Literal["stretch"],
        theme: None,
        on_select: Literal["ignore"],
        config: dict[str, ConfigValue],
    ) -> DeltaGenerator | PlotlyState: ...


_PLOTLY_RENDERER: _PlotlyChartRenderer = st


@dataclass(frozen=True, slots=True)
class VMMEntropyChartSpec:
    """Ordered available values and fixed visible chart semantics."""

    requested_depths: tuple[int, ...]
    entropies: tuple[float, ...]
    trace_name: str
    hover_template: str
    entropy_range: tuple[float, float]
    tick_format: str
    transition_duration: int
    static_plot: bool
    display_modebar: bool
    responsive: bool


def vmm_entropy_chart_spec(record: VMMRecordAnalysis) -> VMMEntropyChartSpec:
    """Collect unrounded entropy values while omitting unavailable depths."""
    points: list[tuple[int, float]] = []
    for row in sorted(record.depth_rows, key=lambda candidate: candidate.depth):
        entropy = row.predictive_entropy_bits
        if entropy is not None:
            points.append((row.depth, entropy))
    return VMMEntropyChartSpec(
        requested_depths=tuple(point[0] for point in points),
        entropies=tuple(point[1] for point in points),
        trace_name="Predictive entropy",
        hover_template=(
            "Requested depth %{x}<br>Predictive entropy %{y:.3f} bits<extra></extra>"
        ),
        entropy_range=(CHART_ENTROPY_MIN, CHART_ENTROPY_MAX),
        tick_format=PLOTLY_NUMBER_FORMAT,
        transition_duration=MOTION_DURATION,
        static_plot=True,
        display_modebar=False,
        responsive=True,
    )


def vmm_entropy_figure(record: VMMRecordAnalysis) -> go.Figure:
    """Build a named line-and-marker figure with no animation frames."""
    spec = vmm_entropy_chart_spec(record)
    trace = go.Scatter(
        x=spec.requested_depths,
        y=spec.entropies,
        name=spec.trace_name,
        mode="lines+markers",
        hovertemplate=spec.hover_template,
        line={"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        marker={
            "color": COLOR_SURFACE,
            "size": CHART_MARKER_SIZE,
            "line": {"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        },
        showlegend=True,
    )
    layout = go.Layout(
        autosize=True,
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"family": FONT_DATA, "color": COLOR_TEXT_SECONDARY},
        hovermode="closest",
        transition={"duration": spec.transition_duration},
        legend={"orientation": "h"},
        xaxis={
            "title": {"text": "Requested context depth"},
            "dtick": 1,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Predictive entropy (bits)"},
            "range": spec.entropy_range,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
            "tickformat": spec.tick_format,
        },
    )
    return go.Figure(data=(trace,), layout=layout, frames=())


def render_vmm_entropy_plot(record: VMMRecordAnalysis, *, key: str) -> None:
    """Render the VMM entropy figure with all Plotly interaction disabled."""
    spec = vmm_entropy_chart_spec(record)
    _ = _PLOTLY_RENDERER.plotly_chart(
        vmm_entropy_figure(record),
        key=key,
        width="stretch",
        theme=None,
        on_select="ignore",
        config={
            "staticPlot": spec.static_plot,
            "displayModeBar": spec.display_modebar,
            "responsive": spec.responsive,
        },
    )
