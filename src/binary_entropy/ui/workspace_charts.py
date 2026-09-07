"""Bounded static Plotly charts for aggregate workspace overviews."""

from typing import Final, Literal, Protocol

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
    COLOR_SECONDARY_SERIES,
    COLOR_SURFACE,
    COLOR_TEXT_SECONDARY,
    FONT_DATA,
    MOTION_DURATION,
    PLOTLY_NUMBER_FORMAT,
)
from binary_entropy.ui.workspace_agreement import AgreementSummary
from binary_entropy.ui.workspace_metrics import entropy_points
from binary_entropy.workbench import WorkbenchResult

AGGREGATE_CHART_LIMIT: Final = 50


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


def aggregate_entropy_figure(result: WorkbenchResult) -> go.Figure:
    """Plot at most 50 stored final entropy values in record order."""
    points = entropy_points(result)[:AGGREGATE_CHART_LIMIT]
    trace = go.Scatter(
        x=tuple(point.sequence_id for point in points),
        y=tuple(point.entropy_bits for point in points),
        name="Final entropy",
        mode="lines+markers",
        hovertemplate=(
            "Sequence %{x}<br>Final entropy %{y:.3f} bits<extra></extra>"
        ),
        line={"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        marker={"color": COLOR_INTERACTIVE, "size": CHART_MARKER_SIZE},
        showlegend=False,
    )
    return go.Figure(
        data=(trace,),
        layout=_layout(
            "Sequence identifier",
            "Final entropy (bits)",
            entropy_axis=True,
        ),
        frames=(),
    )


def agreement_figure(summary: AgreementSummary) -> go.Figure:
    """Plot predictive agreement status counts without Shannon results."""
    labels = (
        "Agreement",
        "Disagreement",
        "Single predictive method",
        "Unavailable",
    )
    counts = (
        summary.agreement_count,
        summary.disagreement_count,
        summary.single_method_count,
        summary.unavailable_count,
    )
    trace = go.Bar(
        x=labels,
        y=counts,
        name="Selected records",
        hovertemplate="%{x}<br>%{y} records<extra></extra>",
        marker={
            "color": (
                COLOR_INTERACTIVE,
                COLOR_SECONDARY_SERIES,
                COLOR_TEXT_SECONDARY,
                COLOR_BORDER_SUBTLE,
            )
        },
        showlegend=False,
    )
    return go.Figure(
        data=(trace,),
        layout=_layout("Predictive status", "Records", entropy_axis=False),
        frames=(),
    )


def render_static_figure(figure: go.Figure, *, key: str) -> None:
    """Render a responsive Plotly figure with interaction and motion disabled."""
    _ = _PLOTLY_RENDERER.plotly_chart(
        figure,
        key=key,
        width="stretch",
        theme=None,
        on_select="ignore",
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "responsive": True,
        },
    )


def _layout(x_title: str, y_title: str, *, entropy_axis: bool) -> go.Layout:
    if entropy_axis:
        yaxis = {
            "title": y_title,
            "range": (CHART_ENTROPY_MIN, CHART_ENTROPY_MAX),
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
            "tickformat": PLOTLY_NUMBER_FORMAT,
        }
    else:
        yaxis = {
            "title": y_title,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        }
    return go.Layout(
        autosize=True,
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"family": FONT_DATA, "color": COLOR_TEXT_SECONDARY},
        hovermode="closest",
        transition={"duration": MOTION_DURATION},
        xaxis={
            "title": x_title,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        },
        yaxis=yaxis,
    )
