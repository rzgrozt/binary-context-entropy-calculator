"""Static Plotly construction for prefix predictive entropy."""

from dataclasses import dataclass

import plotly.graph_objects as go

from binary_entropy.domain import SequenceAnalysis, float_values
from binary_entropy.markov_types import MarkovPrefixResult, MarkovRecordAnalysis
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


@dataclass(frozen=True, slots=True)
class EntropyChartSpec:
    """Exact chart data and interaction contract independent of rendering."""

    depths: tuple[int, ...]
    entropies: tuple[float, ...]
    hover_template: str
    entropy_range: tuple[float, float]
    transition_duration: int
    tick_format: str


@dataclass(frozen=True, slots=True)
class MarkovChartSpec:
    """Available Markov prefix values and exact visible formatting rules."""

    depths: tuple[int, ...]
    probability_a: tuple[float, ...]
    probability_b: tuple[float, ...]
    entropies: tuple[float, ...]
    probability_hover_a: str
    probability_hover_b: str
    entropy_hover: str
    value_range: tuple[float, float]
    tick_format: str


def entropy_chart_spec(analysis: SequenceAnalysis) -> EntropyChartSpec:
    """Derive ordered chart values from prefix results."""
    return EntropyChartSpec(
        depths=tuple(row.depth for row in analysis.rows),
        entropies=tuple(row.entropy_bits for row in analysis.rows),
        hover_template=(
            "Depth %{x}<br>Predictive entropy %{y:.3f} bits<extra></extra>"
        ),
        entropy_range=(CHART_ENTROPY_MIN, CHART_ENTROPY_MAX),
        transition_duration=MOTION_DURATION,
        tick_format=PLOTLY_NUMBER_FORMAT,
    )


def markov_chart_spec(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> MarkovChartSpec:
    """Collect available Markov predictions without rounding source values."""
    available: list[MarkovPrefixResult] = []
    probabilities: list[tuple[float, ...]] = []
    for row in record.rows:
        if row.predictive is None:
            continue
        available.append(row)
        probabilities.append(float_values(row.predictive))
    return MarkovChartSpec(
        depths=tuple(row.depth for row in available),
        probability_a=tuple(row[0] for row in probabilities),
        probability_b=tuple(row[1] for row in probabilities),
        entropies=tuple(
            row.entropy_bits for row in available if row.entropy_bits is not None
        ),
        probability_hover_a=(
            f"Prefix depth %{{x}}<br>P(next {labels[0]}) %{{y:.3f}}<extra></extra>"
        ),
        probability_hover_b=(
            f"Prefix depth %{{x}}<br>P(next {labels[1]}) %{{y:.3f}}<extra></extra>"
        ),
        entropy_hover=(
            "Prefix depth %{x}<br>Predictive entropy %{y:.3f} bits<extra></extra>"
        ),
        value_range=(CHART_ENTROPY_MIN, CHART_ENTROPY_MAX),
        tick_format=PLOTLY_NUMBER_FORMAT,
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
            "tickformat": spec.tick_format,
        },
    )
    return go.Figure(data=(trace,), layout=layout, frames=())


def markov_probability_figure(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> go.Figure:
    """Build the two-series next-symbol probability plot."""
    spec = markov_chart_spec(record, labels)
    traces = (
        go.Scatter(
            x=spec.depths,
            y=spec.probability_a,
            name=f"P(next {labels[0]})",
            mode="lines+markers",
            hovertemplate=spec.probability_hover_a,
            line={"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
            marker={"color": COLOR_INTERACTIVE, "size": CHART_MARKER_SIZE},
        ),
        go.Scatter(
            x=spec.depths,
            y=spec.probability_b,
            name=f"P(next {labels[1]})",
            mode="lines+markers",
            hovertemplate=spec.probability_hover_b,
            line={
                "color": COLOR_SECONDARY_SERIES,
                "width": CHART_LINE_WIDTH,
                "dash": "dash",
            },
            marker={
                "color": COLOR_SURFACE,
                "size": CHART_MARKER_SIZE,
                "symbol": "diamond",
                "line": {
                    "color": COLOR_SECONDARY_SERIES,
                    "width": CHART_LINE_WIDTH,
                },
            },
        ),
    )
    figure = go.Figure(
        data=traces,
        layout=_markov_layout("Next-symbol probability", spec),
        frames=(),
    )
    _ = figure.update_layout(meta={"sequence_id": record.sequence_id})
    return figure


def markov_entropy_figure(record: MarkovRecordAnalysis) -> go.Figure:
    """Build the fixed-range Markov predictive-entropy plot."""
    spec = markov_chart_spec(record, record.model.observable_labels)
    trace = go.Scatter(
        x=spec.depths,
        y=spec.entropies,
        name="Predictive entropy",
        mode="lines+markers",
        hovertemplate=spec.entropy_hover,
        line={"color": COLOR_INTERACTIVE, "width": CHART_LINE_WIDTH},
        marker={"color": COLOR_INTERACTIVE, "size": CHART_MARKER_SIZE},
        showlegend=False,
    )
    figure = go.Figure(
        data=(trace,),
        layout=_markov_layout("Predictive entropy (bits)", spec),
        frames=(),
    )
    _ = figure.update_layout(meta={"sequence_id": record.sequence_id})
    return figure


def _markov_layout(y_title: str, spec: MarkovChartSpec) -> go.Layout:
    return go.Layout(
        autosize=True,
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"family": FONT_DATA, "color": COLOR_TEXT_SECONDARY},
        hovermode="closest",
        transition={"duration": MOTION_DURATION},
        xaxis={
            "title": {"text": "Prefix depth"},
            "dtick": 1,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": y_title},
            "range": spec.value_range,
            "gridcolor": COLOR_BORDER_SUBTLE,
            "zeroline": False,
            "tickformat": spec.tick_format,
        },
    )
