from typing import ClassVar

import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from binary_entropy.domain import BinaryLabels
from binary_entropy.methods.vmm import analyze_vmm_per_sequence
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.vmm_chart import (
    vmm_entropy_chart_spec,
    vmm_entropy_figure,
)
from binary_entropy.vmm_types import MLESmoothing, VMMConfig


class _TitleView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    text: str


class _AxisView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    title: _TitleView
    range: tuple[float, float] | None = None
    tickformat: str | None = None


class _TransitionView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    duration: int


class _LayoutView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    xaxis: _AxisView
    yaxis: _AxisView
    transition: _TransitionView


class _FigureView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    layout: _LayoutView


def _mle_dataset() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (SequenceRecord("mle-unseen", (0, 1, 1)),),
    )


def test_vmm_entropy_spec_when_mle_depths_are_unavailable_omits_them() -> None:
    # Given
    analysis = analyze_vmm_per_sequence(
        _mle_dataset(),
        VMMConfig(smoothing=MLESmoothing(), minimum_support=1),
    )

    # When
    spec = vmm_entropy_chart_spec(analysis.records[0])

    # Then
    assert spec.requested_depths == (0, 1)
    assert spec.entropies == tuple(
        row.predictive_entropy_bits for row in analysis.records[0].depth_rows[:2]
    )
    assert spec.trace_name == "Predictive entropy"
    assert spec.hover_template == (
        "Requested depth %{x}<br>Predictive entropy %{y:.3f} bits<extra></extra>"
    )
    assert spec.entropy_range == (0.0, 1.0)
    assert spec.tick_format == ".3f"
    assert spec.transition_duration == 0
    assert spec.static_plot is True
    assert spec.display_modebar is False
    assert spec.responsive is True


def test_vmm_entropy_figure_when_values_are_available_is_named_exact_and_static() -> (
    None
):
    # Given
    analysis = analyze_vmm_per_sequence(
        _mle_dataset(),
        VMMConfig(smoothing=MLESmoothing(), minimum_support=1),
    )
    record = analysis.records[0]
    spec = vmm_entropy_chart_spec(record)

    # When
    figure = vmm_entropy_figure(record)
    view = _FigureView.model_validate_json(figure.to_json())

    # Then
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    trace = figure.data[0]
    assert isinstance(trace, go.Scatter)
    assert trace.name == spec.trace_name
    assert trace.mode == "lines+markers"
    assert tuple(trace.x) == spec.requested_depths
    assert tuple(trace.y) == spec.entropies
    assert trace.hovertemplate == spec.hover_template
    assert trace.showlegend is True
    assert view.layout.yaxis.range == (0.0, 1.0)
    assert view.layout.yaxis.tickformat == ".3f"
    assert view.layout.xaxis.title.text == "Requested context depth"
    assert view.layout.yaxis.title.text == "Predictive entropy (bits)"
    assert view.layout.transition.duration == 0
    assert len(figure.frames) == 0
