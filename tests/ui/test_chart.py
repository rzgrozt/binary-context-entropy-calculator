import plotly.graph_objects as go

from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.ui.chart import (
    entropy_chart_spec,
    entropy_figure,
    markov_chart_spec,
    markov_entropy_figure,
    markov_probability_figure,
)
from binary_entropy.ui.state import (
    CalculationFailure,
    CalculationSuccess,
    calculate_form,
    default_form,
)
from binary_entropy.ui.workbench_state import (
    WorkbenchCalculationSuccess,
    calculate_workbench,
    default_workbench_form,
)


def test_entropy_figure_when_using_demo_is_sorted_exact_and_static() -> None:
    # Given
    outcome = calculate_form(default_form())
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationSuccess(analysis=analysis):
            pass
        case CalculationFailure(message=message):
            raise AssertionError(message)

    # When
    spec = entropy_chart_spec(analysis)
    figure = entropy_figure(analysis)

    # Then
    assert spec.depths == tuple(range(8))
    assert spec.hover_template == (
        "Depth %{x}<br>Predictive entropy %{y:.3f} bits<extra></extra>"
    )
    assert spec.entropy_range == (0, 1)
    assert spec.transition_duration == 0
    assert isinstance(figure, go.Figure)
    assert figure.layout.autosize is True
    assert figure.layout.height is None
    assert len(figure.frames) == 0
    assert spec.tick_format == ".3f"


def test_markov_figures_when_using_demo_keep_raw_values_and_three_decimal_views() -> (
    None
):
    # Given
    outcome = calculate_workbench(default_workbench_form())
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    analysis = outcome.results[0]
    assert isinstance(analysis, MarkovBatchAnalysis)
    record = analysis.records[0]
    spec = markov_chart_spec(record, analysis.model.observable_labels)

    # When
    probability = markov_probability_figure(record, analysis.model.observable_labels)
    entropy = markov_entropy_figure(record)

    # Then
    assert len(probability.data) == 2
    assert spec.probability_hover_a == (
        "Prefix depth %{x}<br>P(next A) %{y:.3f}<extra></extra>"
    )
    assert spec.value_range == (0, 1)
    assert spec.tick_format == ".3f"
    assert len(probability.frames) == len(entropy.frames) == 0
