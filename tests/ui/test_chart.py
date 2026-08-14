import plotly.graph_objects as go

from binary_entropy.ui.chart import entropy_chart_spec, entropy_figure
from binary_entropy.ui.state import (
    CalculationFailure,
    CalculationSuccess,
    calculate_form,
    default_form,
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
        "Depth %{x}<br>Predictive entropy %{y:.12f} bits<extra></extra>"
    )
    assert spec.entropy_range == (0, 1)
    assert spec.transition_duration == 0
    assert isinstance(figure, go.Figure)
    assert figure.layout.autosize is True
    assert figure.layout.height is None
    assert len(figure.frames) == 0
