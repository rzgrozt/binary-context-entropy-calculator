from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
SCIENTIFIC_WARNING: Final = (
    "The entered sequence alone does not uniquely determine a next-target "
    "probability distribution. HMM predictive results depend on the selected "
    "HMM parameters."
)


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    return app


def test_app_when_initial_has_inputs_but_no_calculated_outputs() -> None:
    # Given / When
    app = _app()

    # Then
    assert app.title[0].value == "Binary Sequence Predictive Entropy Calculator"
    assert "Calculate entropy" in [button.label for button in app.button]
    assert len(app.get("html")) == 1
    assert any("not calculated" in notice.value.lower() for notice in app.info)
    assert "Download prefix CSV" not in [item.label for item in app.download_button]


def test_app_when_calculate_is_clicked_renders_hand_sequence_outputs() -> None:
    # Given
    app = _app()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert any(notice.value == "Calculation complete." for notice in app.success)
    assert len(app.get("html")) == 2
    assert any(
        metric.label == "Context depth" and metric.value == "7" for metric in app.metric
    )
    assert "Download prefix CSV" in [item.label for item in app.download_button]
    assert "Download candidate-summary CSV" in [
        item.label for item in app.download_button
    ]


def test_app_when_probability_sum_is_invalid_shows_specific_error() -> None:
    # Given
    app = _app()
    _ = next(
        item
        for item in app.number_input
        if item.label == "Initial probability for State 1"
    ).set_value(0.7)
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert any("initial must sum to 1" in notice.value for notice in app.error)
    assert len(app.get("html")) == 1


def test_app_when_sequence_is_empty_calculates_depth_zero() -> None:
    # Given
    app = _app()
    _ = app.text_area[0].set_value("")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert len(app.get("html")) == 2
    assert any(
        metric.label == "Context depth" and metric.value == "0" for metric in app.metric
    )
    assert any(
        "unavailable for an empty sequence" in item.value.lower() for item in app.info
    )


def test_app_when_sequence_is_impossible_reports_zero_likelihood() -> None:
    # Given
    app = _app()
    values = {item.label: item for item in app.number_input}
    _ = values["Emission P(A | State 1)"].set_value(1.0)
    _ = values["Emission P(B | State 1)"].set_value(0.0)
    _ = values["Emission P(A | State 2)"].set_value(1.0)
    _ = values["Emission P(B | State 2)"].set_value(0.0)
    _ = app.text_area[0].set_value("B")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert any("zero likelihood" in notice.value for notice in app.error)
    assert len(app.get("html")) == 1


def test_app_when_reset_is_clicked_restores_demo_model_without_calculating() -> None:
    # Given
    app = _app()
    _ = app.text_input[0].set_value("Changed state")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Reset example model"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert app.text_input[0].value == "State 1"
    assert app.number_input[0].value == 0.6
    assert len(app.get("html")) == 1


def test_app_when_calculated_input_changes_replaces_outputs_with_stale_notice() -> None:
    # Given
    app = _app()
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()
    assert len(app.get("html")) == 2

    # When
    _ = app.text_area[0].set_value("A, B")
    _ = app.run()

    # Then
    assert not app.exception
    assert len(app.get("html")) == 1
    assert any(
        "recalculation is required" in item.value.lower() for item in app.warning
    )
    labels = [item.label for item in app.download_button]
    assert "Download prefix CSV" not in labels
    assert "Download candidate-summary CSV" not in labels


def test_app_always_states_scientific_warning_and_pi_convention() -> None:
    # Given / When
    app = _app()

    # Then
    assert any(item.value == SCIENTIFIC_WARNING for item in app.warning)
    all_markdown = "\n".join(item.value for item in app.markdown)
    assert "depth 0 uses q1=pi E without transition" in all_markdown
