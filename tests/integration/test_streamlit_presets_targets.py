from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

from binary_entropy.ui.state import ActualTargetChoice

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    return app


def test_app_when_valid_preset_is_loaded_updates_model_without_calculating() -> None:
    # Given
    app = _app()
    payload = b"""{
      "schema_version": 1,
      "preset_name": "Imported model",
      "state_labels": ["Quiet", "Active"],
      "observable_labels": ["Left", "Right"],
      "initial": [0.25, 0.75],
      "transition": [[0.8, 0.2], [0.1, 0.9]],
      "emission": [[0.6, 0.4], [0.3, 0.7]]
    }"""
    _ = app.file_uploader[0].set_value(("imported.json", payload, "application/json"))
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Load preset JSON"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert app.text_input[0].value == "Quiet"
    assert app.text_input[2].value == "Left"
    assert app.number_input[0].value == 0.25
    assert any("no calculation was run" in item.value for item in app.success)
    assert len(app.get("html")) == 1


def test_app_when_invalid_preset_is_loaded_preserves_current_model() -> None:
    # Given
    app = _app()
    _ = app.text_input[0].set_value("Current state")
    _ = app.run()
    _ = app.file_uploader[0].set_value(
        ("invalid.json", b"{not-json}", "application/json")
    )
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Load preset JSON"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    assert app.text_input[0].value == "Current state"
    assert any("could not be decoded" in item.value for item in app.error)
    assert len(app.get("html")) == 1


def test_app_when_lower_probability_target_is_selected_assesses_final_prediction() -> (
    None
):
    # Given
    app = _app()
    _ = app.radio[0].set_value(ActualTargetChoice.FIRST)
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Actual next target"] == "A"
    assert metrics["Actual-target probability"] == "0.402822877316"
    assert any("lower probability" in item.value for item in app.markdown)


def test_app_when_modal_target_is_selected_uses_modal_wording() -> None:
    # Given
    app = _app()
    _ = app.radio[0].set_value(ActualTargetChoice.SECOND)
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert any("is modal" in item.value for item in app.markdown)


def test_app_when_predictive_targets_are_tied_uses_tied_wording() -> None:
    # Given
    app = _app()
    emissions = {
        item.label: item
        for item in app.number_input
        if item.label.startswith("Emission")
    }
    for input_widget in emissions.values():
        _ = input_widget.set_value(0.5)
    _ = app.radio[0].set_value(ActualTargetChoice.FIRST)
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    assert any("probabilities are tied" in item.value for item in app.markdown)


def test_app_when_actual_target_has_zero_probability_labels_infinite_surprisal() -> (
    None
):
    # Given
    app = _app()
    values = {item.label: item for item in app.number_input}
    _ = values["Emission P(A | State 1)"].set_value(1.0)
    _ = values["Emission P(B | State 1)"].set_value(0.0)
    _ = values["Emission P(A | State 2)"].set_value(1.0)
    _ = values["Emission P(B | State 2)"].set_value(0.0)
    _ = app.text_area[0].set_value("")
    _ = app.radio[0].set_value(ActualTargetChoice.SECOND)
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate entropy"
    ).click()
    _ = app.run()

    # Then
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Realized surprisal (bits)"] == "Infinite (zero probability)"
