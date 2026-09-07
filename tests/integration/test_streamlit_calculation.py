from pathlib import Path
from typing import Final

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
HMM_METHOD: Final = ["Hidden Markov Model"]
TAB_KEY: Final = "workspace-method-tabs"
SCOPE_KEY: Final = "workspace-sequence-scope"
SUBVIEW_KEY: Final = "workspace-subview"


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    _ = app.multiselect[0].set_value(HMM_METHOD)
    _ = app.run()
    assert not app.exception
    return app


def _select_workspace(app: AppTest, tab: str, subview: str) -> AppTest:
    app.session_state[TAB_KEY] = tab
    app.session_state[SCOPE_KEY] = "One"
    app.session_state[SUBVIEW_KEY] = subview
    _ = app.run()
    assert not app.exception
    return app


def test_app_when_hmm_workspace_opens_has_inputs_but_no_calculated_outputs() -> None:
    # Given / When
    app = _app()

    # Then
    assert app.title[0].value == (
        "Binary Sequence Probability, Prediction & Entropy Workbench"
    )
    assert "Calculate selected methods" in [button.label for button in app.button]
    assert [tab.label for tab in app.tabs] == [
        "Method Comparison",
        "Markov Chain",
        "Hidden Markov Model",
        "Observed Shannon Entropy",
    ]
    assert any("not calculated" in notice.value.lower() for notice in app.info)
    assert not app.metric
    assert not app.dataframe
    assert not app.download_button


def test_app_when_calculate_is_clicked_renders_hand_sequence_outputs() -> None:
    # Given
    app = _app()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    app = _select_workspace(app, "Hidden Markov Model", "Overview")

    # Then
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {
        "Final P(next A)": "0.403",
        "Final P(next B)": "0.597",
        "Prediction": "B",
        "Predictive entropy (bits)": "0.973",
        "Sequence depth": "7",
    }
    assert "Download selected HMM prefix CSV — sequence-001" in [
        item.label for item in app.download_button
    ]
    assert "Download selected HMM candidate-summary CSV — sequence-001" in [
        item.label for item in app.download_button
    ]


def test_app_when_source_probability_changes_updates_read_only_complement() -> None:
    # Given
    app = _app()
    source = next(
        item
        for item in app.number_input
        if item.label == "Initial probability for State 1"
    )
    _ = source.set_value(0.7)
    _ = app.run()

    # Then
    assert not app.exception
    complement = next(
        item
        for item in app.number_input
        if item.label == "Derived probability for State 2 (1 - p)"
    )
    assert complement.disabled
    assert complement.value == 0.30000000000000004


def test_app_when_sequence_is_empty_calculates_depth_zero() -> None:
    # Given
    app = _app()
    _ = app.text_area[0].set_value("")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    app = _select_workspace(app, "Hidden Markov Model", "Evidence")

    # Then
    assert not app.exception
    evidence = app.dataframe[0].value
    assert evidence["Depth"].tolist() == [0]
    assert evidence.loc[0, "Observed context"] == "(empty prefix)"
    assert evidence.loc[0, "P(next A)"] == pytest.approx(0.62)
    assert evidence.loc[0, "P(next B)"] == pytest.approx(0.38)
    assert evidence.loc[0, "Next-hidden State 1"] == pytest.approx(0.6)
    assert evidence.loc[0, "Next-hidden State 2"] == pytest.approx(0.4)
    assert evidence.loc[0, "Posterior status"] == "Unavailable before observation"


def test_app_when_sequence_is_impossible_reports_zero_likelihood() -> None:
    # Given
    app = _app()
    values = {item.label: item for item in app.number_input}
    _ = values["Emission probability for A given State 1"].set_value(1.0)
    _ = values["Emission probability for A given State 2"].set_value(1.0)
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("B")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    app = _select_workspace(app, "Hidden Markov Model", "Overview")

    # Then
    assert not app.exception
    assert any("zero likelihood" in notice.value for notice in app.error)


def test_app_when_reset_is_clicked_restores_demo_model_without_calculating() -> None:
    # Given
    app = _app()
    state = next(
        item for item in app.text_input if item.label == "Hidden state 1 label"
    )
    _ = state.set_value("Changed state")
    _ = app.run()

    # When
    _ = next(
        button for button in app.button if button.label == "Reset HMM example model"
    ).click()
    _ = app.run()

    # Then
    assert not app.exception
    state = next(
        item for item in app.text_input if item.label == "Hidden state 1 label"
    )
    initial = next(
        item
        for item in app.number_input
        if item.label == "Initial probability for State 1"
    )
    assert state.value == "State 1"
    assert initial.value == 0.6


def test_app_when_calculated_input_changes_replaces_outputs_with_stale_notice() -> None:
    # Given
    app = _app()
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()

    # When
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("A, B")
    _ = app.run()

    # Then
    assert not app.exception
    assert any("recalculation required" in item.value.lower() for item in app.warning)
    labels = [item.label for item in app.download_button]
    assert "Download HMM prefix CSV" not in labels
    assert "Download HMM candidate-summary CSV" not in labels


def test_app_when_hmm_evidence_opens_encodes_filtering_conventions() -> None:
    # Given
    app = _app()

    # When
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    app = _select_workspace(app, "Hidden Markov Model", "Evidence")

    # Then
    evidence = app.dataframe[0].value
    assert evidence["Depth"].tolist() == list(range(8))
    assert evidence.loc[0, "Observed context"] == "(empty prefix)"
    assert evidence.loc[0, "Next-hidden State 1"] == pytest.approx(0.6)
    assert evidence.loc[0, "P(next A)"] == pytest.approx(0.62)
    assert evidence.loc[1, "Posterior State 1"] == pytest.approx(27 / 31)
    assert evidence.loc[1, "Next-hidden State 1"] == pytest.approx(
        0.635483870967742
    )
    assert evidence.loc[1, "P(next A)"] == pytest.approx(0.644838709677419)
