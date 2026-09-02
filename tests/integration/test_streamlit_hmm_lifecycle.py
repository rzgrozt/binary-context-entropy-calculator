from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
ALL_METHODS: Final = [
    "Markov Chain",
    "Hidden Markov Model",
    "Observed Shannon Entropy",
]
METHODS_WITHOUT_HMM: Final = ["Markov Chain", "Observed Shannon Entropy"]


def _markov_workspace() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    assert app.multiselect[0].value == ["Markov Chain"]
    _ = next(button for button in app.button if button.label == "Continue").click()
    _ = app.run()
    assert not app.exception
    workflow = next(
        item for item in app.selectbox if item.label == "Markov workflow"
    )
    _ = workflow.set_value("First-order Markov")
    _ = app.run()
    assert not app.exception
    return app


def _calculate(app: AppTest) -> AppTest:
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    assert not app.exception
    return app


def test_hmm_when_added_inside_workspace_rehydrates_example_and_calculates() -> None:
    # Given
    app = _markov_workspace()

    # When
    _ = app.multiselect[0].set_value(ALL_METHODS)
    _ = app.run()
    app = _calculate(app)

    # Then
    labels = {
        item.label: item.value
        for item in app.text_input
        if item.label.startswith("Hidden state")
    }
    numbers = {item.label: item for item in app.number_input}
    assert labels == {
        "Hidden state 1 label": "State 1",
        "Hidden state 2 label": "State 2",
    }
    assert numbers["Initial probability for State 1"].value == 0.6
    assert numbers["Derived probability for State 2 (1 - p)"].value == 0.4
    assert numbers["Derived probability for State 2 (1 - p)"].disabled
    assert numbers["Transition probability to State 1 from State 1"].value == 0.7
    assert numbers["Derived probability to State 2 from State 1 (1 - p)"].value == (
        1.0 - 0.7
    )
    assert numbers["Transition probability to State 1 from State 2"].value == 0.2
    assert numbers["Derived probability to State 2 from State 2 (1 - p)"].value == 0.8
    assert numbers["Emission probability for A given State 1"].value == 0.9
    assert numbers["Derived probability for B given State 1 (1 - p)"].value == (
        1.0 - 0.9
    )
    assert numbers["Emission probability for A given State 2"].value == 0.2
    assert numbers["Derived probability for B given State 2 (1 - p)"].value == 0.8
    comparison = next(
        item.value for item in app.dataframe if "Method" in item.value.columns
    )
    assert comparison["Method"].tolist() == ALL_METHODS
    assert "Hidden Markov Model" in [item.value for item in app.subheader]


def test_hmm_when_deselected_and_reselected_preserves_edited_source_state() -> None:
    # Given
    app = _markov_workspace()
    _ = app.multiselect[0].set_value(ALL_METHODS)
    _ = app.run()
    _ = next(
        item for item in app.text_input if item.label == "Hidden state 1 label"
    ).set_value("Calm")
    _ = next(
        item
        for item in app.number_input
        if item.label == "Initial probability for State 1"
    ).set_value(0.37)
    _ = app.run()

    # When
    _ = app.multiselect[0].set_value(METHODS_WITHOUT_HMM)
    _ = app.run()
    _ = app.multiselect[0].set_value(ALL_METHODS)
    _ = app.run()

    # Then
    state = next(
        item for item in app.text_input if item.label == "Hidden state 1 label"
    )
    source = next(
        item
        for item in app.number_input
        if item.label == "Initial probability for Calm"
    )
    complement = next(
        item
        for item in app.number_input
        if item.label == "Derived probability for State 2 (1 - p)"
    )
    assert state.value == "Calm"
    assert source.value == 0.37
    assert complement.value == 1.0 - 0.37
    assert complement.disabled


def test_hmm_when_followup_method_event_submits_empty_fields_restores_shadow() -> None:
    # Given
    app = _markov_workspace()
    _ = app.multiselect[0].set_value(ALL_METHODS[:2])
    _ = app.run()
    state_inputs = {
        item.label: item
        for item in app.text_input
        if item.label.startswith("Hidden state")
    }
    number_inputs = {item.label: item for item in app.number_input}
    _ = state_inputs["Hidden state 1 label"].set_value("")
    _ = state_inputs["Hidden state 2 label"].set_value("")
    _ = number_inputs["Initial probability for State 1"].set_value(0.0)

    # When
    _ = app.multiselect[0].set_value(ALL_METHODS)
    _ = app.run()
    app = _calculate(app)

    # Then
    labels = {
        item.label: item.value
        for item in app.text_input
        if item.label.startswith("Hidden state")
    }
    numbers = {item.label: item for item in app.number_input}
    comparison = next(
        item.value for item in app.dataframe if "Method" in item.value.columns
    )
    assert labels == {
        "Hidden state 1 label": "State 1",
        "Hidden state 2 label": "State 2",
    }
    assert numbers["Initial probability for State 1"].value == 0.6
    assert numbers["Derived probability for State 2 (1 - p)"].value == 0.4
    assert comparison["Method"].tolist() == ALL_METHODS


def test_preset_name_when_changed_after_all_methods_calculate_stales_only_hmm() -> None:
    # Given
    app = _markov_workspace()
    _ = app.multiselect[0].set_value(ALL_METHODS)
    _ = app.run()
    app = _calculate(app)

    # When
    _ = next(item for item in app.text_input if item.label == "Preset name").set_value(
        "renamed model"
    )
    _ = app.run()

    # Then
    assert not app.exception
    assert any("Hidden Markov Model" in item.value for item in app.warning)
    assert "Markov Chain" in [item.value for item in app.subheader]
    assert "Observed-symbol Shannon entropy" in [item.value for item in app.subheader]
    assert "Hidden Markov Model" not in [item.value for item in app.subheader]
    comparison = next(
        item.value for item in app.dataframe if "Method" in item.value.columns
    )
    assert comparison["Method"].tolist() == [ALL_METHODS[0], ALL_METHODS[2]]
    downloads = [item.label for item in app.download_button]
    assert "Download Markov model JSON" in downloads
    assert "Download Markov prefix CSV" in downloads
    assert "Download Markov batch-summary CSV" in downloads
    assert "Download HMM preset JSON" not in downloads
    assert "Download HMM prefix CSV" not in downloads
    assert "Download HMM candidate-summary CSV" not in downloads
