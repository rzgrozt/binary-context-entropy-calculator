from typing import Final

from .stimulus_app_support import (
    ANALYZER_TABS,
    MODE_KEY,
    analyzer_record,
    choice_by_key,
    click_action,
    select_mode,
    workbench,
)

METHOD_OPTIONS: Final = [
    "Markov Chain",
    "Hidden Markov Model",
    "Observed Shannon Entropy",
]


def test_workspace_when_launched_exposes_exact_modes_and_unchanged_analyzer() -> None:
    # Given / When
    app = workbench()

    # Then
    mode = choice_by_key(app, MODE_KEY)
    assert mode.label == "Workspace mode"
    assert mode.options == ["Analyzer", "Stimulus Search"]
    assert mode.value == "Analyzer"
    methods = next(item for item in app.multiselect if item.label == "Analysis methods")
    assert methods.options == METHOD_OPTIONS
    assert methods.value == ["Markov Chain"]
    workflow = next(item for item in app.selectbox if item.label == "Markov workflow")
    assert workflow.value == "Variable-order Markov"
    assert [tab.label for tab in app.tabs] == ANALYZER_TABS
    assert "Calculate selected methods" in [item.label for item in app.button]


def test_stimulus_search_when_selected_replaces_analyzer_presentation_only() -> None:
    # Given
    app = workbench()

    # When
    app = select_mode(app, "Stimulus Search")

    # Then
    assert choice_by_key(app, MODE_KEY).value == "Stimulus Search"
    number_labels = {item.label for item in app.number_input}
    assert {
        "Sequence length",
        "Desired count",
        "Candidate limit",
        "Minimum context support",
    } <= number_labels
    search_seed = next(item for item in app.text_input if item.label == "Search seed")
    assert search_seed.key == "stimulus-search-seed"
    assert "Search stimuli" in [item.label for item in app.button]
    assert "Calculate selected methods" not in [item.label for item in app.button]
    assert "Markov workflow" not in [item.label for item in app.selectbox]
    assert not app.tabs


def test_analyzer_snapshot_when_modes_round_trip_preserves_identity_and_state() -> None:
    # Given
    app = workbench()
    sequence = next(item for item in app.text_area if item.label == "Observed sequence")
    _ = sequence.set_value("A,A,B,A,A,B,A,A")
    _ = app.run()
    app = click_action(app, "Calculate selected methods")
    original = analyzer_record(app.session_state)

    # When
    app = select_mode(app, "Stimulus Search")
    app = select_mode(app, "Analyzer")

    # Then
    assert analyzer_record(app.session_state) is original
    restored_sequence = next(
        item for item in app.text_area if item.label == "Observed sequence"
    )
    assert restored_sequence.value == "A,A,B,A,A,B,A,A"
    assert [tab.label for tab in app.tabs] == ANALYZER_TABS
    assert "Calculate selected methods" in [item.label for item in app.button]
