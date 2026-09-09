from .stimulus_app_support import (
    SearchFixture,
    click_action,
    configure_search,
    matching_result,
    search_snapshot,
    workbench,
)


def test_matching_when_complements_are_created_is_explicit_and_snapshot_safe() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    app = click_action(app, "Search stimuli")
    snapshot = search_snapshot(app.session_state)

    # When
    app = click_action(app, "Create complements")
    app = click_action(app, "Match stimuli")

    # Then
    matching = matching_result(app.session_state)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Complement candidates"] == "8"
    assert metrics["Matched pairs"] == str(len(matching.pairs))
    assert metrics["Unmatched candidates"] == str(len(matching.unmatched))
    assert len(matching.pairs) > 0
    assert len(matching.pairs) * 2 + len(matching.unmatched) == 16
    pairs = next(
        item.value
        for item in app.dataframe
        if {
            "Pair ID",
            "First stimulus ID",
            "Second stimulus ID",
            "Total distance",
        }
        <= set(item.value.columns)
    )
    assert len(pairs) == len(matching.pairs)
    assert search_snapshot(app.session_state) is snapshot


def test_assignment_when_seeded_runs_separately_before_descriptive_qc() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    app = click_action(app, "Search stimuli")
    snapshot = search_snapshot(app.session_state)
    seed = next(item for item in app.number_input if item.label == "Assignment seed")
    _ = seed.set_value(5)
    _ = app.run()

    # When
    app = click_action(app, "Assign targets")

    # Then
    assert app.session_state["_stimulus_assignment_seed"] == 5
    assignments = next(
        item.value
        for item in app.dataframe
        if {"Stimulus ID", "Target", "Congruency"} <= set(item.value.columns)
    )
    assert len(assignments) == 8
    assert set(assignments["Target"]) <= {"A", "B"}
    assert "Run descriptive QC" in [item.label for item in app.button]
    assert "Descriptive quality control" not in [item.value for item in app.subheader]
    assert search_snapshot(app.session_state) is snapshot


def test_quality_control_when_run_reports_descriptive_assignment_counts() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    app = click_action(app, "Search stimuli")
    seed = next(item for item in app.number_input if item.label == "Assignment seed")
    _ = seed.set_value(5)
    _ = app.run()
    app = click_action(app, "Assign targets")

    # When
    app = click_action(app, "Run descriptive QC")

    # Then
    assert "Descriptive quality control" in [item.value for item in app.subheader]
    metrics = {item.label: item.value for item in app.metric}
    prediction_counts = sum(
        int(metrics[label])
        for label in ("Predicted A", "Predicted B", "Ties", "Unavailable")
    )
    assignment_counts = sum(
        int(metrics[label])
        for label in (
            "Expected targets",
            "Unexpected targets",
            "Tie targets",
            "Unassigned targets",
        )
    )
    assert metrics["Stimuli"] == "8"
    assert prediction_counts == 8
    assert assignment_counts == 8
    assert "Imbalance flags" in metrics


def test_quality_control_is_available_before_target_assignment() -> None:
    app = click_action(configure_search(workbench(), SearchFixture()), "Search stimuli")
    app = click_action(app, "Run descriptive QC")
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Unassigned targets"] == "8"


def test_selected_stimulus_can_use_existing_analyzer_diagnostics() -> None:
    app = click_action(configure_search(workbench(), SearchFixture()), "Search stimuli")
    control = next(
        item
        for item in app.checkbox
        if item.label == "Show analyzer diagnostics, plot and sequence exports"
    )
    _ = control.check()
    _ = app.run()
    assert not app.exception
    assert "Variable-order Markov" in [item.value for item in app.subheader]
    assert "Download Context model export (JSON)" in [
        item.label for item in app.download_button
    ]
