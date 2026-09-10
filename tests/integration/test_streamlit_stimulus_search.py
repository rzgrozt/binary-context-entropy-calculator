from typing import Final

from bspe import InclusiveRange, SearchStatus

from .stimulus_app_support import (
    CANDIDATE_KEY,
    SearchFixture,
    choice_by_key,
    click_action,
    configure_search,
    search_failure,
    search_snapshot,
    workbench,
)

EXPORT_LABELS: Final = {
    "Candidate CSV",
    "Scientific CSV",
    "Reproducibility configuration JSON",
    "Experiment-ready CSV",
}
SEARCH_SNAPSHOT_KEY: Final = "_stimulus_search_snapshot"
SEARCH_FAILURE_KEY: Final = "_stimulus_search_failure"


def test_search_when_seed_is_int64_max_preserves_exact_configuration() -> None:
    # Given
    maximum_seed = 2**63 - 1
    app = configure_search(workbench(), SearchFixture(seed=maximum_seed))

    # When
    app = click_action(app, "Search stimuli")

    # Then
    assert search_snapshot(app.session_state).config.seed == maximum_seed
    search_seed = next(item for item in app.text_input if item.label == "Search seed")
    assert search_seed.key == "stimulus-search-seed"
    assert search_seed.value == str(maximum_seed)
    assert not app.exception


def test_search_when_seed_is_malformed_retains_raw_value_and_stores_failure() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    app = click_action(app, "Search stimuli")
    assert SEARCH_SNAPSHOT_KEY in app.session_state.filtered_state
    raw_seed = "not-a-seed"
    search_seed = next(item for item in app.text_input if item.label == "Search seed")
    _ = search_seed.set_value(raw_seed)
    _ = app.run()
    assert SEARCH_SNAPSHOT_KEY not in app.session_state.filtered_state
    assert SEARCH_FAILURE_KEY not in app.session_state.filtered_state

    # When
    app = click_action(app, "Search stimuli")

    # Then
    assert SEARCH_FAILURE_KEY in app.session_state.filtered_state
    assert SEARCH_SNAPSHOT_KEY not in app.session_state.filtered_state
    assert search_failure(app.session_state).message == (
        f"invalid stimulus search seed: {raw_seed}"
    )
    retained_seed = next(item for item in app.text_input if item.label == "Search seed")
    assert retained_seed.value == raw_seed
    assert not app.exception


def test_search_when_seed_overflows_int64_retains_raw_value_and_stores_failure() -> (
    None
):
    # Given
    app = configure_search(workbench(), SearchFixture())
    app = click_action(app, "Search stimuli")
    assert SEARCH_SNAPSHOT_KEY in app.session_state.filtered_state
    raw_seed = str(2**63)
    search_seed = next(item for item in app.text_input if item.label == "Search seed")
    _ = search_seed.set_value(raw_seed)
    _ = app.run()
    assert SEARCH_SNAPSHOT_KEY not in app.session_state.filtered_state
    assert SEARCH_FAILURE_KEY not in app.session_state.filtered_state

    # When
    app = click_action(app, "Search stimuli")

    # Then
    assert SEARCH_FAILURE_KEY in app.session_state.filtered_state
    assert SEARCH_SNAPSHOT_KEY not in app.session_state.filtered_state
    assert search_failure(app.session_state).message == (
        f"invalid stimulus search seed: {raw_seed}"
    )
    retained_seed = next(item for item in app.text_input if item.label == "Search seed")
    assert retained_seed.value == raw_seed
    assert not app.exception


def test_search_when_small_seeded_fixture_runs_shows_bounded_results_and_exports() -> (
    None
):
    # Given
    app = configure_search(
        workbench(),
        SearchFixture(desired_count=4, candidate_limit=32),
    )

    # When
    app = click_action(app, "Search stimuli")

    # Then
    snapshot = search_snapshot(app.session_state)
    assert snapshot.status is SearchStatus.COMPLETE
    assert snapshot.config.seed == 17
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Status"] == "Complete"
    assert metrics["Requested"] == "4"
    assert metrics["Evaluated"] == "32"
    assert metrics["Accepted"] == "32"
    assert metrics["Selected"] == "4"
    candidates = next(
        item.value
        for item in app.dataframe
        if {"Stimulus ID", "Sequence", "Rank score"} <= set(item.value.columns)
    )
    assert len(candidates) == 25
    assert candidates["Sequence"].str.fullmatch("[AB]{8}").all()
    captions = "\n".join(item.value for item in app.caption)
    assert "Showing 25 of 32 rows" in captions

    candidate = choice_by_key(app, CANDIDATE_KEY)
    selected_id = candidate.options[1]
    _ = candidate.set_value(selected_id)
    _ = app.run()
    assert not app.exception
    detail_metrics = {item.label: item.value for item in app.metric}
    assert detail_metrics["Stimulus ID"] == selected_id
    assert search_snapshot(app.session_state) is snapshot

    downloads = {item.label: item for item in app.download_button}
    assert set(downloads) == EXPORT_LABELS
    assert all(not downloads[label].disabled for label in EXPORT_LABELS)


def test_search_when_hard_constraint_is_impossible_reports_unrelaxed_partial() -> None:
    # Given
    app = configure_search(
        workbench(),
        SearchFixture(desired_count=4, candidate_limit=16),
    )
    constraint = next(
        item for item in app.checkbox if item.label == "Constrain A count"
    )
    _ = constraint.check()
    _ = app.run()
    numbers = {item.label: item for item in app.number_input}
    _ = numbers["Maximum A count"].set_value(9)
    _ = numbers["Minimum A count"].set_value(9)
    _ = app.run()

    # When
    app = click_action(app, "Search stimuli")

    # Then
    snapshot = search_snapshot(app.session_state)
    assert snapshot.status is SearchStatus.PARTIAL
    assert snapshot.config.constraints.a_count == InclusiveRange(9, 9)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Status"] == "Partial"
    assert metrics["Requested"] == "4"
    assert metrics["Evaluated"] == "16"
    assert metrics["Accepted"] == "0"
    assert metrics["Selected"] == "0"
    violations = next(
        item.value
        for item in app.dataframe
        if {"Constraint", "Failure count", "Frequency"} <= set(item.value.columns)
    )
    assert violations["Constraint"].tolist() == ["a_count"]
    assert violations["Failure count"].tolist() == [16]
    assert violations["Frequency"].tolist() == [1.0]
    retained = {item.label: item.value for item in app.number_input}
    assert retained["Minimum A count"] == 9
    assert retained["Maximum A count"] == 9
