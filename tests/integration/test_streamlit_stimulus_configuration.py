from bspe import (
    AdditiveSmoothing,
    InclusiveRange,
    PreferenceMetric,
    SoftPreference,
)

from .stimulus_app_support import (
    SearchFixture,
    analyzer_record,
    click_action,
    configure_search,
    search_snapshot,
    select_mode,
    workbench,
)


def test_configuration_when_opened_exposes_every_constraint_and_model_choice() -> None:
    # Given
    app = select_mode(workbench(), "Stimulus Search")

    # When
    for checkbox in app.checkbox:
        if checkbox.label.startswith("Constrain "):
            _ = checkbox.check()
    _ = app.run()

    # Then
    checkbox_labels = {item.label for item in app.checkbox}
    assert {
        "Constrain predicted probability",
        "Constrain predictive entropy",
        "Constrain effective depth",
        "Constrain context support",
        "Constrain A count",
        "Constrain A proportion",
        "Constrain switches",
        "Constrain switch rate",
        "Constrain longest A run",
        "Constrain longest B run",
        "Constrain longest run",
    } <= checkbox_labels
    number_labels = {item.label for item in app.number_input}
    assert {
        "Minimum predicted probability",
        "Maximum predicted probability",
        "Minimum predictive entropy",
        "Maximum predictive entropy",
        "Minimum effective depth",
        "Maximum effective depth",
        "Minimum context support constraint",
        "Maximum context support constraint",
        "Minimum A count",
        "Maximum A count",
        "Minimum A proportion",
        "Maximum A proportion",
        "Minimum switches",
        "Maximum switches",
        "Minimum switch rate",
        "Maximum switch rate",
        "Minimum longest A run",
        "Maximum longest A run",
        "Minimum longest B run",
        "Maximum longest B run",
        "Minimum longest run",
        "Maximum longest run",
    } <= number_labels
    selects = {item.label: item for item in app.selectbox}
    assert selects["Predicted symbol"].options == ["A", "B", "either"]
    assert selects["Predicted symbol"].value == "either"
    assert selects["VMM smoothing"].options == [
        "Krichevsky-Trofimov (alpha = 0.500)",
        "Maximum likelihood (alpha = 0.000)",
        "Custom additive smoothing",
    ]
    preferences = next(
        item for item in app.multiselect if item.label == "Soft preferences"
    )
    assert preferences.options == [metric.value for metric in PreferenceMetric]


def test_constraint_when_switch_rate_is_configured_reaches_stored_search() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    constraint = next(
        item for item in app.checkbox if item.label == "Constrain switch rate"
    )
    _ = constraint.check()
    _ = app.run()
    numbers = {item.label: item for item in app.number_input}
    _ = numbers["Minimum switch rate"].set_value(0.25)
    _ = numbers["Maximum switch rate"].set_value(0.75)
    _ = app.run()

    # When
    app = click_action(app, "Search stimuli")

    # Then
    assert search_snapshot(app.session_state).config.constraints.switch_rate == (
        InclusiveRange(0.25, 0.75)
    )


def test_smoothing_when_custom_alpha_is_selected_reaches_vmm_config() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    smoothing = next(item for item in app.selectbox if item.label == "VMM smoothing")
    _ = smoothing.set_value("Custom additive smoothing")
    _ = app.run()
    alpha = next(
        item for item in app.number_input if item.label == "Custom additive alpha"
    )
    _ = alpha.set_value(0.125)
    _ = app.run()

    # When
    app = click_action(app, "Search stimuli")

    # Then
    configured = search_snapshot(app.session_state).config.vmm_config.smoothing
    assert isinstance(configured, AdditiveSmoothing)
    assert configured.alpha == 0.125


def test_preferences_when_selected_out_of_order_are_stored_in_enum_order() -> None:
    # Given
    app = configure_search(workbench(), SearchFixture())
    preferences = next(
        item for item in app.multiselect if item.label == "Soft preferences"
    )
    _ = preferences.set_value(["switch_rate", "predictive_entropy"])
    _ = app.run()
    numbers = {item.label: item for item in app.number_input}
    _ = numbers["Predictive entropy target"].set_value(0.6)
    _ = numbers["Predictive entropy weight"].set_value(2.0)
    _ = numbers["Switch rate target"].set_value(0.4)
    _ = numbers["Switch rate weight"].set_value(3.0)
    _ = app.run()

    # When
    app = click_action(app, "Search stimuli")

    # Then
    assert search_snapshot(app.session_state).config.preferences == (
        SoftPreference(PreferenceMetric.PREDICTIVE_ENTROPY, 0.6, 2.0),
        SoftPreference(PreferenceMetric.SWITCH_RATE, 0.4, 3.0),
    )


def test_search_input_when_edited_invalidates_stimulus_only() -> None:
    # Given
    app = workbench()
    sequence = next(item for item in app.text_area if item.label == "Observed sequence")
    _ = sequence.set_value("A,A,B,A,A,B,A,A")
    _ = app.run()
    app = click_action(app, "Calculate selected methods")
    analyzer = analyzer_record(app.session_state)
    app = configure_search(app, SearchFixture())
    app = click_action(app, "Search stimuli")
    app = select_mode(app, "Analyzer")
    assert "_stimulus_search_snapshot" in app.session_state.filtered_state
    app = select_mode(app, "Stimulus Search")
    app = click_action(app, "Create complements")
    assert "_stimulus_search_snapshot" in app.session_state.filtered_state
    assert "_stimulus_complements" in app.session_state.filtered_state

    # When
    length = next(item for item in app.number_input if item.label == "Sequence length")
    _ = length.set_value(9)
    _ = app.run()

    # Then
    assert "_stimulus_search_snapshot" not in app.session_state.filtered_state
    assert "_stimulus_complements" not in app.session_state.filtered_state
    app = select_mode(app, "Analyzer")
    assert analyzer_record(app.session_state) is analyzer
