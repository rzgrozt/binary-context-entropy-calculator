from pathlib import Path
from typing import Final, Protocol

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup, Markdown, Radio, Selectbox

from bspe.ui.session import WorkbenchCalculationRecord
from bspe.ui.text import joined_text

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
ALL_METHODS: Final = [
    "Markov Chain",
    "Hidden Markov Model",
    "Observed Shannon Entropy",
]
TAB_KEY: Final = "workspace-method-tabs"


class _WorkbenchSession(Protocol):
    def __getitem__(self, key: str) -> WorkbenchCalculationRecord: ...


def _workbench_record(state: _WorkbenchSession) -> WorkbenchCalculationRecord:
    return state["_calculation_record"]


def _app(methods: list[str] | None = None) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    if methods is not None:
        _ = app.multiselect[0].set_value(methods)
        _ = app.run()
    continue_buttons = [item for item in app.button if item.label == "Continue"]
    if continue_buttons:
        _ = continue_buttons[0].click()
        _ = app.run()
    assert not app.exception
    return app


def _batch(app: AppTest, size: int) -> AppTest:
    mode = next(item for item in app.selectbox if item.label == "Input mode")
    _ = mode.set_value("Batch paste")
    _ = app.run()
    payload = "\n".join("A,A,B,A" if index % 2 else "B,A,B,B" for index in range(size))
    area = next(item for item in app.text_area if item.label == "Batch sequences")
    _ = area.set_value(payload)
    _ = app.run()
    return app


def _calculate(app: AppTest) -> AppTest:
    button = next(
        item for item in app.button if item.label == "Calculate selected methods"
    )
    _ = button.click()
    _ = app.run()
    assert not app.exception
    return app


def _first_order_per_sequence(app: AppTest) -> AppTest:
    workflow = next(item for item in app.selectbox if item.label == "Markov workflow")
    _ = workflow.set_value("First-order Markov")
    _ = app.run()
    result_scope = next(
        item for item in app.selectbox if item.label == "Markov result scope"
    )
    _ = result_scope.set_value("Per-sequence analysis")
    _ = app.run()
    assert not app.exception
    return app


def _set_key(app: AppTest, key: str, value: str) -> AppTest:
    app.session_state[key] = value
    _ = app.run()
    assert not app.exception
    return app


def _choice(app: AppTest, key: str) -> Selectbox[str] | Radio[str] | ButtonGroup[str]:
    for item in app.selectbox:
        if item.key == key:
            return item
    for item in app.radio:
        if item.key == key:
            return item
    for item in app.button_group:
        if item.key == key:
            return item
    message = f"missing workspace choice: {key}"
    raise AssertionError(message)


@pytest.mark.parametrize("subview", ["Overview", "Compare", "Evidence"])
def test_workspace_selectors_when_tabs_change_persist_without_recalculation(
    subview: str,
) -> None:
    # Given
    app = _calculate(_app(ALL_METHODS))
    record = _workbench_record(app.session_state)
    assert isinstance(record, WorkbenchCalculationRecord)
    scope = _choice(app, "workspace-sequence-scope")
    _ = scope.set_value("One")
    _ = app.run()
    sequence = next(
        item for item in app.selectbox if item.key == "workspace-sequence-id"
    )
    _ = sequence.set_value("sequence-001")
    _ = app.run()
    view = _choice(app, "workspace-subview")
    _ = view.set_value(subview)
    _ = app.run()

    # When
    for tab in (
        "Method Comparison",
        "Markov Chain",
        "Hidden Markov Model",
        "Observed Shannon Entropy",
    ):
        app = _set_key(app, TAB_KEY, tab)

    # Then
    assert app.session_state["workspace-sequence-scope"] == "One"
    assert app.session_state["workspace-sequence-id"] == "sequence-001"
    assert app.session_state["workspace-subview"] == subview
    assert _workbench_record(app.session_state) is record
    assert [tab.label for tab in app.tabs] == [
        "Method Comparison",
        "Markov Chain",
        "Hidden Markov Model",
        "Observed Shannon Entropy",
    ]
    active_tab = next(
        tab for tab in app.tabs if tab.label == "Observed Shannon Entropy"
    )
    active_nodes = tuple(active_tab)
    active_keys = [node.key for node in active_nodes]
    assert "workspace-sequence-scope" in active_keys
    assert "workspace-subview" in active_keys
    projection_index = next(
        index
        for index, node in enumerate(active_nodes)
        if isinstance(node, Markdown) and "Selected sequence" in node.value
    )
    assert active_keys.index("workspace-sequence-scope") < active_keys.index(
        "workspace-subview"
    ) < projection_index
    assert not any(
        button.label == "Calculate selected methods"
        for expander in app.expander
        for button in expander.button
    )
    assert any(
        button.label == "Calculate selected methods" for button in app.button
    )


def test_workspace_when_one_sequence_is_selected_renders_one_detail_slice() -> None:
    # Given
    app = _app()
    app.session_state[TAB_KEY] = "Markov Chain"
    app.session_state["workspace-sequence-scope"] = "One"
    app.session_state["workspace-sequence-id"] = "sequence-002"
    app.session_state["workspace-subview"] = "Overview"
    app = _batch(app, 3)

    # When
    app = _calculate(app)
    app = _set_key(app, TAB_KEY, "Markov Chain")

    # Then
    assert len(app.get("plotly_chart")) <= 1
    markov_tab = next(item for item in app.tabs if item.label == "Markov Chain")
    tab_text = "\n".join(item.value for item in markov_tab.markdown)
    assert "sequence-002" in tab_text
    assert "sequence-001" not in tab_text
    assert "sequence-003" not in tab_text
    assert not any("sequence-" in item.label.lower() for item in app.expander)


def test_markov_evidence_when_one_per_sequence_model_is_selected_shows_diagnostics(
) -> None:
    # Given
    app = _batch(_first_order_per_sequence(_app()), 3)
    app = _calculate(app)
    scope = _choice(app, "workspace-sequence-scope")
    _ = scope.set_value("One")
    _ = app.run()
    sequence = next(
        item for item in app.selectbox if item.key == "workspace-sequence-id"
    )
    _ = sequence.set_value("sequence-002")
    _ = app.run()

    # When
    view = _choice(app, "workspace-subview")
    _ = view.set_value("Evidence")
    _ = app.run()
    app = _set_key(app, TAB_KEY, "Markov Chain")

    # Then
    transition_frames = [
        item.value for item in app.dataframe if "Current state" in item.value.columns
    ]
    assert len(transition_frames) == 1
    assert [metric.label for metric in app.metric] == [
        "Whole-model empirical conditional entropy (bits)",
        "Long-run stationary distribution",
        "Long-run entropy rate (bits/symbol)",
    ]
    assert not app.exception


def test_markov_evidence_when_all_per_sequence_models_are_selected_stays_bounded(
) -> None:
    # Given
    app = _batch(_first_order_per_sequence(_app()), 3)
    app = _calculate(app)
    scope = _choice(app, "workspace-sequence-scope")
    _ = scope.set_value("All")
    _ = app.run()

    # When
    view = _choice(app, "workspace-subview")
    _ = view.set_value("Evidence")
    _ = app.run()
    app = _set_key(app, TAB_KEY, "Markov Chain")

    # Then
    assert not any(
        item.label.startswith("Advanced Markov Statistics") for item in app.expander
    )
    assert not any("Current state" in item.value.columns for item in app.dataframe)
    assert [item.value for item in app.info] == [
        joined_text(
            (
                "Select One sequence scope to inspect independently fitted ",
                "Markov model transition evidence and advanced statistics.",
            )
        )
    ]
    assert not app.exception


def test_workspace_when_all_has_one_hundred_records_keeps_rendering_bounded() -> None:
    # Given
    app = _app()
    app.session_state[TAB_KEY] = "Markov Chain"
    app.session_state["workspace-sequence-scope"] = "All"
    app.session_state["workspace-subview"] = "Overview"
    app = _batch(app, 100)

    # When
    app = _calculate(app)

    # Then
    assert len(app.get("plotly_chart")) <= 1
    assert len(app.metric) <= 8
    assert len(app.dataframe) <= 3
    assert all(len(item.value) <= 25 for item in app.dataframe)
    assert not any("sequence-" in item.label.lower() for item in app.expander)


def test_hmm_tab_when_disabled_is_clean_and_nonnumeric() -> None:
    # Given
    app = _calculate(_app())

    # When
    app = _set_key(app, TAB_KEY, "Hidden Markov Model")

    # Then
    assert any("HMM disabled" in item.value for item in app.info)
    assert not app.metric
    assert not app.dataframe
    assert not any("HMM" in item.label for item in app.download_button)


def test_hmm_tab_when_selected_but_uncalculated_is_clean_and_nonnumeric() -> None:
    # Given
    app = _app(ALL_METHODS)

    # When
    app = _set_key(app, TAB_KEY, "Hidden Markov Model")

    # Then
    assert any("Not calculated" in item.value for item in app.info)
    assert not app.metric
    assert not app.dataframe
    assert not any("HMM" in item.label for item in app.download_button)


def test_hmm_when_stale_hides_values_while_other_methods_remain_current() -> None:
    # Given
    app = _calculate(_app(ALL_METHODS))
    preset = next(item for item in app.text_input if item.label == "Preset name")

    # When
    _ = preset.set_value("changed after calculation")
    _ = app.run()
    app = _set_key(app, TAB_KEY, "Hidden Markov Model")

    # Then
    assert any("Recalculation required" in item.value for item in app.warning)
    assert not app.metric
    assert not app.dataframe
    assert not any("HMM" in item.label for item in app.download_button)
    app = _set_key(app, TAB_KEY, "Markov Chain")
    assert app.metric or app.dataframe
    app = _set_key(app, TAB_KEY, "Observed Shannon Entropy")
    metrics = {metric.label: metric.value for metric in app.metric}
    assert tuple(metrics) == (
        "Selected records",
        "Mean entropy (bits)",
        "Median entropy (bits)",
        "Predictive confidence",
    )
    assert metrics["Predictive confidence"] == "Not applicable"
    assert not app.dataframe
    assert not app.exception
