"""Stable native Streamlit controls for one editable binary HMM."""

from typing import Final

import streamlit as st

from binary_entropy.domain import BinaryLabels
from binary_entropy.errors import BinaryEntropyError
from binary_entropy.parsing import parse_sequence
from binary_entropy.ui.model_inputs import (
    probability_widget_entries,
    render_probability_inputs,
)
from binary_entropy.ui.preset_inputs import (
    OBSERVABLE_0_KEY,
    OBSERVABLE_1_KEY,
    PRESET_NAME_KEY,
    STATE_0_KEY,
    STATE_1_KEY,
    render_preset_import,
    reset_example_model,
)
from binary_entropy.ui.state import (
    ActualTargetChoice,
    CalculatorForm,
    default_form,
)
from binary_entropy.ui.text import joined_text

SEQUENCE_KEY: Final = "sequence_text"
ACTUAL_TARGET_KEY: Final = "actual_target"
SEQUENCE_ID_KEY: Final = "sequence_id"


def initialize_form_widgets() -> None:
    """Populate every keyed control once with the demonstration values."""
    form = default_form()
    entries = (
        (STATE_0_KEY, form.model.state_labels[0]),
        (STATE_1_KEY, form.model.state_labels[1]),
        (OBSERVABLE_0_KEY, form.model.observable_labels[0]),
        (OBSERVABLE_1_KEY, form.model.observable_labels[1]),
        *probability_widget_entries(form.model),
        (SEQUENCE_KEY, form.sequence_text),
        (ACTUAL_TARGET_KEY, form.actual_target),
        (SEQUENCE_ID_KEY, form.sequence_id),
        (PRESET_NAME_KEY, form.preset_name),
    )
    for key, value in entries:
        if key not in st.session_state:
            st.session_state[key] = value


def render_form() -> CalculatorForm:
    """Render model, sequence, metadata, and preset-import controls."""
    initialize_form_widgets()
    _ = st.header("Model")
    _ = st.info(
        joined_text(
            (
                "Demo HMM: the initial values are a hand-verified two-state example. ",
                "Edit any parameter before calculation.",
            )
        )
    )
    state_columns = st.columns(2)
    state_0 = state_columns[0].text_input("Hidden state 1 label", key=STATE_0_KEY)
    state_1 = state_columns[1].text_input("Hidden state 2 label", key=STATE_1_KEY)
    observable_columns = st.columns(2)
    observable_0 = observable_columns[0].text_input(
        "Variable 1 label", key=OBSERVABLE_0_KEY
    )
    observable_1 = observable_columns[1].text_input(
        "Variable 2 label", key=OBSERVABLE_1_KEY
    )
    _ = st.caption(
        "Labels must be nonempty, distinct, and contain no commas or line breaks."
    )
    state_labels = (state_0 or "", state_1 or "")
    observable_labels = (observable_0 or "", observable_1 or "")
    model = render_probability_inputs(state_labels, observable_labels)
    _ = st.button(
        "Reset example model",
        on_click=reset_example_model,
        type="secondary",
    )
    _ = st.header("Sequence")
    sequence_text = st.text_area(
        "Observed sequence",
        key=SEQUENCE_KEY,
        height=112,
        help="Separate variable labels with commas, spaces, or line breaks.",
    )
    _render_sequence_feedback(sequence_text or "", state_labels, observable_labels)
    target_labels = {
        ActualTargetChoice.NONE: "None",
        ActualTargetChoice.FIRST: observable_labels[0] or "Variable 1",
        ActualTargetChoice.SECOND: observable_labels[1] or "Variable 2",
    }
    actual_target = st.radio(
        "Optional actual next target",
        options=tuple(ActualTargetChoice),
        format_func=target_labels.__getitem__,
        key=ACTUAL_TARGET_KEY,
        horizontal=True,
        help="Assessed separately against the final predictive distribution.",
    )
    metadata_columns = st.columns(2)
    sequence_id = metadata_columns[0].text_input(
        "Sequence ID for candidate-summary CSV", key=SEQUENCE_ID_KEY
    )
    preset_name = metadata_columns[1].text_input("Preset name", key=PRESET_NAME_KEY)
    form = CalculatorForm(
        model=model,
        sequence_text=sequence_text or "",
        actual_target=actual_target or ActualTargetChoice.NONE,
        sequence_id=sequence_id or "",
        preset_name=preset_name or "",
    )
    render_preset_import(form)
    return form


def _render_sequence_feedback(
    text: str,
    state_labels: tuple[str, str],
    observable_labels: tuple[str, str],
) -> None:
    try:
        labels = BinaryLabels(states=state_labels, observables=observable_labels)
        sequence = parse_sequence(text, labels)
    except BinaryEntropyError as error:
        _ = st.error(str(error))
        return
    length = len(sequence)
    _ = st.caption(f"Sequence length: {length}. Context depth: {length}.")
