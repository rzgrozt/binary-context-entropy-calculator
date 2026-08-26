"""Transactional JSON preset controls and keyed model updates."""

from typing import Final

import streamlit as st

from binary_entropy.ui.hmm_session import write_hmm_widgets
from binary_entropy.ui.inputs import OBSERVABLE_A_KEY, OBSERVABLE_B_KEY
from binary_entropy.ui.session import (
    clear_calculation,
    clear_preset_import,
    preset_import_outcome,
    store_preset_import,
)
from binary_entropy.ui.state import (
    CalculatorForm,
    PresetImportFailure,
    PresetImportSuccess,
    default_form,
    import_preset,
)
from binary_entropy.ui.text import joined_text

OBSERVABLE_0_KEY: Final = OBSERVABLE_A_KEY
OBSERVABLE_1_KEY: Final = OBSERVABLE_B_KEY


def render_preset_import(form: CalculatorForm) -> None:
    """Render upload and explicit load controls for model-only presets."""
    _ = st.subheader("Model preset JSON")
    _ = st.caption(
        joined_text(
            (
                "A preset contains the model labels and probabilities only. It does ",
                "not contain the observed sequence or optional actual target.",
            )
        )
    )
    _ = st.button(
        "Reset HMM example model",
        on_click=reset_example_model,
        type="secondary",
        key="hmm_reset_example",
    )
    uploaded = st.file_uploader(
        "Upload model preset JSON",
        type=("json",),
        accept_multiple_files=False,
        key="hmm_preset_upload",
    )
    payload = uploaded.getvalue() if uploaded is not None else None
    _ = st.button(
        "Load preset JSON",
        on_click=_load_preset,
        args=(payload, form),
        type="secondary",
        key="hmm_load_preset",
    )
    match preset_import_outcome():  # noqa: RUF100  # noqa: MATCH_OK
        case PresetImportSuccess():
            _ = st.success(
                "Preset loaded. Model inputs updated; no calculation was run."
            )
        case PresetImportFailure(message=message):
            _ = st.error(message)
        case None:
            pass


def write_model_widgets(form: CalculatorForm) -> None:
    """Update every model widget from one validated form transaction."""
    entries = (
        (OBSERVABLE_0_KEY, form.model.observable_labels[0]),
        (OBSERVABLE_1_KEY, form.model.observable_labels[1]),
    )
    for key, value in entries:
        st.session_state[key] = value
    write_hmm_widgets(form.model, form.preset_name)


def reset_example_model() -> None:
    """Restore the hand-verified model without running a calculation."""
    write_model_widgets(default_form())
    clear_calculation()
    clear_preset_import()


def _load_preset(payload: bytes | None, current: CalculatorForm) -> None:
    if payload is None:
        store_preset_import(
            PresetImportFailure(message="Select a JSON preset to load.")
        )
        return
    outcome = import_preset(payload, current)
    store_preset_import(outcome)
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case PresetImportSuccess(form=imported):
            write_model_widgets(imported)
            clear_calculation()
        case PresetImportFailure():
            pass
