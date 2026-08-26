"""Compact setup gate and persistent method selection."""

from typing import Final

import streamlit as st

from binary_entropy.ui.hmm_session import hmm_control_state, write_hmm_widgets
from binary_entropy.ui.workbench_state import METHOD_OPTIONS, MethodChoice

METHODS_KEY: Final = "workbench_selected_methods"
SETUP_COMPLETE_KEY: Final = "workbench_setup_complete"


def render_method_selection() -> tuple[MethodChoice, ...]:
    """Render the shared method selector and return its typed ordered values."""
    if METHODS_KEY not in st.session_state:
        st.session_state[METHODS_KEY] = [MethodChoice.MARKOV.value]
    selected = st.multiselect(
        "Analysis methods",
        options=tuple(method.value for method in METHOD_OPTIONS),
        key=METHODS_KEY,
        help="Choose one or more methods. Controls appear only for selected methods.",
        on_change=_synchronize_selected_hmm_widgets,
    )
    return tuple(MethodChoice(value) for value in selected)


def setup_is_complete() -> bool:
    """Return whether the explicit setup action has opened the workbench."""
    return st.session_state.get(SETUP_COMPLETE_KEY, False) is True


def render_continue(methods: tuple[MethodChoice, ...]) -> None:
    """Render the explicit setup action without performing a calculation."""
    _ = st.button(
        "Continue",
        type="primary",
        disabled=not methods,
        on_click=_complete_setup,
    )


def _complete_setup() -> None:
    st.session_state[SETUP_COMPLETE_KEY] = True


def _synchronize_selected_hmm_widgets() -> None:
    if MethodChoice.HMM.value not in st.session_state.get(METHODS_KEY, ()):
        return
    state = hmm_control_state()
    write_hmm_widgets(state.model, state.preset_name)
