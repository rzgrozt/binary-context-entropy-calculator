"""Persistent HMM controls independent of Streamlit widget lifecycle."""

from dataclasses import dataclass, replace
from typing import Final

import streamlit as st

from binary_entropy.ui.state import ModelForm, default_form

HMM_CONTROL_STATE_KEY: Final = "_method_state_hidden_markov_model"
STATE_0_KEY: Final = "hmm_state_label_0"
STATE_1_KEY: Final = "hmm_state_label_1"
PRESET_NAME_KEY: Final = "hmm_preset_name"
INITIAL_KEY: Final = "hmm_initial_p"
INITIAL_COMPLEMENT_KEY: Final = "hmm_initial_complement"
TRANSITION_KEYS: Final = ("hmm_transition_0_p", "hmm_transition_1_p")
TRANSITION_COMPLEMENT_KEYS: Final = (
    "hmm_transition_0_complement",
    "hmm_transition_1_complement",
)
EMISSION_KEYS: Final = ("hmm_emission_0_p", "hmm_emission_1_p")
EMISSION_COMPLEMENT_KEYS: Final = (
    "hmm_emission_0_complement",
    "hmm_emission_1_complement",
)


@dataclass(frozen=True, slots=True)
class HMMControlState:
    """HMM model inputs retained while their widgets are not rendered."""

    model: ModelForm
    preset_name: str


def hmm_control_state() -> HMMControlState:
    """Return the persistent HMM state, creating the example state once."""
    match st.session_state.get(HMM_CONTROL_STATE_KEY):
        case HMMControlState() as state:
            return state
        case _:
            form = default_form()
            state = HMMControlState(form.model, form.preset_name)
            st.session_state[HMM_CONTROL_STATE_KEY] = state
            return state


def hydrate_hmm_widgets() -> None:
    """Restore pruned HMM widget keys before any HMM widget is constructed."""
    for key, value in _widget_entries(hmm_control_state()):
        if key not in st.session_state:
            st.session_state[key] = value


def store_hmm_controls(model: ModelForm, preset_name: str) -> None:
    """Snapshot exact visible HMM sources into lifecycle-independent state."""
    st.session_state[HMM_CONTROL_STATE_KEY] = HMMControlState(model, preset_name)


def stored_hmm_model(observable_labels: tuple[str, str]) -> ModelForm:
    """Return hidden HMM inputs with the current shared observable labels."""
    return replace(hmm_control_state().model, observable_labels=observable_labels)


def write_hmm_widgets(model: ModelForm, preset_name: str) -> None:
    """Update persistent HMM state and every visible HMM widget value together."""
    state = HMMControlState(model, preset_name)
    st.session_state[HMM_CONTROL_STATE_KEY] = state
    for key, value in _widget_entries(state):
        st.session_state[key] = value


def _widget_entries(state: HMMControlState) -> tuple[tuple[str, str | float], ...]:
    model = state.model
    initial = model.initial[0]
    transition_0 = model.transition[0][0]
    transition_1 = model.transition[1][0]
    emission_0 = model.emission[0][0]
    emission_1 = model.emission[1][0]
    return (
        (STATE_0_KEY, model.state_labels[0]),
        (STATE_1_KEY, model.state_labels[1]),
        (PRESET_NAME_KEY, state.preset_name),
        (INITIAL_KEY, initial),
        (INITIAL_COMPLEMENT_KEY, 1.0 - initial),
        (TRANSITION_KEYS[0], transition_0),
        (TRANSITION_COMPLEMENT_KEYS[0], 1.0 - transition_0),
        (TRANSITION_KEYS[1], transition_1),
        (TRANSITION_COMPLEMENT_KEYS[1], 1.0 - transition_1),
        (EMISSION_KEYS[0], emission_0),
        (EMISSION_COMPLEMENT_KEYS[0], 1.0 - emission_0),
        (EMISSION_KEYS[1], emission_1),
        (EMISSION_COMPLEMENT_KEYS[1], 1.0 - emission_1),
    )
