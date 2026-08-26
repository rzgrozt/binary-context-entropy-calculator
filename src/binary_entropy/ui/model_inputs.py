"""Complement-derived controls for one fixed two-state binary HMM."""

import streamlit as st

from binary_entropy.ui.hmm_session import (
    EMISSION_COMPLEMENT_KEYS,
    EMISSION_KEYS,
    INITIAL_COMPLEMENT_KEY,
    INITIAL_KEY,
    TRANSITION_COMPLEMENT_KEYS,
    TRANSITION_KEYS,
)
from binary_entropy.ui.state import ModelForm
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT, format_ui_decimal


def render_probability_inputs(
    state_labels: tuple[str, str],
    observable_labels: tuple[str, str],
) -> ModelForm:
    """Render three HMM expanders with one editable value per binary row."""
    with st.expander("Initial hidden-state distribution", expanded=True):
        initial = _probability_row(
            f"Initial probability for {state_labels[0]}",
            f"Derived probability for {state_labels[1]} (1 - p)",
            INITIAL_KEY,
            INITIAL_COMPLEMENT_KEY,
        )
    with st.expander("Transition matrix", expanded=True):
        _ = st.caption("Rows are current hidden states; columns are next states.")
        transition = tuple(
            _probability_row(
                f"Transition probability to {state_labels[0]} from {state_labels[row]}",
                joined_text(
                    (
                        f"Derived probability to {state_labels[1]} from ",
                        f"{state_labels[row]} (1 - p)",
                    )
                ),
                TRANSITION_KEYS[row],
                TRANSITION_COMPLEMENT_KEYS[row],
            )
            for row in range(2)
        )
    with st.expander("Emission matrix", expanded=True):
        _ = st.caption("Rows are hidden states; columns are observed symbols.")
        emission = tuple(
            _probability_row(
                joined_text(
                    (
                        f"Emission probability for {observable_labels[0]} given ",
                        state_labels[row],
                    )
                ),
                joined_text(
                    (
                        f"Derived probability for {observable_labels[1]} given ",
                        f"{state_labels[row]} (1 - p)",
                    )
                ),
                EMISSION_KEYS[row],
                EMISSION_COMPLEMENT_KEYS[row],
            )
            for row in range(2)
        )
    return ModelForm(
        state_labels=state_labels,
        observable_labels=observable_labels,
        initial=initial,
        transition=(transition[0], transition[1]),
        emission=(emission[0], emission[1]),
    )


def _probability_row(
    source_label: str,
    complement_label: str,
    source_key: str,
    complement_key: str,
) -> tuple[float, float]:
    with st.container(key=f"probability-row-{source_key.replace('_', '-')}"):
        columns = st.columns(2)
        source = columns[0].number_input(
            source_label,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format=UI_NUMBER_FORMAT,
            key=source_key,
        )
        complement = 1.0 - source
        st.session_state[complement_key] = complement
        _ = columns[1].number_input(
            complement_label,
            min_value=0.0,
            max_value=1.0,
            format=UI_NUMBER_FORMAT,
            key=complement_key,
            disabled=True,
        )
        _ = st.caption(f"Row sum: {format_ui_decimal(source + complement)}.")
    return source, complement
