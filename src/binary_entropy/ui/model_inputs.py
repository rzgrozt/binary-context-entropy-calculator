"""Probability controls for one editable two-state binary HMM."""

import math
from typing import Final

import streamlit as st

from binary_entropy.presentation import format_decimal
from binary_entropy.ui.state import ModelForm
from binary_entropy.ui.text import joined_text

INITIAL_0_KEY: Final = "initial_0"
INITIAL_1_KEY: Final = "initial_1"
TRANSITION_KEYS: Final = (
    ("transition_00", "transition_01"),
    ("transition_10", "transition_11"),
)
EMISSION_KEYS: Final = (
    ("emission_00", "emission_01"),
    ("emission_10", "emission_11"),
)


def render_probability_inputs(
    state_labels: tuple[str, str],
    observable_labels: tuple[str, str],
) -> ModelForm:
    """Render initial, transition, and emission controls with orientations."""
    _ = st.subheader("Initial hidden-state distribution, pi")
    initial = _probability_row(
        (
            f"Initial probability for {state_labels[0]}",
            f"Initial probability for {state_labels[1]}",
        ),
        (INITIAL_0_KEY, INITIAL_1_KEY),
    )
    _ = st.caption(f"Initial sum: {format_decimal(math.fsum(initial))}. Required: 1.")
    _ = st.subheader("Transition matrix, T")
    _ = st.caption("Rows are current hidden states; columns are next hidden states.")
    transition = tuple(
        _labeled_probability_row(
            f"From {state_labels[row]}",
            (
                f"Transition P({state_labels[0]} | {state_labels[row]})",
                f"Transition P({state_labels[1]} | {state_labels[row]})",
            ),
            TRANSITION_KEYS[row],
        )
        for row in range(2)
    )
    _ = st.subheader("Emission matrix, E")
    _ = st.caption("Rows are hidden states; columns are observed variables.")
    emission = tuple(
        _labeled_probability_row(
            f"Given {state_labels[row]}",
            (
                f"Emission P({observable_labels[0]} | {state_labels[row]})",
                f"Emission P({observable_labels[1]} | {state_labels[row]})",
            ),
            EMISSION_KEYS[row],
        )
        for row in range(2)
    )
    _ = st.caption(
        joined_text(
            (
                "Every probability must be in [0, 1]. The initial vector and each ",
                "matrix row must sum to 1 within tolerance 0.000000000001.",
            )
        )
    )
    return ModelForm(
        state_labels=state_labels,
        observable_labels=observable_labels,
        initial=initial,
        transition=(transition[0], transition[1]),
        emission=(emission[0], emission[1]),
    )


def probability_widget_entries(model: ModelForm) -> tuple[tuple[str, float], ...]:
    """Return stable widget keys and probability values for one model."""
    return (
        (INITIAL_0_KEY, model.initial[0]),
        (INITIAL_1_KEY, model.initial[1]),
        (TRANSITION_KEYS[0][0], model.transition[0][0]),
        (TRANSITION_KEYS[0][1], model.transition[0][1]),
        (TRANSITION_KEYS[1][0], model.transition[1][0]),
        (TRANSITION_KEYS[1][1], model.transition[1][1]),
        (EMISSION_KEYS[0][0], model.emission[0][0]),
        (EMISSION_KEYS[0][1], model.emission[0][1]),
        (EMISSION_KEYS[1][0], model.emission[1][0]),
        (EMISSION_KEYS[1][1], model.emission[1][1]),
    )


def _labeled_probability_row(
    row_label: str,
    labels: tuple[str, str],
    keys: tuple[str, str],
) -> tuple[float, float]:
    _ = st.markdown(f"**{row_label}**")
    row = _probability_row(labels, keys)
    _ = st.caption(f"Row sum: {format_decimal(math.fsum(row))}. Required: 1.")
    return row


def _probability_row(
    labels: tuple[str, str],
    keys: tuple[str, str],
) -> tuple[float, float]:
    row_key = keys[0].replace("_", "-")
    with st.container(key=f"probability-row-{row_key}"):
        columns = st.columns(2)
        values = tuple(
            columns[index].number_input(
                labels[index],
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.12f",
                key=keys[index],
            )
            for index in range(2)
        )
    return values[0], values[1]
