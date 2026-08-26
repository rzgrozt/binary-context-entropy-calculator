"""Composition of selected method controls and shared data intake."""

import streamlit as st

from binary_entropy.ui.inputs import render_intake, render_observable_labels
from binary_entropy.ui.method_controls import (
    default_markov_controls,
    hidden_hmm_controls,
    render_hmm_controls,
    render_markov_controls,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workbench_state import MethodChoice, WorkbenchForm


def render_workbench_form(methods: tuple[MethodChoice, ...]) -> WorkbenchForm:
    """Render only selected method controls and one shared intake."""
    observable_labels = render_observable_labels()
    if MethodChoice.MARKOV in methods:
        markov = render_markov_controls()
    else:
        markov = default_markov_controls()
    if MethodChoice.HMM in methods:
        hmm_model, preset_name = render_hmm_controls(observable_labels)
    else:
        hmm_model, preset_name = hidden_hmm_controls(observable_labels)
    if MethodChoice.SHANNON in methods:
        _ = st.subheader("Observed Shannon Entropy controls")
        _ = st.caption(
            joined_text(
                (
                    "Observed Shannon entropy has no model controls; it summarizes ",
                    "submitted symbols.",
                )
            )
        )
    intake = render_intake(observable_labels)
    return WorkbenchForm(
        methods=methods,
        intake=intake,
        markov=markov,
        hmm_model=hmm_model,
        preset_name=preset_name,
    )
