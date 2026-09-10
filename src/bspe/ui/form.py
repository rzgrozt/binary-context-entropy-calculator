"""Composition of selected method controls and shared data intake."""

import streamlit as st

from bspe.ui.inputs import render_intake
from bspe.ui.method_controls import (
    default_markov_controls,
    hidden_hmm_controls,
    render_hmm_controls,
    render_markov_controls,
)
from bspe.ui.workbench_state import MethodChoice, WorkbenchForm


def render_workbench_form(
    methods: tuple[MethodChoice, ...],
    observable_labels: tuple[str, str],
) -> WorkbenchForm:
    """Render only selected method controls and one shared intake."""
    if MethodChoice.MARKOV in methods:
        with st.expander("Markov workflow settings", expanded=False):
            markov = render_markov_controls()
    else:
        markov = default_markov_controls()
    if MethodChoice.HMM in methods:
        with st.expander("Hidden Markov Model settings", expanded=False):
            hmm_model, preset_name = render_hmm_controls(observable_labels)
    else:
        hmm_model, preset_name = hidden_hmm_controls(observable_labels)
    with st.expander("Training data intake", expanded=False):
        intake = render_intake(observable_labels)
    return WorkbenchForm(
        methods=methods,
        intake=intake,
        markov=markov,
        hmm_model=hmm_model,
        preset_name=preset_name,
    )
