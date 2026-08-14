"""Streamlit entry point for the predictive entropy calculator."""

from pathlib import Path
from typing import Final

import streamlit as st

from binary_entropy.ui.inputs import render_form
from binary_entropy.ui.results_view import render_results
from binary_entropy.ui.session import (
    calculation_record,
    store_calculation,
    submission_failure,
)
from binary_entropy.ui.state import calculate_form
from binary_entropy.ui.text import joined_text

STYLES_PATH: Final = Path(__file__).parent / "assets" / "styles.css"
SCIENTIFIC_WARNING: Final = joined_text(
    (
        "The entered sequence alone does not uniquely determine a next-target ",
        "probability distribution. HMM predictive results depend on the selected ",
        "HMM parameters.",
    )
)

_ = st.set_page_config(
    page_title="Binary Sequence Predictive Entropy Calculator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
_ = st.html(STYLES_PATH)
with st.container(key="calculator-layout"):
    with st.container(key="header-region"):
        _ = st.title("Binary Sequence Predictive Entropy Calculator")
        _ = st.markdown(
            joined_text(
                (
                    "Configure one two-state hidden Markov model (HMM), ",
                    "enter a binary sequence, and inspect next-target predictive ",
                    "entropy across every prefix.",
                )
            )
        )
        _ = st.markdown(
            joined_text(
                (
                    "**Predictive probabilities are conditional on the currently ",
                    "selected HMM.**",
                )
            )
        )

    with st.container(key="configuration-region"):
        form = render_form()
        _ = st.header("Calculate")
        _ = st.caption(
            "Inputs may be edited freely. No calculation runs automatically."
        )
        if st.button("Calculate entropy", type="primary"):
            store_calculation(calculate_form(form), form)

    with st.container(key="results-region"):
        render_results(form, calculation_record(), submission_failure(form))

    with st.container(key="interpretation-region"):
        _ = st.header("How to interpret this")
        _ = st.warning(SCIENTIFIC_WARNING)
        _ = st.markdown(
            joined_text(
                (
                    "- **HMM predictive entropy** measures uncertainty in the ",
                    "next observable ",
                    "under the selected model after a prefix.\n",
                    "- **Observed-symbol Shannon entropy** describes the ",
                    "composition of the ",
                    "entered sequence; it is not a next-target prediction.\n",
                    "- **Actual-target surprisal** is the realized ",
                    "self-information of an ",
                    "optional user-selected next target under the final prediction.",
                )
            )
        )
        with st.expander("Method and definitions"):
            _ = st.markdown(
                joined_text(
                    (
                        "The initial distribution `pi` is the hidden-state ",
                        "distribution at the first observation. Therefore, ",
                        "**depth 0 uses q1=pi E without transition**. After each ",
                        "observation, filtering forms the hidden posterior, ",
                        "applies `T` to obtain the next-hidden distribution, ",
                        "and applies `E` to obtain the next-observable predictive ",
                        "distribution. Predictive entropy is binary Shannon ",
                        "entropy in bits, and candidate ",
                        "surprisal is `-log2(probability)`.",
                    )
                )
            )
