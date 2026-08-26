"""Streamlit entry point for the binary sequence scientific workbench."""

from pathlib import Path
from typing import Final

import streamlit as st

from binary_entropy.ui.form import render_workbench_form
from binary_entropy.ui.results_view import render_results
from binary_entropy.ui.session import (
    store_workbench_calculation,
    workbench_calculation_record,
    workbench_submission_failure,
)
from binary_entropy.ui.setup import (
    render_continue,
    render_method_selection,
    setup_is_complete,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workbench_state import calculate_workbench

STYLES_PATH: Final = Path(__file__).parent / "assets" / "styles.css"
PAGE_TITLE: Final = "Binary Sequence Probability, Prediction & Entropy Workbench"

_ = st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
_ = st.html(STYLES_PATH)

with st.container(key="workbench-layout"):
    with st.container(key="header-region"):
        _ = st.title(PAGE_TITLE)
        _ = st.markdown(
            joined_text(
                (
                    "Fit and compare selected binary-sequence methods without ",
                    "crossing record boundaries.",
                )
            )
        )

    if not setup_is_complete():
        with st.container(key="setup-region"):
            _ = st.subheader("Analysis setup")
            _ = st.caption(
                "Select the methods that apply. Markov Chain is selected by default."
            )
            setup_methods = render_method_selection()
            render_continue(setup_methods)
        st.stop()

    with st.container(key="workbench-columns"):
        configuration_column, results_column = st.columns(
            (4, 6),
            gap="large",
            vertical_alignment="top",
        )
        with configuration_column, st.container(key="configuration-region"):
            _ = st.header("Analysis configuration")
            methods = render_method_selection()
            if methods:
                _ = st.caption(
                    "Active methods: " + ", ".join(method.value for method in methods)
                )
            else:
                _ = st.error("Select at least one analysis method.")
            form = render_workbench_form(methods)
            _ = st.caption(
                joined_text(
                    (
                        "No calculation runs while editing. Submit the current ",
                        "immutable configuration explicitly.",
                    )
                )
            )
            if st.button(
                "Calculate selected methods",
                type="primary",
                disabled=not methods,
            ):
                store_workbench_calculation(calculate_workbench(form), form)
        with results_column, st.container(key="results-region"):
            render_results(
                form,
                workbench_calculation_record(),
                workbench_submission_failure(form),
            )