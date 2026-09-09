"""Streamlit entry point for the binary sequence scientific workbench."""

from pathlib import Path
from typing import Final, assert_never

import streamlit as st

from binary_entropy.ui.results_view import render_results
from binary_entropy.ui.session import workbench_submission_failure
from binary_entropy.ui.sidebar import render_sidebar
from binary_entropy.ui.stimulus_view import (
    render_stimulus_search,
    render_stimulus_search_sidebar,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workspace_mode import (
    WorkspaceMode,
    preserve_hidden_workspace_widget_state,
    render_workspace_mode,
)

STYLES_PATH: Final = Path(__file__).parent / "assets" / "styles.css"
PAGE_TITLE: Final = "Binary Sequence Probability, Prediction & Entropy Workbench"

_ = st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
_ = st.html(STYLES_PATH)

with st.sidebar:
    mode = render_workspace_mode()
    preserve_hidden_workspace_widget_state(mode)

match mode:
    case WorkspaceMode.ANALYZER:
        with st.sidebar:
            sidebar = render_sidebar()
        form = sidebar.form
        calculation = sidebar.calculation
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
            with st.container(key="results-region"):
                render_results(
                    form,
                    calculation,
                    workbench_submission_failure(form),
                )
    case remaining if remaining is not WorkspaceMode.STIMULUS_SEARCH:
        assert_never(remaining)
    case WorkspaceMode.STIMULUS_SEARCH:
        with st.sidebar:
            render_stimulus_search_sidebar()
        render_stimulus_search()
