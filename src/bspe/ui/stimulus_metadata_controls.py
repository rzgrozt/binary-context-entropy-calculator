"""Explicit post-analysis presentation mapping controls."""

import streamlit as st

from bspe.stimulus_metadata import PresentationMetadata
from bspe.stimulus_search_types import InvalidStimulusSearchConfigurationError
from bspe.ui.stimulus_session import store_presentation_metadata


def render_presentation_controls() -> None:
    """Assign experiment labels without regenerating statistical sequences."""
    with st.expander("Presentation mapping and condition", expanded=False):
        _ = st.caption(
            "The scientific sequence and predictions remain A/B. These labels are "
            "export metadata for the experiment, such as circle/triangle."
        )
        first = st.text_input("Present A as", value="A", key="stimulus-map-a")
        second = st.text_input("Present B as", value="B", key="stimulus-map-b")
        condition = st.text_input(
            "Condition label (optional)", key="stimulus-condition"
        )
        if st.button("Apply presentation metadata"):
            try:
                metadata = PresentationMetadata((first, second), condition)
            except InvalidStimulusSearchConfigurationError as error:
                _ = st.error(str(error))
            else:
                store_presentation_metadata(metadata)
                st.rerun()
