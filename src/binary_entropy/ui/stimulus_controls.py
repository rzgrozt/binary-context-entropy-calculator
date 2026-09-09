"""Native controls for bounded stimulus search."""

from typing import Final

import streamlit as st

from binary_entropy import (
    InvalidStimulusSearchConfigurationError,
    InvalidVMMConfigurationError,
    StimulusSearchConfig,
    search_stimuli,
)
from binary_entropy.ui.help_text import UI_HELP
from binary_entropy.ui.stimulus_constraint_controls import (
    CONSTRAINT_WIDGET_KEYS,
    render_constraint_controls,
)
from binary_entropy.ui.stimulus_preference_controls import (
    PREFERENCE_WIDGET_KEYS,
    render_preference_controls,
)
from binary_entropy.ui.stimulus_session import (
    invalidate_stimulus_search,
    store_search_failure,
    store_search_snapshot,
)
from binary_entropy.ui.stimulus_vmm_controls import (
    VMM_WIDGET_KEYS,
    render_vmm_controls,
)

SEARCH_WIDGET_KEYS: Final = (
    frozenset(
        {
            "stimulus-search-sequence-length",
            "stimulus-search-desired-count",
            "stimulus-search-candidate-limit",
            "stimulus-search-seed",
        }
    )
    | CONSTRAINT_WIDGET_KEYS
    | PREFERENCE_WIDGET_KEYS
    | VMM_WIDGET_KEYS
)


def render_search_controls() -> None:
    """Render search settings and submit only on explicit action."""
    _ = st.subheader("Search configuration")
    sequence_length = int(
        st.number_input(
            "Sequence length",
            min_value=1,
            max_value=32,
            value=8,
            key="stimulus-search-sequence-length",
            on_change=invalidate_stimulus_search,
            help=UI_HELP["sequence_length"],
        )
    )
    desired_count = int(
        st.number_input(
            "Desired count",
            min_value=1,
            max_value=5_000,
            value=8,
            key="stimulus-search-desired-count",
            on_change=invalidate_stimulus_search,
            help=UI_HELP["desired_count"],
        )
    )
    candidate_limit = int(
        st.number_input(
            "Candidate limit",
            min_value=1,
            max_value=5_000,
            value=32,
            key="stimulus-search-candidate-limit",
            on_change=invalidate_stimulus_search,
            help=UI_HELP["candidate_limit"],
        )
    )
    raw_search_seed = st.text_input(
        "Search seed",
        value="17",
        key="stimulus-search-seed",
        help=UI_HELP["search_seed"],
        on_change=invalidate_stimulus_search,
    )
    vmm_values = render_vmm_controls()
    with st.expander("Hard constraints", expanded=False):
        constraint_values = render_constraint_controls()
    with st.expander("Soft preferences", expanded=False):
        preferences = render_preference_controls(sequence_length)
    if st.button("Search stimuli", type="primary"):
        try:
            search_seed = int(raw_search_seed, 10)
        except ValueError:
            error = InvalidStimulusSearchConfigurationError("seed", raw_search_seed)
            store_search_failure(str(error))
            return
        try:
            config = StimulusSearchConfig(
                sequence_length=sequence_length,
                desired_stimuli=desired_count,
                seed=search_seed,
                candidate_limit=candidate_limit,
                vmm_config=vmm_values.config(),
                constraints=constraint_values.constraints(),
                preferences=preferences,
            )
            store_search_snapshot(search_stimuli(config))
        except (
            InvalidStimulusSearchConfigurationError,
            InvalidVMMConfigurationError,
        ) as error:
            store_search_failure(str(error))
