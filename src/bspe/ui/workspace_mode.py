"""Top-level workspace selection shared by both application modes."""

from enum import StrEnum
from typing import Final, assert_never

import streamlit as st

from bspe.ui.stimulus_controls import SEARCH_WIDGET_KEYS

WORKSPACE_MODE_KEY: Final = "workbench-workspace-mode"
_ANALYZER_WIDGET_KEYS: Final = frozenset(
    {
        "hmm_emission_0_complement",
        "hmm_emission_0_p",
        "hmm_emission_1_complement",
        "hmm_emission_1_p",
        "hmm_initial_complement",
        "hmm_initial_p",
        "hmm_preset_name",
        "hmm_state_label_0",
        "hmm_state_label_1",
        "hmm_transition_0_complement",
        "hmm_transition_0_p",
        "hmm_transition_1_complement",
        "hmm_transition_1_p",
        "markov_custom_alpha",
        "markov_estimation",
        "markov_prefix_mode",
        "markov_result_scope",
        "markov_workflow",
        "shared_actual_target",
        "shared_batch_sequences",
        "shared_csv_id_column",
        "shared_csv_sequence_column",
        "shared_csv_target_column",
        "shared_input_mode",
        "shared_observable_a",
        "shared_observable_b",
        "shared_sequence_id",
        "shared_single_sequence",
        "vmm_custom_alpha",
        "vmm_minimum_support",
        "vmm_smoothing",
        "workbench_selected_methods",
        "workspace-method-tabs",
        "workspace-sequence-id",
        "workspace-sequence-ids",
        "workspace-sequence-scope",
        "workspace-subview",
    }
)
_STIMULUS_WIDGET_KEYS: Final = SEARCH_WIDGET_KEYS | frozenset(
    {
        "stimulus-assignment-seed-input",
        "stimulus-match-a-proportion",
        "stimulus-match-effective-depth",
        "stimulus-match-entropy",
        "stimulus-match-prediction-strength",
        "stimulus-match-run-structure",
        "stimulus-match-switch-rate",
        "stimulus-search-candidate-id",
    }
)


class WorkspaceMode(StrEnum):
    """Sibling workbench projections available to the researcher."""

    ANALYZER = "Analyzer"
    STIMULUS_SEARCH = "Stimulus Search"


WORKSPACE_MODE_OPTIONS: Final = tuple(mode.value for mode in WorkspaceMode)


def render_workspace_mode() -> WorkspaceMode:
    """Render the presentation-only top-level workspace selector."""
    selected = st.segmented_control(
        "Workspace mode",
        options=WORKSPACE_MODE_OPTIONS,
        default=WorkspaceMode.ANALYZER.value,
        key=WORKSPACE_MODE_KEY,
    )
    return WorkspaceMode(selected or WorkspaceMode.ANALYZER.value)


def preserve_hidden_workspace_widget_state(mode: WorkspaceMode) -> None:
    """Keep sibling-mode widget values through Streamlit cleanup."""
    match mode:
        case WorkspaceMode.ANALYZER:
            hidden_keys = _STIMULUS_WIDGET_KEYS
        case remaining if remaining is not WorkspaceMode.STIMULUS_SEARCH:
            assert_never(remaining)
        case WorkspaceMode.STIMULUS_SEARCH:
            hidden_keys = _ANALYZER_WIDGET_KEYS
    for key in hidden_keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
