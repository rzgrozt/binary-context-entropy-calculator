"""Stable Streamlit keys and typed workspace presentation controls."""

from enum import StrEnum
from typing import Final, assert_never

import streamlit as st

from binary_entropy.records import SequenceDataset
from binary_entropy.ui.workspace_selection import (
    SequenceScope,
    WorkspaceSelection,
    reconcile_selection,
)

TAB_KEY: Final = "workspace-method-tabs"
SCOPE_KEY: Final = "workspace-sequence-scope"
SEQUENCE_KEY: Final = "workspace-sequence-id"
SEQUENCES_KEY: Final = "workspace-sequence-ids"
SUBVIEW_KEY: Final = "workspace-subview"
MAX_MULTIPLE_SELECTIONS: Final = 12


class WorkspaceSubview(StrEnum):
    """Available result projections beneath the active method tab."""

    OVERVIEW = "Overview"
    COMPARE = "Compare"
    EVIDENCE = "Evidence"


def render_workspace_selection(dataset: SequenceDataset) -> WorkspaceSelection:
    """Render persistent native scope and sequence controls."""
    identifiers = tuple(str(record.sequence_id) for record in dataset.records)
    scope_value = st.segmented_control(
        "Sequence scope",
        options=tuple(scope.value for scope in SequenceScope),
        default=SequenceScope.ALL.value,
        key=SCOPE_KEY,
    )
    scope = SequenceScope(scope_value or SequenceScope.ALL.value)
    match scope:
        case SequenceScope.ONE:
            if st.session_state.get(SEQUENCE_KEY) not in identifiers:
                st.session_state[SEQUENCE_KEY] = identifiers[0]
            selected = st.selectbox(
                "Sequence identifier",
                options=identifiers,
                key=SEQUENCE_KEY,
            )
            selection = WorkspaceSelection(
                scope,
                (selected,),
            )
            return reconcile_selection(selection, dataset)
        case SequenceScope.MULTIPLE:
            stored = st.session_state.get(SEQUENCES_KEY, ())
            st.session_state[SEQUENCES_KEY] = [
                identifier for identifier in identifiers if identifier in stored
            ][:MAX_MULTIPLE_SELECTIONS]
            clicked = st.multiselect(
                "Sequence identifiers",
                options=identifiers,
                max_selections=MAX_MULTIPLE_SELECTIONS,
                key=SEQUENCES_KEY,
            )
            selection = WorkspaceSelection(scope, tuple(clicked))
            return reconcile_selection(selection, dataset)
        case remaining:
            if remaining is SequenceScope.ALL:
                _ = st.caption(
                    f"All {len(identifiers)} accepted records are included."
                )
                selection = WorkspaceSelection(remaining, identifiers)
                return reconcile_selection(selection, dataset)
            assert_never(remaining)


def render_subview_control() -> WorkspaceSubview:
    """Render the persistent Overview, Compare, and Evidence selector."""
    value = st.segmented_control(
        "Result view",
        options=tuple(item.value for item in WorkspaceSubview),
        default=WorkspaceSubview.OVERVIEW.value,
        key=SUBVIEW_KEY,
    )
    return WorkspaceSubview(value or WorkspaceSubview.OVERVIEW.value)
