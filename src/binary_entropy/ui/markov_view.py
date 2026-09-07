"""Bounded compatibility renderer for submitted first-order Markov results."""

import streamlit as st

from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.ui.markov_results import (
    markov_prefix_dataframe,
    markov_record_dataframe,
)
from binary_entropy.ui.text import joined_text


def render_markov_result(analysis: MarkovBatchAnalysis) -> None:
    """Render one bounded summary and evidence table for legacy callers."""
    _ = st.subheader("Markov Chain")
    _ = st.caption(
        joined_text(
            (
                f"First-order {analysis.result_scope.value.replace('_', ' ')}; ",
                "records remain in deterministic input order.",
            )
        )
    )
    _ = st.dataframe(
        markov_record_dataframe(analysis).head(25),
        hide_index=True,
        width="stretch",
        height="content",
    )
    record = analysis.records[0]
    _ = st.markdown(f"**Sequence evidence:** `{record.sequence_id}`")
    _ = st.dataframe(
        markov_prefix_dataframe(record, analysis.model.observable_labels).head(25),
        hide_index=True,
        width="stretch",
        height="content",
    )
