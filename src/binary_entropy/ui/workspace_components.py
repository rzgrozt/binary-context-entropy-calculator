"""Shared native components for bounded workspace projections."""

from collections.abc import Sequence
from typing import Final, assert_never

import pandas as pd
import streamlit as st

from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT
from binary_entropy.ui.workspace_metrics import MetricValue
from binary_entropy.ui.workspace_selection import SequenceScope

COMPARISON_ROW_LIMIT: Final = 25
EVIDENCE_ROW_LIMIT: Final = 50


def render_scope_heading(
    scope: SequenceScope,
    identifiers: Sequence[str],
) -> None:
    """Identify one record or summarize a batch without enumerating IDs."""
    match scope:
        case SequenceScope.ONE:
            _ = st.markdown(f"**Selected sequence:** {identifiers[0]}")
            return
        case SequenceScope.MULTIPLE:
            _ = st.markdown(f"**Selected sequence scope:** {len(identifiers)} records")
            return
        case remaining:
            if remaining is SequenceScope.ALL:
                _ = st.markdown(
                    f"**Selected sequence scope:** all {len(identifiers)} records"
                )
                return
            assert_never(remaining)


def render_metrics(values: tuple[MetricValue, ...]) -> None:
    """Render compact native metric rows with at most three columns."""
    for start in range(0, len(values), 3):
        row = values[start : start + 3]
        columns = st.columns(len(row))
        for column, metric in zip(columns, row, strict=True):
            _ = column.metric(metric.label, metric.value)


def render_bounded_dataframe(
    frame: pd.DataFrame,
    limit: int,
    order: str,
) -> None:
    """Render deterministic rows with an explicit truncation route."""
    visible = frame.head(limit)
    _ = st.caption(f"Deterministic order: {order}.")
    if len(frame) > len(visible):
        _ = st.caption(
            joined_text(
                (
                    f"Showing {len(visible)} of {len(frame)} rows. ",
                    "Use the complete export for all rows.",
                )
            )
        )
    float_columns = visible.select_dtypes(include=float).columns
    column_config = {
        str(column): st.column_config.NumberColumn(format=UI_NUMBER_FORMAT)
        for column in float_columns
    }
    _ = st.dataframe(
        visible,
        hide_index=True,
        width="stretch",
        height="content",
        column_config=column_config,
    )
