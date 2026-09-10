"""Shared native components for bounded workspace projections."""

from collections.abc import Mapping, Sequence
from typing import Final, assert_never

import pandas as pd  # noqa: RUF100  # noqa: PANDAS_OK
import streamlit as st

from bspe.ui.text import joined_text
from bspe.ui.tokens import UI_NUMBER_FORMAT
from bspe.ui.workspace_metrics import MetricValue
from bspe.ui.workspace_selection import SequenceScope

COMPARISON_ROW_LIMIT: Final = 25
EVIDENCE_ROW_LIMIT: Final = 50

type DataframeCell = str | int | float | bool | None


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
            _ = column.metric(metric.label, metric.value, help=metric.help_text)


def render_bounded_dataframe(
    frame: pd.DataFrame,
    limit: int,
    order: str,
    *,
    column_help: Mapping[str, str] | None = None,
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
    help_map = column_help or {}
    float_columns = set(visible.select_dtypes(include=float).columns)
    column_config = {
        str(column): st.column_config.NumberColumn(
            format=UI_NUMBER_FORMAT if column in float_columns else None,
            help=help_map.get(str(column)),
        )
        for column in visible.columns
        if column in float_columns or str(column) in help_map
    }
    _ = st.dataframe(
        visible,
        hide_index=True,
        width="stretch",
        height="content",
        column_config=column_config,
    )


def render_bounded_records(
    records: Sequence[Mapping[str, DataframeCell]],
    limit: int,
    order: str,
    *,
    columns: tuple[str, ...] | None = None,
    column_help: Mapping[str, str] | None = None,
) -> None:
    """Render records through the existing bounded dataframe presentation."""
    render_bounded_dataframe(
        pd.DataFrame.from_records(records, columns=columns),
        limit,
        order,
        column_help=column_help,
    )
