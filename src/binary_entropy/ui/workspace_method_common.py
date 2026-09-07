"""Shared metric and aggregate-chart rendering for method workspaces."""

import streamlit as st

from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workspace_charts import (
    AGGREGATE_CHART_LIMIT,
    aggregate_entropy_figure,
    render_static_figure,
)
from binary_entropy.ui.workspace_components import render_metrics
from binary_entropy.ui.workspace_metrics import (
    aggregate_metric_values,
    entropy_points,
    single_metric_values,
)
from binary_entropy.ui.workspace_selection import SequenceScope
from binary_entropy.workbench import WorkbenchResult


def render_overview_metrics(
    result: WorkbenchResult,
    labels: tuple[str, str],
    scope: SequenceScope,
) -> None:
    """Render single-record or aggregate stored-value metrics by scope."""
    values = (
        single_metric_values(result, labels)
        if scope is SequenceScope.ONE
        else aggregate_metric_values(result, labels)
    )
    render_metrics(values)


def render_aggregate_chart(result: WorkbenchResult, key: str) -> None:
    """Render one capped final-entropy chart with explicit truncation."""
    points = entropy_points(result)
    if not points:
        _ = st.info("No available final entropy values for this scope.")
        return
    _ = st.caption("Stored final entropy by sequence in deterministic record order.")
    if len(points) > AGGREGATE_CHART_LIMIT:
        _ = st.caption(
            joined_text(
                (
                    f"Chart shows the first {AGGREGATE_CHART_LIMIT} of ",
                    f"{len(points)} available records.",
                )
            )
        )
    render_static_figure(aggregate_entropy_figure(result), key=key)
