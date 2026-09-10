"""Active-tab workspace orchestration and predictive comparison."""

from dataclasses import dataclass
from typing import assert_never

import pandas as pd
import streamlit as st

from bspe.records import SequenceDataset
from bspe.ui.comparison import comparison_dataframe
from bspe.ui.text import joined_text
from bspe.ui.tokens import format_ui_decimal
from bspe.ui.workbench_state import WorkbenchForm
from bspe.ui.workspace_agreement import (
    AgreementSummary,
    agreement_summary,
)
from bspe.ui.workspace_charts import (
    agreement_figure,
    render_static_figure,
)
from bspe.ui.workspace_components import (
    COMPARISON_ROW_LIMIT,
    EVIDENCE_ROW_LIMIT,
    render_bounded_dataframe,
    render_metrics,
    render_scope_heading,
)
from bspe.ui.workspace_method_context import MethodRenderContext
from bspe.ui.workspace_method_renderers import render_method_result
from bspe.ui.workspace_metrics import MetricValue
from bspe.ui.workspace_selection import SequenceScope
from bspe.ui.workspace_session import WorkspaceSubview
from bspe.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class MethodWorkspaceContext:
    """Typed inputs for one active method projection."""

    result: WorkbenchResult
    form: WorkbenchForm
    dataset: SequenceDataset
    scope: SequenceScope
    subview: WorkspaceSubview


@dataclass(frozen=True, slots=True)
class ComparisonWorkspaceContext:
    """Typed inputs for a cross-method projection."""

    results: tuple[WorkbenchResult, ...]
    form: WorkbenchForm
    scope: SequenceScope
    subview: WorkspaceSubview


def render_comparison_workspace(context: ComparisonWorkspaceContext) -> None:
    """Render agreement separately from exact cross-method values."""
    if not context.results:
        _ = st.info(
            "Method comparison requires at least one valid current result."
        )
        return
    identifiers = tuple(
        str(record.sequence_id) for record in context.results[0].records
    )
    render_scope_heading(context.scope, identifiers)
    _ = st.caption(
        joined_text(
            (
                "Predictive uncertainty and observed-symbol entropy are not ",
                "interchangeable. ",
                "Observed Shannon Entropy is excluded from predictive agreement.",
            )
        )
    )
    summary = agreement_summary(
        context.results,
        context.form.intake.observable_labels,
    )
    subview = context.subview
    match subview:
        case WorkspaceSubview.OVERVIEW:
            _render_agreement_overview(summary, context.scope)
            return
        case WorkspaceSubview.COMPARE:
            render_bounded_dataframe(
                comparison_dataframe(
                    context.results,
                    context.form.intake.observable_labels,
                ),
                COMPARISON_ROW_LIMIT,
                "submitted record, then method",
            )
            return
        case WorkspaceSubview.EVIDENCE:
            frame = pd.DataFrame.from_records(
                ((row.sequence_id, row.status) for row in summary.rows),
                columns=("Sequence ID", "Predictive agreement status"),
            )
            if frame.empty:
                _ = st.info(
                    "Predictive agreement is unavailable without a predictive method."
                )
            else:
                render_bounded_dataframe(
                    frame,
                    EVIDENCE_ROW_LIMIT,
                    "submitted record",
                )
            _ = st.info(
                "Open a method tab for method-specific assumptions and evidence."
            )
            return
    assert_never(subview)


def render_method_workspace(context: MethodWorkspaceContext) -> None:
    """Render only the active method and selected sequence projection."""
    render_method_result(
        MethodRenderContext(
            context.result,
            context.form,
            context.dataset,
            context.scope,
            context.subview,
        )
    )


def _render_agreement_overview(
    summary: AgreementSummary,
    scope: SequenceScope,
) -> None:
    if scope is SequenceScope.ONE:
        status = summary.rows[0].status if summary.rows else "Unavailable"
        render_metrics((MetricValue("Predictive agreement", status),))
        return
    rate = (
        "Unavailable"
        if summary.agreement_rate is None
        else format_ui_decimal(summary.agreement_rate)
    )
    render_metrics(
        (
            MetricValue("Agreement rate", rate),
            MetricValue("Agreement records", str(summary.agreement_count)),
            MetricValue("Disagreement records", str(summary.disagreement_count)),
            MetricValue("Unavailable records", str(summary.unavailable_count)),
        )
    )
    if summary.rows:
        _ = st.caption(
            "Distribution of predictive agreement states across selected records."
        )
        render_static_figure(agreement_figure(summary), key="workspace-agreement")
