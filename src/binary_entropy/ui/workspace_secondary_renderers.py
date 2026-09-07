"""HMM and Shannon active-method workspace renderers."""

import pandas as pd
import streamlit as st

from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.ui.chart import entropy_figure
from binary_entropy.ui.results import prefix_dataframe
from binary_entropy.ui.shannon_results import (
    shannon_prefix_dataframe,
    shannon_record_dataframe,
)
from binary_entropy.ui.workspace_charts import render_static_figure
from binary_entropy.ui.workspace_components import (
    COMPARISON_ROW_LIMIT,
    EVIDENCE_ROW_LIMIT,
    render_bounded_dataframe,
    render_scope_heading,
)
from binary_entropy.ui.workspace_method_common import (
    render_aggregate_chart,
    render_overview_metrics,
)
from binary_entropy.ui.workspace_method_context import MethodRenderContext
from binary_entropy.ui.workspace_selection import SequenceScope
from binary_entropy.ui.workspace_session import WorkspaceSubview


def render_hmm(
    analysis: HMMBatchAnalysis,
    context: MethodRenderContext,
) -> None:
    """Render stored HMM metrics, trend, comparison, or prefix evidence."""
    form = context.form
    scope = context.scope
    identifiers = tuple(str(record.sequence_id) for record in analysis.records)
    labels = form.intake.observable_labels
    render_scope_heading(scope, identifiers)
    _ = st.caption(
        "Configured two-state HMM; each sequence is filtered independently."
    )
    if context.subview is WorkspaceSubview.EVIDENCE:
        model = form.hmm_model.to_model()
        frame = pd.concat(
            tuple(
                prefix_dataframe(record.analysis, model)
                for record in analysis.records
            ),
            ignore_index=True,
        )
        render_bounded_dataframe(
            frame,
            EVIDENCE_ROW_LIMIT,
            "record, then prefix depth",
        )
        return
    if context.subview is WorkspaceSubview.OVERVIEW:
        render_overview_metrics(analysis, labels, scope)
        if scope is SequenceScope.ONE:
            record = analysis.records[0]
            _ = st.caption("HMM predictive entropy by sequence depth, in bits.")
            render_static_figure(
                entropy_figure(record.analysis),
                key=f"workspace-hmm-{record.sequence_id}",
            )
        else:
            render_aggregate_chart(analysis, "workspace-hmm-aggregate")
        return
    frame = pd.DataFrame.from_records(
        (
            (
                str(record.sequence_id),
                record.analysis.rows[-1].depth,
                record.analysis.rows[-1].entropy_bits,
            )
            for record in analysis.records
        ),
        columns=("Sequence ID", "Sequence depth", "Predictive entropy (bits)"),
    )
    render_bounded_dataframe(frame, COMPARISON_ROW_LIMIT, "submitted record")


def render_shannon(
    analysis: ShannonBatchAnalysis,
    context: MethodRenderContext,
) -> None:
    """Render descriptive Shannon metrics, comparison, or prefix evidence."""
    scope = context.scope
    identifiers = tuple(str(record.sequence_id) for record in analysis.records)
    render_scope_heading(scope, identifiers)
    _ = st.caption("Observed Shannon entropy is descriptive, not a prediction.")
    if context.subview is WorkspaceSubview.EVIDENCE:
        frame = pd.concat(
            tuple(
                shannon_prefix_dataframe(record) for record in analysis.records
            ),
            ignore_index=True,
        )
        render_bounded_dataframe(
            frame,
            EVIDENCE_ROW_LIMIT,
            "record, then prefix depth",
        )
        return
    if context.subview is WorkspaceSubview.OVERVIEW:
        render_overview_metrics(analysis, analysis.observable_labels, scope)
        if scope is not SequenceScope.ONE:
            render_aggregate_chart(analysis, "workspace-shannon-aggregate")
        return
    render_bounded_dataframe(
        shannon_record_dataframe(analysis),
        COMPARISON_ROW_LIMIT,
        "submitted record",
    )
