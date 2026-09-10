"""VMM and first-order Markov active-method workspace renderers."""

from typing import assert_never

import pandas as pd
import streamlit as st

from bspe.markov_types import MarkovBatchAnalysis
from bspe.methods.hmm import HMMBatchAnalysis
from bspe.methods.shannon import ShannonBatchAnalysis
from bspe.records import SequenceDataset
from bspe.ui.chart import markov_entropy_figure
from bspe.ui.markov_model_view import render_markov_model_evidence
from bspe.ui.markov_results import (
    markov_prefix_dataframe,
    markov_record_dataframe,
)
from bspe.ui.text import joined_text
from bspe.ui.tokens import format_ui_decimal
from bspe.ui.vmm_chart import (
    render_vmm_entropy_plot,
    vmm_entropy_chart_spec,
)
from bspe.ui.vmm_results import vmm_depth_dataframe, vmm_record_dataframe
from bspe.ui.workspace_charts import render_static_figure
from bspe.ui.workspace_components import (
    COMPARISON_ROW_LIMIT,
    EVIDENCE_ROW_LIMIT,
    render_bounded_dataframe,
    render_scope_heading,
)
from bspe.ui.workspace_method_common import (
    render_aggregate_chart,
    render_overview_metrics,
)
from bspe.ui.workspace_method_context import MethodRenderContext
from bspe.ui.workspace_secondary_renderers import (
    render_hmm,
    render_shannon,
)
from bspe.ui.workspace_selection import SequenceScope
from bspe.ui.workspace_session import WorkspaceSubview
from bspe.vmm_types import VMMAnalysis


def render_method_result(context: MethodRenderContext) -> None:
    """Render one active method from its immutable selected projection."""
    result = context.result
    match result:
        case VMMAnalysis() as analysis:
            return _render_vmm(analysis, context)
        case MarkovBatchAnalysis() as analysis:
            return _render_markov(analysis, context)
        case HMMBatchAnalysis() as analysis:
            return render_hmm(analysis, context)
        case ShannonBatchAnalysis() as analysis:
            return render_shannon(analysis, context)
    assert_never(result)


def _render_vmm(
    analysis: VMMAnalysis,
    context: MethodRenderContext,
) -> None:
    dataset = context.dataset
    scope = context.scope
    identifiers = tuple(str(record.sequence_id) for record in analysis.records)
    render_scope_heading(scope, identifiers)
    smoothing_name = type(analysis.config.smoothing).__name__.replace("Smoothing", "")
    _ = st.caption(
        joined_text(
            (
                f"Variable-order {analysis.result_scope.value.replace('_', ' ')}; ",
                "smoothing ",
                f"{smoothing_name}, ",
                f"alpha {format_ui_decimal(analysis.config.smoothing.alpha)}; ",
                f"minimum support {analysis.config.minimum_support}.",
            )
        )
    )
    subview = context.subview
    match subview:
        case WorkspaceSubview.OVERVIEW:
            render_overview_metrics(analysis, dataset.labels.observables, scope)
            if scope is SequenceScope.ONE:
                record = analysis.records[0]
                if vmm_entropy_chart_spec(record).requested_depths:
                    _ = st.caption(
                        joined_text(
                            (
                                "Predictive entropy by requested context depth; ",
                                "longer contexts are not assumed better.",
                            )
                        )
                    )
                    render_vmm_entropy_plot(
                        record,
                        key=f"workspace-vmm-{record.sequence_id}",
                    )
            else:
                render_aggregate_chart(analysis, "workspace-vmm-aggregate")
            return None
        case WorkspaceSubview.COMPARE:
            return _render_vmm_records(analysis, dataset)
        case WorkspaceSubview.EVIDENCE:
            frame = pd.concat(
                tuple(
                    vmm_depth_dataframe(analysis, dataset, record)
                    for record in analysis.records
                ),
                ignore_index=True,
            )
            render_bounded_dataframe(
                frame,
                EVIDENCE_ROW_LIMIT,
                "record, then requested depth",
            )
            return None
    assert_never(subview)


def _render_markov(
    analysis: MarkovBatchAnalysis,
    context: MethodRenderContext,
) -> None:
    scope = context.scope
    identifiers = tuple(str(record.sequence_id) for record in analysis.records)
    render_scope_heading(scope, identifiers)
    _ = st.caption(
        joined_text(
            (
                f"First-order {analysis.result_scope.value.replace('_', ' ')}; ",
                "estimator ",
                f"{analysis.model.estimation_method.value.replace('_', ' ')}, ",
                f"alpha {format_ui_decimal(analysis.model.smoothing_alpha)}.",
            )
        )
    )
    subview = context.subview
    match subview:
        case WorkspaceSubview.OVERVIEW:
            render_overview_metrics(
                analysis,
                analysis.model.observable_labels,
                scope,
            )
            if scope is SequenceScope.ONE:
                record = analysis.records[0]
                if any(row.predictive is not None for row in record.rows):
                    _ = st.caption("Predictive entropy by prefix depth, in bits.")
                    render_static_figure(
                        markov_entropy_figure(record),
                        key=f"workspace-markov-{record.sequence_id}",
                    )
            else:
                render_aggregate_chart(analysis, "workspace-markov-aggregate")
            return None
        case WorkspaceSubview.COMPARE:
            return _render_markov_records(analysis)
        case WorkspaceSubview.EVIDENCE:
            render_markov_model_evidence(analysis, scope)
            frame = pd.concat(
                tuple(
                    markov_prefix_dataframe(
                        record,
                        analysis.model.observable_labels,
                    )
                    for record in analysis.records
                ),
                ignore_index=True,
            )
            render_bounded_dataframe(
                frame,
                EVIDENCE_ROW_LIMIT,
                "record, then prefix depth",
            )
            return None
    assert_never(subview)


def _render_vmm_records(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> None:
    render_bounded_dataframe(
        vmm_record_dataframe(analysis, dataset),
        COMPARISON_ROW_LIMIT,
        "submitted record",
    )


def _render_markov_records(analysis: MarkovBatchAnalysis) -> None:
    render_bounded_dataframe(
        markov_record_dataframe(analysis),
        COMPARISON_ROW_LIMIT,
        "submitted record",
    )
