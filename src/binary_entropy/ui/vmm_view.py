"""Compact native Streamlit presentation for variable-order Markov results."""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import format_ui_decimal
from binary_entropy.ui.vmm_artifacts import render_vmm_artifacts
from binary_entropy.ui.vmm_chart import (
    render_vmm_entropy_plot,
    vmm_entropy_chart_spec,
)
from binary_entropy.ui.vmm_evidence import vmm_target_classification_label
from binary_entropy.ui.vmm_results import (
    vmm_context_label,
    vmm_depth_column_config,
    vmm_depth_dataframe,
    vmm_final_column_config,
    vmm_prediction_label,
    vmm_record_dataframe,
)
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMAnalysis,
    VMMDepthStatus,
    VMMRecordAnalysis,
    VMMResultScope,
    VMMSmoothing,
)

_MLE_UNAVAILABLE_TEXT: Final = (
    "MLE unavailable: unseen context has no occurrences in the training dataset."
)
_SUPPORT_UNAVAILABLE_TEXT: Final = (
    "Prediction unavailable: no context meets the configured minimum support."
)


@dataclass(frozen=True, slots=True)
class _VMMViewContext:
    analysis: VMMAnalysis
    dataset: SequenceDataset
    labels: tuple[str, str]
    smoothing: VMMSmoothing


def render_vmm_result(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> None:
    """Render all VMM final quantities and context-depth evidence."""
    labels = dataset.labels.observables
    _ = st.subheader("Variable-order Markov")
    _ = st.caption(
        joined_text(
            (
                "The model detects and predicts recurrent finite-context statistical ",
                "dependencies in binary sequences.",
            )
        )
    )
    _ = st.markdown(_configuration_summary(analysis))
    _ = st.caption(
        joined_text(
            (
                "Unsupported suffixes back off to shorter supported contexts. ",
                "Every submitted record remains independent.",
            )
        )
    )
    _ = st.markdown("Per-record final predictions in deterministic input order.")
    _ = st.dataframe(
        vmm_record_dataframe(analysis, dataset),
        hide_index=True,
        width="stretch",
        height="content",
        column_config=vmm_final_column_config(labels),
    )

    source_records = {record.sequence_id: record for record in dataset.records}
    context = _VMMViewContext(
        analysis,
        dataset,
        labels,
        analysis.config.smoothing,
    )
    for record_index, record in enumerate(analysis.records):
        source_record = source_records[record.sequence_id]
        chart_key = f"vmm-entropy-chart-{record_index}-{record.sequence_id}"
        if len(analysis.records) == 1:
            _render_record(record, source_record, context, chart_key=chart_key)
        else:
            with st.expander(
                f"VMM sequence: {record.sequence_id}",
                expanded=True,
            ):
                _render_record(record, source_record, context, chart_key=chart_key)
    render_vmm_artifacts(analysis, dataset)


def _configuration_summary(analysis: VMMAnalysis) -> str:
    match analysis.config.smoothing:
        case MLESmoothing(alpha=alpha):
            estimation = "Maximum likelihood (MLE)"
        case KTSmoothing(alpha=alpha):
            estimation = "Krichevsky-Trofimov smoothing"
        case AdditiveSmoothing(alpha=alpha):
            estimation = "Custom additive smoothing"
    match analysis.result_scope:
        case VMMResultScope.POOLED:
            scope = "Pooled model"
        case VMMResultScope.PER_SEQUENCE:
            scope = "Per-sequence analysis"
    return joined_text(
        (
            f"**{scope}.** {estimation} with alpha ",
            f"{format_ui_decimal(alpha)}; minimum context support ",
            f"{analysis.config.minimum_support}.",
        )
    )


def _render_record(
    record: VMMRecordAnalysis,
    source_record: SequenceRecord,
    context: _VMMViewContext,
    *,
    chart_key: str,
) -> None:
    labels = context.labels
    _ = st.text(f"Sequence ID: {record.sequence_id}")
    metrics = (
        (
            "Effective predictive context depth",
            _integer_display(record.effective_context_depth),
        ),
        (
            "Actual context used",
            (
                "Unavailable"
                if record.context_used is None
                else vmm_context_label(record.context_used, labels)
            ),
        ),
        ("Support count", _integer_display(record.support_count)),
        (f"P(next {labels[0]})", _decimal_display(record.probability_a)),
        (f"P(next {labels[1]})", _decimal_display(record.probability_b)),
        ("Predicted target", vmm_prediction_label(record, labels)),
        (
            "Predictive Shannon entropy (bits)",
            _decimal_display(record.predictive_entropy_bits),
        ),
        (
            f"Surprisal of {labels[0]} (bits)",
            _decimal_display(record.surprisal_a_bits),
        ),
        (
            f"Surprisal of {labels[1]} (bits)",
            _decimal_display(record.surprisal_b_bits),
        ),
    )
    columns = st.columns(3)
    for index, (label, value) in enumerate(metrics):
        _ = columns[index % 3].metric(label, value)

    if record.probability_a is None or record.probability_b is None:
        match context.smoothing:
            case MLESmoothing():
                match record.depth_rows[0].status:
                    case VMMDepthStatus.UNAVAILABLE:
                        unavailable_text = _MLE_UNAVAILABLE_TEXT
                    case VMMDepthStatus.ACCEPTED | VMMDepthStatus.LOW_SUPPORT:
                        unavailable_text = _SUPPORT_UNAVAILABLE_TEXT
            case KTSmoothing() | AdditiveSmoothing():
                unavailable_text = _SUPPORT_UNAVAILABLE_TEXT
        _ = st.info(unavailable_text)
    if source_record.actual_target_index is not None:
        _render_target(record, source_record, labels)

    _render_entropy_chart(record, chart_key=chart_key)
    _ = st.markdown("**Exact context-depth evidence**")
    _ = st.caption(
        "Exact fallback values follow deterministic ascending requested depth order."
    )
    _ = st.dataframe(
        vmm_depth_dataframe(context.analysis, context.dataset, record),
        hide_index=True,
        width="stretch",
        height="content",
        column_config=vmm_depth_column_config(labels),
    )


def _render_entropy_chart(
    record: VMMRecordAnalysis,
    *,
    chart_key: str,
) -> None:
    spec = vmm_entropy_chart_spec(record)
    if not spec.requested_depths:
        _ = st.info(
            joined_text(
                (
                    "Context-depth entropy chart unavailable: no predictive entropy ",
                    "values are defined. Exact unavailable evidence remains in the ",
                    "table.",
                )
            )
        )
        return
    _ = st.markdown("**Predictive entropy by requested context depth**")
    _ = st.caption(
        joined_text(
            (
                "Available predictive entropy values only; longer contexts are not ",
                "assumed better or monotonic. Exact values remain in the table below.",
            )
        )
    )
    render_vmm_entropy_plot(record, key=chart_key)


def _render_target(
    record: VMMRecordAnalysis,
    source_record: SequenceRecord,
    labels: tuple[str, str],
) -> None:
    target_index = source_record.actual_target_index
    if target_index is None:
        return
    _ = st.caption(
        joined_text(
            (
                "Evaluation-only: In-sample evaluation, not held out. The target ",
                "did not affect fitting, context selection, backoff, or prediction.",
            )
        )
    )
    assessment = record.target_assessment
    columns = st.columns(4)
    _ = columns[0].metric("Actual target", labels[target_index])
    _ = columns[1].metric(
        "Actual-target probability",
        "Unavailable"
        if assessment is None
        else format_ui_decimal(assessment.probability),
    )
    _ = columns[2].metric(
        "Actual-target surprisal (bits)",
        "Unavailable"
        if assessment is None
        else format_ui_decimal(assessment.surprisal_bits),
    )
    _ = columns[3].metric(
        "Target assessment",
        "Unavailable"
        if assessment is None
        else vmm_target_classification_label(assessment),
    )


def _integer_display(value: int | None) -> str:
    return "Unavailable" if value is None else str(value)


def _decimal_display(value: float | None) -> str:
    return "Unavailable" if value is None else format_ui_decimal(value)
