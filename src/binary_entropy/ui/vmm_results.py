"""Unrounded native dataframe adapters for variable-order Markov results."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from streamlit.elements.lib.column_types import ColumnConfig

from binary_entropy.information import surprisal
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT
from binary_entropy.ui.vmm_evidence import (
    vmm_context_reason,
    vmm_depth_selection,
    vmm_estimation_rule,
    vmm_support_status,
    vmm_target_classification_label,
)
from binary_entropy.vmm_types import (
    VMMAnalysis,
    VMMDepthAnalysis,
    VMMRecordAnalysis,
)

type VMMCell = str | int | float | None
type VMMRow = tuple[VMMCell, ...]
type _EvaluationCells = tuple[str, str | None, float | None, float | None]


@dataclass(frozen=True, slots=True)
class _DepthFrameContext:
    analysis: VMMAnalysis
    source: SequenceRecord
    labels: tuple[str, str]


def vmm_context_label(
    context: BinarySequence,
    labels: tuple[str, str],
) -> str:
    """Map an internal binary suffix to observable labels."""
    label = ", ".join(labels[index] for index in context)
    return label or "Order 0 (no suffix)"


def vmm_prediction_label(
    record: VMMRecordAnalysis,
    labels: tuple[str, str],
) -> str:
    """Distinguish an unavailable prediction from a probability tie."""
    if record.probability_a is None or record.probability_b is None:
        return "Unavailable"
    target = record.predicted_target_index
    return "Tie" if target is None else labels[target]


def vmm_final_column_config(
    labels: tuple[str, str],
) -> dict[str, ColumnConfig]:
    """Return three-decimal display settings for raw final-summary floats."""
    number_column = st.column_config.NumberColumn(format=UI_NUMBER_FORMAT)
    return {
        f"P(next {labels[0]})": number_column,
        f"P(next {labels[1]})": number_column,
        "Predictive Shannon entropy (bits)": number_column,
        f"Surprisal of {labels[0]} (bits)": number_column,
        f"Surprisal of {labels[1]} (bits)": number_column,
        "Actual-target probability": number_column,
        "Actual-target surprisal (bits)": number_column,
    }


def vmm_depth_column_config(
    labels: tuple[str, str],
) -> dict[str, ColumnConfig]:
    """Return three-decimal display settings for raw depth-evidence floats."""
    number_column = st.column_config.NumberColumn(format=UI_NUMBER_FORMAT)
    return {
        "Smoothing alpha": number_column,
        f"P(next {labels[0]})": number_column,
        f"P(next {labels[1]})": number_column,
        "Predictive entropy (bits)": number_column,
        "Target probability": number_column,
        "Target surprisal (bits)": number_column,
    }


def vmm_record_dataframe(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> pd.DataFrame:
    """Build one unrounded final-prediction row per independent record."""
    labels = dataset.labels.observables
    source_records = {record.sequence_id: record for record in dataset.records}
    rows = tuple(
        _record_row(record, source_records[record.sequence_id], labels)
        for record in analysis.records
    )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Sequence ID",
            "Effective predictive context depth",
            "Actual context used",
            "Support count",
            f"P(next {labels[0]})",
            f"P(next {labels[1]})",
            "Prediction",
            "Predictive Shannon entropy (bits)",
            f"Surprisal of {labels[0]} (bits)",
            f"Surprisal of {labels[1]} (bits)",
            "Actual target",
            "Actual-target probability",
            "Actual-target surprisal (bits)",
            "Target classification",
            "Evaluation status",
        ),
    )


def vmm_depth_dataframe(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
    record: VMMRecordAnalysis,
) -> pd.DataFrame:
    """Build every context-depth evidence row in ascending depth order."""
    labels = dataset.labels.observables
    source = next(
        source for source in dataset.records if source.sequence_id == record.sequence_id
    )
    context = _DepthFrameContext(analysis, source, labels)
    rows = tuple(
        _depth_row(context, record, row)
        for row in sorted(record.depth_rows, key=lambda candidate: candidate.depth)
    )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Dataset role",
            "Record ID",
            "Workflow",
            "Result scope",
            "Requested depth",
            "Actual depth",
            "Context",
            "Context occurrence count",
            f"Next {labels[0]} count",
            f"Next {labels[1]} count",
            "Support rule",
            "Support status",
            "Sparse status",
            "Estimation rule",
            "Smoothing alpha",
            "Suffix-backoff selection",
            "Suffix-backoff reason",
            f"P(next {labels[0]})",
            f"P(next {labels[1]})",
            "Predictive entropy (bits)",
            "Evaluation status",
            "Observed target",
            "Target probability",
            "Target surprisal (bits)",
        ),
    )


def _depth_row(
    context: _DepthFrameContext,
    record: VMMRecordAnalysis,
    row: VMMDepthAnalysis,
) -> VMMRow:
    support, sparse = vmm_support_status(row.status)
    evaluation_status, target, target_probability, target_surprisal = _evaluation_cells(
        context, row
    )
    return (
        "training",
        record.sequence_id,
        "variable_order_markov",
        context.analysis.result_scope.value,
        row.depth,
        record.effective_context_depth,
        vmm_context_label(row.matched_suffix, context.labels),
        row.support,
        row.count_a,
        row.count_b,
        f"minimum_support={context.analysis.config.minimum_support}",
        support,
        sparse,
        vmm_estimation_rule(context.analysis.config.smoothing),
        context.analysis.config.smoothing.alpha,
        vmm_depth_selection(record, row),
        vmm_context_reason(row.status, context.analysis.config.smoothing),
        row.probability_a,
        row.probability_b,
        row.predictive_entropy_bits,
        evaluation_status,
        target,
        target_probability,
        target_surprisal,
    )


def _evaluation_cells(
    context: _DepthFrameContext,
    row: VMMDepthAnalysis,
) -> _EvaluationCells:
    target_index = context.source.actual_target_index
    if target_index is None:
        return "Not supplied", None, None, None
    probability = (row.probability_a, row.probability_b)[target_index]
    if probability is None:
        return (
            "In-sample evaluation unavailable",
            context.labels[target_index],
            None,
            None,
        )
    return (
        "In-sample evaluation, not held out",
        context.labels[target_index],
        probability,
        surprisal(probability),
    )


def _record_row(
    record: VMMRecordAnalysis,
    source_record: SequenceRecord,
    labels: tuple[str, str],
) -> VMMRow:
    assessment = record.target_assessment
    actual_target = source_record.actual_target_index
    if actual_target is None:
        target_label = None
        classification = "Not supplied"
        evaluation_status = "Not supplied"
    elif assessment is None:
        target_label = labels[actual_target]
        classification = "Unavailable"
        evaluation_status = "In-sample evaluation unavailable"
    else:
        target_label = labels[actual_target]
        classification = vmm_target_classification_label(assessment)
        evaluation_status = "In-sample evaluation, not held out"
    return (
        record.sequence_id,
        (
            "Unavailable"
            if record.effective_context_depth is None
            else record.effective_context_depth
        ),
        (
            "Unavailable"
            if record.context_used is None
            else vmm_context_label(record.context_used, labels)
        ),
        "Unavailable" if record.support_count is None else record.support_count,
        record.probability_a,
        record.probability_b,
        vmm_prediction_label(record, labels),
        record.predictive_entropy_bits,
        record.surprisal_a_bits,
        record.surprisal_b_bits,
        target_label,
        None if assessment is None else assessment.probability,
        None if assessment is None else assessment.surprisal_bits,
        classification,
        evaluation_status,
    )
