"""High-precision CSV rows for experimental VMM raw artifacts."""

from typing import Final

from binary_entropy._vmm_serialization_common import (
    CONFIGURED_DEPTH_SELECTION,
    DATASET_ROLE,
    EXPERIMENTAL_NOTICE,
    EXPERIMENTAL_STATUS,
    ExportContext,
    RecordPair,
    backoff_reason,
    backoff_selection,
    context_reason,
    csv_data,
    estimation_rule,
    evaluation_status,
    labeled_sequence,
    record_pairs,
    selected_row,
    support_status,
)
from binary_entropy.information import surprisal
from binary_entropy.markov_csv import CsvCell, markov_csv_text
from binary_entropy.records import BinarySequence
from binary_entropy.vmm_types import VMMDepthAnalysis

COLUMNS: Final = (
    "artifact_name",
    "experimental_status",
    "experimental_notice",
    "method",
    "dataset_role",
    "training_dataset_identifier",
    "evaluation_dataset_identifier",
    "record_identifier",
    "source_order",
    "observable_A_label",
    "observable_B_label",
    "sequence_stimulus",
    "consumed_prefix_stimulus",
    "sequence_length",
    "consumed_prefix_depth",
    "displayed_context",
    "requested_depth",
    "actual_depth",
    "workflow",
    "result_scope",
    "configured_depth_selection",
    "estimation_rule",
    "smoothing_alpha",
    "suffix_backoff_selection",
    "suffix_backoff_reason",
    "context_occurrence_count",
    "next_A_count",
    "next_B_count",
    "support_rule",
    "support_status",
    "sparse_status",
    "next_A_probability",
    "next_B_probability",
    "predictive_entropy_bits",
    "observed_target",
    "target_probability",
    "target_surprisal_bits",
    "target_classification",
    "evaluation_status",
)


def context_evidence_csv(context: ExportContext) -> str:
    """Serialize every retained depth row in source and ascending-depth order."""
    rows = tuple(
        _evidence_row(context, pair, row)
        for pair in record_pairs(context)
        for row in pair.analysis.depth_rows
    )
    return markov_csv_text(COLUMNS, rows)


def evaluation_csv(context: ExportContext) -> str:
    """Serialize one final evaluation row per ordered training record."""
    rows = tuple(_evaluation_row(context, pair) for pair in record_pairs(context))
    return markov_csv_text(COLUMNS, rows)


def _evidence_row(
    context: ExportContext,
    pair: RecordPair,
    row: VMMDepthAnalysis,
) -> tuple[CsvCell, ...]:
    record = pair.analysis
    labels = context.dataset.labels.observables
    selected = row.depth == record.effective_context_depth
    support, sparse = support_status(row.status)
    target = record.target_assessment if selected else None
    target_index = pair.source.actual_target_index
    target_probability = (
        None
        if target_index is None
        else (row.probability_a, row.probability_b)[target_index]
    )
    if target_index is None:
        row_evaluation_status = "Not supplied"
    elif target_probability is None:
        row_evaluation_status = "In-sample evaluation unavailable"
    else:
        row_evaluation_status = "In-sample evaluation, not held out"
    if record.effective_context_depth is None:
        selection = "no_context_selected"
    elif selected:
        selection = "selected"
    elif row.depth > record.effective_context_depth:
        selection = "rejected_for_backoff"
    else:
        selection = "not_selected_shorter_context"
    return (
        *_provenance(context, pair, "Context evidence export"),
        _context_text(row.matched_suffix, labels),
        row.depth,
        record.effective_context_depth,
        "variable_order_markov",
        context.analysis.result_scope.value,
        CONFIGURED_DEPTH_SELECTION,
        estimation_rule(context.analysis),
        context.analysis.config.smoothing.alpha,
        selection,
        context_reason(row, context.analysis.config.smoothing),
        row.support,
        row.count_a,
        row.count_b,
        f"minimum_support={context.analysis.config.minimum_support}",
        support,
        sparse,
        row.probability_a,
        row.probability_b,
        row.predictive_entropy_bits,
        None if target_index is None else csv_data(labels[target_index]),
        target_probability,
        None if target_probability is None else surprisal(target_probability),
        None if target is None else target.classification.value,
        row_evaluation_status,
    )


def _evaluation_row(
    context: ExportContext,
    pair: RecordPair,
) -> tuple[CsvCell, ...]:
    record = pair.analysis
    labels = context.dataset.labels.observables
    selected = selected_row(record)
    support, sparse = (
        support_status(selected.status)
        if selected is not None
        else ("unavailable", "unavailable")
    )
    target = record.target_assessment
    target_index = pair.source.actual_target_index
    return (
        *_provenance(context, pair, "Evaluation export"),
        (
            None
            if record.context_used is None
            else _context_text(record.context_used, labels)
        ),
        len(record.sequence),
        record.effective_context_depth,
        "variable_order_markov",
        context.analysis.result_scope.value,
        CONFIGURED_DEPTH_SELECTION,
        estimation_rule(context.analysis),
        context.analysis.config.smoothing.alpha,
        backoff_selection(record),
        backoff_reason(record, context.analysis.config.smoothing),
        None if selected is None else selected.support,
        None if selected is None else selected.count_a,
        None if selected is None else selected.count_b,
        f"minimum_support={context.analysis.config.minimum_support}",
        support,
        sparse,
        record.probability_a,
        record.probability_b,
        record.predictive_entropy_bits,
        None if target_index is None else csv_data(labels[target_index]),
        None if target is None else target.probability,
        None if target is None else target.surprisal_bits,
        None if target is None else target.classification.value,
        evaluation_status(pair),
    )


def _provenance(
    context: ExportContext,
    pair: RecordPair,
    artifact_name: str,
) -> tuple[CsvCell, ...]:
    labels = context.dataset.labels.observables
    stimulus = csv_data(",".join(labeled_sequence(pair.analysis.sequence, labels)))
    return (
        artifact_name,
        EXPERIMENTAL_STATUS,
        EXPERIMENTAL_NOTICE,
        "vmm",
        DATASET_ROLE,
        context.training_identifier,
        None,
        csv_data(str(pair.source.sequence_id)),
        pair.source_order,
        csv_data(labels[0]),
        csv_data(labels[1]),
        stimulus,
        stimulus,
        len(pair.analysis.sequence),
        len(pair.analysis.sequence),
    )


def _context_text(context: BinarySequence, labels: tuple[str, str]) -> str:
    if not context:
        return "Order 0 (no suffix)"
    return csv_data(",".join(labeled_sequence(context, labels)))
