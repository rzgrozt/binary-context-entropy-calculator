"""Deterministic experimental raw exports for variable-order Markov results."""

import json
from dataclasses import dataclass
from typing import Literal, TypedDict

from binary_entropy._vmm_serialization_common import (
    CONFIGURED_DEPTH_SELECTION,
    DATASET_ROLE,
    EXPERIMENTAL_NOTICE,
    EXPERIMENTAL_STATUS,
    ExportContext,
    JsonMetric,
    RecordPair,
    backoff_reason,
    backoff_selection,
    estimation_rule,
    evaluation_status,
    export_context,
    json_metric,
    labeled_sequence,
    record_pairs,
    selected_row,
    support_status,
)
from binary_entropy._vmm_serialization_csv import (
    context_evidence_csv,
    evaluation_csv,
)
from binary_entropy.information import binary_entropy
from binary_entropy.records import SequenceDataset
from binary_entropy.vmm_types import (
    VMMAnalysis,
    VMMDepthStatus,
    VMMModel,
    VMMResultScope,
)


class _ContextDistributionJson(TypedDict):
    displayed_context: list[str]
    context_depth: int
    context_occurrence_count: int
    next_a_count: int
    next_b_count: int
    support_status: str
    sparse_status: str
    support_rule: str
    next_a_probability: float
    next_b_probability: float
    predictive_entropy_bits: float


class _ModelJson(TypedDict):
    model_identifier: str
    source_record_identifiers: list[str]
    source_orders: list[int]
    source_sequence_count: int
    fitted_context_distributions: list[_ContextDistributionJson]


class _RecordJson(TypedDict):
    record_identifier: str
    source_order: int
    sequence_stimulus: list[str]
    consumed_prefix_stimulus: list[str]
    sequence_length: int
    consumed_prefix_depth: int
    displayed_context: list[str] | None
    requested_depth: int
    actual_depth: int | None
    suffix_backoff_selection: str
    suffix_backoff_reason: str | None
    context_occurrence_count: int | None
    next_a_count: int | None
    next_b_count: int | None
    support_status: str
    sparse_status: str
    next_a_probability: float | None
    next_b_probability: float | None
    predictive_entropy_bits: float | None
    observed_target: str | None
    target_probability: float | None
    target_surprisal_bits: JsonMetric
    target_classification: str | None
    evaluation_status: str


class _ContextModelJson(TypedDict):
    schema_version: Literal[1]
    artifact_name: Literal["Context model export"]
    experimental_status: Literal["experimental"]
    experimental_notice: str
    method: Literal["vmm"]
    dataset_role: Literal["training"]
    training_dataset_identifier: str
    evaluation_dataset_identifier: None
    observable_labels: list[str]
    workflow: Literal["variable_order_markov"]
    result_scope: VMMResultScope
    configured_depth_selection: str
    minimum_support: int
    estimation_rule: str
    smoothing_alpha: float
    support_rule: str
    records: list[_RecordJson]
    models: list[_ModelJson]


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    """One shared or record-specific fitted model and its ordered sources."""

    identifier: str
    sources: tuple[RecordPair, ...]
    model: VMMModel


def vmm_context_model_json(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> bytes:
    """Serialize configured selection, stimuli, and every fitted distribution."""
    context = export_context(analysis, dataset)
    pairs = record_pairs(context)
    payload: _ContextModelJson = {
        "schema_version": 1,
        "artifact_name": "Context model export",
        "experimental_status": EXPERIMENTAL_STATUS,
        "experimental_notice": EXPERIMENTAL_NOTICE,
        "method": "vmm",
        "dataset_role": DATASET_ROLE,
        "training_dataset_identifier": context.training_identifier,
        "evaluation_dataset_identifier": None,
        "observable_labels": list(dataset.labels.observables),
        "workflow": "variable_order_markov",
        "result_scope": analysis.result_scope,
        "configured_depth_selection": CONFIGURED_DEPTH_SELECTION,
        "minimum_support": analysis.config.minimum_support,
        "estimation_rule": estimation_rule(analysis),
        "smoothing_alpha": analysis.config.smoothing.alpha,
        "support_rule": f"minimum_support={analysis.config.minimum_support}",
        "records": [_record_json(context, pair) for pair in pairs],
        "models": [_model_json(context, entry) for entry in _model_entries(context)],
    }
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    return (text + "\n").encode("utf-8", errors="strict")


def vmm_context_evidence_csv(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> str:
    """Serialize every examined suffix in record and requested-depth order."""
    return context_evidence_csv(export_context(analysis, dataset))


def vmm_evaluation_csv(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> str:
    """Serialize final predictions and truthfully labeled in-sample targets."""
    return evaluation_csv(export_context(analysis, dataset))


def _model_entries(context: ExportContext) -> tuple[_ModelEntry, ...]:
    pairs = record_pairs(context)
    entries = {
        VMMResultScope.POOLED: (_ModelEntry("pooled", pairs, pairs[0].analysis.model),),
        VMMResultScope.PER_SEQUENCE: tuple(
            _ModelEntry(str(pair.source.sequence_id), (pair,), pair.analysis.model)
            for pair in pairs
        ),
    }
    return entries[context.analysis.result_scope]


def _model_json(context: ExportContext, entry: _ModelEntry) -> _ModelJson:
    labels = context.dataset.labels.observables
    alpha = context.analysis.config.smoothing.alpha
    minimum_support = context.analysis.config.minimum_support
    distributions: list[_ContextDistributionJson] = []
    for counts in entry.model.context_counts:
        probability_a = (counts.count_a + alpha) / (counts.support + 2.0 * alpha)
        probability_b = (counts.count_b + alpha) / (counts.support + 2.0 * alpha)
        status = (
            VMMDepthStatus.ACCEPTED
            if counts.support >= minimum_support
            else VMMDepthStatus.LOW_SUPPORT
        )
        accepted, sparse = support_status(status)
        distributions.append(
            {
                "displayed_context": list(labeled_sequence(counts.context, labels)),
                "context_depth": len(counts.context),
                "context_occurrence_count": counts.support,
                "next_a_count": counts.count_a,
                "next_b_count": counts.count_b,
                "support_status": accepted,
                "sparse_status": sparse,
                "support_rule": f"minimum_support={minimum_support}",
                "next_a_probability": probability_a,
                "next_b_probability": probability_b,
                "predictive_entropy_bits": binary_entropy(probability_a),
            }
        )
    return {
        "model_identifier": entry.identifier,
        "source_record_identifiers": [
            str(pair.source.sequence_id) for pair in entry.sources
        ],
        "source_orders": [pair.source_order for pair in entry.sources],
        "source_sequence_count": entry.model.source_sequence_count,
        "fitted_context_distributions": distributions,
    }


def _record_json(context: ExportContext, pair: RecordPair) -> _RecordJson:
    labels = context.dataset.labels.observables
    record = pair.analysis
    selected = selected_row(record)
    support, sparse = (
        support_status(selected.status)
        if selected is not None
        else ("unavailable", "unavailable")
    )
    target = record.target_assessment
    target_index = pair.source.actual_target_index
    stimulus = list(labeled_sequence(record.sequence, labels))
    return {
        "record_identifier": str(record.sequence_id),
        "source_order": pair.source_order,
        "sequence_stimulus": stimulus,
        "consumed_prefix_stimulus": stimulus,
        "sequence_length": len(record.sequence),
        "consumed_prefix_depth": len(record.sequence),
        "displayed_context": (
            None
            if record.context_used is None
            else list(labeled_sequence(record.context_used, labels))
        ),
        "requested_depth": len(record.sequence),
        "actual_depth": record.effective_context_depth,
        "suffix_backoff_selection": backoff_selection(record),
        "suffix_backoff_reason": backoff_reason(
            record,
            context.analysis.config.smoothing,
        ),
        "context_occurrence_count": None if selected is None else selected.support,
        "next_a_count": None if selected is None else selected.count_a,
        "next_b_count": None if selected is None else selected.count_b,
        "support_status": support,
        "sparse_status": sparse,
        "next_a_probability": record.probability_a,
        "next_b_probability": record.probability_b,
        "predictive_entropy_bits": record.predictive_entropy_bits,
        "observed_target": None if target_index is None else labels[target_index],
        "target_probability": None if target is None else target.probability,
        "target_surprisal_bits": json_metric(
            None if target is None else target.surprisal_bits
        ),
        "target_classification": (
            None if target is None else target.classification.value
        ),
        "evaluation_status": evaluation_status(pair),
    }
