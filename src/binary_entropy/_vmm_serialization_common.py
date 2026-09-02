"""Shared provenance and status values for VMM raw exports."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final

from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMAnalysis,
    VMMDepthAnalysis,
    VMMDepthStatus,
    VMMRecordAnalysis,
    VMMSmoothing,
)

EXPERIMENTAL_STATUS: Final = "experimental"
EXPERIMENTAL_NOTICE: Final = (
    "Experimental raw VMM artifact; retain its schema and source details when reused."
)
MLE_UNAVAILABLE_REASON: Final = (
    "MLE unavailable: unseen context has no occurrences in the training dataset."
)
CONFIGURED_DEPTH_SELECTION: Final = "deepest_supported_suffix"
DATASET_ROLE: Final = "training"
type JsonMetric = float | str | None
_ESTIMATION_RULES: Final = {
    KTSmoothing: "krichevsky_trofimov",
    MLESmoothing: "maximum_likelihood",
    AdditiveSmoothing: "additive_smoothing",
}
_SUPPORT_STATUSES: Final = {
    VMMDepthStatus.ACCEPTED: ("accepted", "not_sparse"),
    VMMDepthStatus.LOW_SUPPORT: ("low_support", "sparse"),
    VMMDepthStatus.UNAVAILABLE: ("unavailable", "unavailable"),
}
_UNAVAILABLE_REASONS: Final = {
    KTSmoothing: "Unseen context has no occurrences in the training dataset.",
    MLESmoothing: MLE_UNAVAILABLE_REASON,
    AdditiveSmoothing: "Unseen context has no occurrences in the training dataset.",
}


@dataclass(frozen=True, slots=True)
class ExportContext:
    """Inputs and stable identity shared by all three artifacts."""

    analysis: VMMAnalysis
    dataset: SequenceDataset
    training_identifier: str


@dataclass(frozen=True, slots=True)
class RecordPair:
    """One analyzed record paired with its ordered source stimulus."""

    analysis: VMMRecordAnalysis
    source: SequenceRecord
    source_order: int


def export_context(analysis: VMMAnalysis, dataset: SequenceDataset) -> ExportContext:
    """Build shared export provenance from immutable analysis inputs."""
    identity = (
        ("observable_labels", dataset.labels.observables),
        (
            "records",
            tuple((record.sequence_id, record.sequence) for record in dataset.records),
        ),
    )
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")
    digest = hashlib.sha256(canonical).hexdigest()
    return ExportContext(analysis, dataset, f"sha256:{digest}")


def record_pairs(context: ExportContext) -> tuple[RecordPair, ...]:
    """Pair analysis records to source records without changing source order."""
    return tuple(
        RecordPair(analyzed, source, source_order)
        for source_order, (analyzed, source) in enumerate(
            zip(context.analysis.records, context.dataset.records, strict=True),
            start=1,
        )
    )


def estimation_rule(analysis: VMMAnalysis) -> str:
    """Name the configured estimator without inferring from alpha alone."""
    return _ESTIMATION_RULES[type(analysis.config.smoothing)]


def labeled_sequence(
    sequence: BinarySequence,
    labels: tuple[str, str],
) -> tuple[str, ...]:
    """Map internal observable indices to retained source labels."""
    return tuple(labels[index] for index in sequence)


def csv_data(text: str) -> str:
    """Neutralize spreadsheet formula prefixes in untrusted text cells."""
    return f"'{text}" if text.lstrip().startswith(("=", "+", "-", "@")) else text


def support_status(status: VMMDepthStatus) -> tuple[str, str]:
    """Return separate support and sparse statuses for one examined context."""
    return _SUPPORT_STATUSES[status]


def context_reason(row: VMMDepthAnalysis, smoothing: VMMSmoothing) -> str | None:
    """State why an examined context was rejected when a reason applies."""
    reasons = {
        VMMDepthStatus.ACCEPTED: None,
        VMMDepthStatus.LOW_SUPPORT: (
            "Context support is below the configured minimum."
        ),
        VMMDepthStatus.UNAVAILABLE: _UNAVAILABLE_REASONS[type(smoothing)],
    }
    return reasons[row.status]


def backoff_selection(record: VMMRecordAnalysis) -> str:
    """Name whether the requested full suffix was selected or backed off."""
    if record.effective_context_depth is None:
        return "unavailable"
    if record.effective_context_depth == len(record.sequence):
        return "requested_depth_selected"
    return "backed_off_to_shorter_suffix"


def backoff_reason(
    record: VMMRecordAnalysis,
    smoothing: VMMSmoothing,
) -> str | None:
    """Explain the deepest requested suffix outcome from retained evidence."""
    if record.effective_context_depth == len(record.sequence):
        return None
    reason = context_reason(record.depth_rows[-1], smoothing)
    return reason or "No context meets the configured minimum support."


def selected_row(record: VMMRecordAnalysis) -> VMMDepthAnalysis | None:
    """Return the retained evidence row selected for final prediction."""
    return next(
        (
            row
            for row in record.depth_rows
            if row.depth == record.effective_context_depth
        ),
        None,
    )


def evaluation_status(pair: RecordPair) -> str:
    """Label training-record target assessment without claiming held-out data."""
    if pair.source.actual_target_index is None:
        return "Not supplied"
    if pair.analysis.target_assessment is None:
        return "In-sample evaluation unavailable"
    return "In-sample evaluation, not held out"


def json_metric(value: float | None) -> JsonMetric:
    """Represent defined infinite surprisal without non-standard JSON numbers."""
    if value is None or math.isfinite(value):
        return value
    return "infinity" if value > 0.0 else "-infinity"
