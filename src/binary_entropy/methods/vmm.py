"""Binary variable-order Markov fitting and suffix analysis."""

from dataclasses import dataclass
from typing import Final

from binary_entropy.domain import (
    ObservableIndex,
    TargetAssessment,
    TargetClassification,
)
from binary_entropy.information import binary_entropy, surprisal
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord
from binary_entropy.vmm_types import (
    VMMAnalysis,
    VMMConfig,
    VMMContextCount,
    VMMDepthAnalysis,
    VMMDepthStatus,
    VMMModel,
    VMMRecordAnalysis,
    VMMResultScope,
    VMMSmoothing,
)

DEFAULT_VMM_CONFIG: Final = VMMConfig()


@dataclass(frozen=True, slots=True)
class _DepthEvidence:
    depth: int
    matched_suffix: BinarySequence
    support: int
    count_a: int
    count_b: int
    probability_a: float | None
    probability_b: float | None
    predictive_entropy_bits: float | None


def fit_vmm(dataset: SequenceDataset) -> VMMModel:
    """Fit continuation counts at every observed order without joining records."""
    return _fit_records(dataset.records)


def analyze_vmm(
    dataset: SequenceDataset,
    config: VMMConfig = DEFAULT_VMM_CONFIG,
) -> VMMAnalysis:
    """Analyze every record against one boundary-preserving pooled model."""
    return _analyze_vmm(dataset, config, VMMResultScope.POOLED)


def analyze_vmm_per_sequence(
    dataset: SequenceDataset,
    config: VMMConfig = DEFAULT_VMM_CONFIG,
) -> VMMAnalysis:
    """Analyze every record against counts fitted only from that record."""
    return _analyze_vmm(dataset, config, VMMResultScope.PER_SEQUENCE)


def _fit_records(records: tuple[SequenceRecord, ...]) -> VMMModel:
    mutable_counts: dict[BinarySequence, list[int]] = {}
    for record in records:
        sequence = record.sequence
        for depth in range(len(sequence)):
            for start in range(len(sequence) - depth):
                context = sequence[start : start + depth]
                outcome = sequence[start + depth]
                counts = mutable_counts.setdefault(context, [0, 0])
                counts[outcome] += 1
    context_counts = tuple(
        VMMContextCount(context, counts[0], counts[1])
        for context, counts in sorted(
            mutable_counts.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )
    return VMMModel(
        context_counts=context_counts,
        source_sequence_count=len(records),
    )


def _analyze_vmm(
    dataset: SequenceDataset,
    config: VMMConfig,
    result_scope: VMMResultScope,
) -> VMMAnalysis:
    match result_scope:
        case VMMResultScope.POOLED:
            pooled_model = fit_vmm(dataset)
            records = tuple(
                _analyze_record(record, pooled_model, config)
                for record in dataset.records
            )
        case VMMResultScope.PER_SEQUENCE:
            records = tuple(
                _analyze_record(record, _fit_records((record,)), config)
                for record in dataset.records
            )
    return VMMAnalysis(
        config=config,
        records=records,
        result_scope=result_scope,
    )


def _analyze_record(
    record: SequenceRecord,
    model: VMMModel,
    config: VMMConfig,
) -> VMMRecordAnalysis:
    context_index = {row.context: row for row in model.context_counts}
    evidence: list[_DepthEvidence] = []
    for depth in range(len(record.sequence) + 1):
        suffix = () if depth == 0 else record.sequence[-depth:]
        count_row = context_index.get(suffix)
        if count_row is None:
            count_a = 0
            count_b = 0
            support = 0
            probability_a = None
            probability_b = None
            entropy_bits = None
        else:
            count_a = count_row.count_a
            count_b = count_row.count_b
            support = count_row.support
            probability_a, probability_b = _smoothed_probabilities(
                count_row,
                config.smoothing,
            )
            entropy_bits = binary_entropy(probability_a)
        evidence.append(
            _DepthEvidence(
                depth=depth,
                matched_suffix=suffix,
                support=support,
                count_a=count_a,
                count_b=count_b,
                probability_a=probability_a,
                probability_b=probability_b,
                predictive_entropy_bits=entropy_bits,
            )
        )
    selected_depth = max(
        (row.depth for row in evidence if row.support >= config.minimum_support),
        default=None,
    )
    depth_rows = tuple(_depth_result(row, config.minimum_support) for row in evidence)
    selected = next(
        (row for row in depth_rows if row.depth == selected_depth),
        None,
    )
    probability_a = selected.probability_a if selected is not None else None
    probability_b = selected.probability_b if selected is not None else None
    predicted_index = _predicted_index(probability_a, probability_b)
    target_assessment = (
        _assess_target(record.actual_target_index, probability_a, probability_b)
        if record.actual_target_index is not None
        and probability_a is not None
        and probability_b is not None
        else None
    )
    return VMMRecordAnalysis(
        sequence_id=record.sequence_id,
        sequence=record.sequence,
        model=model,
        effective_context_depth=selected.depth if selected is not None else None,
        context_used=selected.matched_suffix if selected is not None else None,
        support_count=selected.support if selected is not None else None,
        probability_a=probability_a,
        probability_b=probability_b,
        predicted_target_index=predicted_index,
        predictive_entropy_bits=(
            selected.predictive_entropy_bits if selected is not None else None
        ),
        surprisal_a_bits=(
            surprisal(probability_a) if probability_a is not None else None
        ),
        surprisal_b_bits=(
            surprisal(probability_b) if probability_b is not None else None
        ),
        depth_rows=depth_rows,
        target_assessment=target_assessment,
    )


def _smoothed_probabilities(
    counts: VMMContextCount,
    smoothing: VMMSmoothing,
) -> tuple[float, float]:
    alpha = smoothing.alpha
    denominator = counts.support + 2.0 * alpha
    return (
        (counts.count_a + alpha) / denominator,
        (counts.count_b + alpha) / denominator,
    )


def _depth_result(
    evidence: _DepthEvidence,
    minimum_support: int,
) -> VMMDepthAnalysis:
    if evidence.support == 0:
        status = VMMDepthStatus.UNAVAILABLE
    elif evidence.support >= minimum_support:
        status = VMMDepthStatus.ACCEPTED
    else:
        status = VMMDepthStatus.LOW_SUPPORT
    return VMMDepthAnalysis(
        depth=evidence.depth,
        matched_suffix=evidence.matched_suffix,
        support=evidence.support,
        count_a=evidence.count_a,
        count_b=evidence.count_b,
        probability_a=evidence.probability_a,
        probability_b=evidence.probability_b,
        predictive_entropy_bits=evidence.predictive_entropy_bits,
        status=status,
    )


def _predicted_index(
    probability_a: float | None,
    probability_b: float | None,
) -> ObservableIndex | None:
    if probability_a is None or probability_b is None or probability_a == probability_b:
        return None
    return 0 if probability_a > probability_b else 1


def _assess_target(
    actual_target_index: ObservableIndex,
    probability_a: float,
    probability_b: float,
) -> TargetAssessment:
    if probability_a == probability_b:
        classification = TargetClassification.TIED
    else:
        modal_index: ObservableIndex = 0 if probability_a > probability_b else 1
        classification = (
            TargetClassification.MODAL
            if actual_target_index == modal_index
            else TargetClassification.LOWER_PROBABILITY
        )
    probability = probability_a if actual_target_index == 0 else probability_b
    return TargetAssessment(
        actual_target_index=actual_target_index,
        probability=probability,
        surprisal_bits=surprisal(probability),
        classification=classification,
    )
