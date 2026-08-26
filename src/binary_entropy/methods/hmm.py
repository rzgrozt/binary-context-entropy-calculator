"""Independent-record adapter for the legacy fixed HMM analysis."""

from dataclasses import dataclass, field
from typing import Literal

from binary_entropy.analysis import analyze_sequence, assess_target
from binary_entropy.domain import BinaryHMM, SequenceAnalysis, TargetAssessment
from binary_entropy.records import SequenceDataset, SequenceId


@dataclass(frozen=True, slots=True)
class HMMRecordAnalysis:
    """Legacy HMM analysis and optional external-target evaluation for one record."""

    sequence_id: SequenceId
    analysis: SequenceAnalysis
    target_assessment: TargetAssessment | None


@dataclass(frozen=True, slots=True)
class HMMBatchAnalysis:
    """Independent legacy HMM results with no pooled model fitting."""

    records: tuple[HMMRecordAnalysis, ...]
    method: Literal["hmm"] = field(default="hmm", init=False)


def analyze_hmm(dataset: SequenceDataset, model: BinaryHMM) -> HMMBatchAnalysis:
    """Call the unchanged legacy analyzer independently for every record."""
    results: list[HMMRecordAnalysis] = []
    for record in dataset.records:
        analysis = analyze_sequence(model, record.sequence)
        target_assessment = (
            assess_target(analysis.rows[-1].predictive, record.actual_target_index)
            if record.actual_target_index is not None
            else None
        )
        results.append(
            HMMRecordAnalysis(record.sequence_id, analysis, target_assessment)
        )
    return HMMBatchAnalysis(records=tuple(results))
