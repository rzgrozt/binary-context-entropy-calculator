"""Predictive agreement projections from immutable result values."""

from dataclasses import dataclass
from typing import assert_never

from binary_entropy.domain import FloatArray, float_values
from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.vmm_types import VMMAnalysis
from binary_entropy.workbench import WorkbenchResult

_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PredictionValue:
    """One stored predictive outcome used for aggregate presentation."""

    sequence_id: str
    prediction: str | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class AgreementRow:
    """Predictive agreement status for one selected record."""

    sequence_id: str
    status: str


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    """Counts across predictive methods only."""

    rows: tuple[AgreementRow, ...]
    agreement_count: int
    disagreement_count: int
    single_method_count: int
    unavailable_count: int
    agreement_rate: float | None


def prediction_label(
    probability_a: float,
    probability_b: float,
    labels: tuple[str, str],
) -> str:
    """Distinguish probability ties from modal predictions."""
    if abs(probability_a - probability_b) < _TIE_TOLERANCE:
        return "Tie"
    return labels[0] if probability_a > probability_b else labels[1]


def prediction_values(
    result: WorkbenchResult,
    labels: tuple[str, str],
) -> tuple[PredictionValue, ...]:
    """Return stored predictive labels and confidence, excluding Shannon."""
    match result:
        case VMMAnalysis(records=records):
            return _prediction_values_from_pairs(
                tuple(
                    (str(item.sequence_id), item.probability_a, item.probability_b)
                    for item in records
                ),
                labels,
            )
        case MarkovBatchAnalysis(records=records):
            return _prediction_values_from_pairs(
                tuple(
                    _markov_probabilities(
                        str(item.sequence_id), item.rows[-1].predictive
                    )
                    for item in records
                ),
                labels,
            )
        case HMMBatchAnalysis(records=records):
            return _prediction_values_from_pairs(
                tuple(
                    _markov_probabilities(
                        str(item.sequence_id),
                        item.analysis.rows[-1].predictive,
                    )
                    for item in records
                ),
                labels,
            )
        case ShannonBatchAnalysis():
            return ()
    assert_never(result)


def _prediction_values_from_pairs(
    pairs: tuple[tuple[str, float | None, float | None], ...],
    labels: tuple[str, str],
) -> tuple[PredictionValue, ...]:
    return tuple(
        PredictionValue(
            identifier,
            (
                None
                if probability_a is None or probability_b is None
                else prediction_label(probability_a, probability_b, labels)
            ),
            (
                None
                if probability_a is None or probability_b is None
                else max(probability_a, probability_b)
            ),
        )
        for identifier, probability_a, probability_b in pairs
    )


def agreement_summary(
    results: tuple[WorkbenchResult, ...],
    labels: tuple[str, str],
) -> AgreementSummary:
    """Summarize agreement across predictive methods; Shannon is excluded."""
    methods = tuple(
        values
        for result in results
        if (values := prediction_values(result, labels))
    )
    identifiers = tuple(item.sequence_id for item in methods[0]) if methods else ()
    rows: list[AgreementRow] = []
    for index, identifier in enumerate(identifiers):
        predictions = tuple(method[index].prediction for method in methods)
        if any(prediction is None for prediction in predictions):
            status = "Unavailable"
        elif len(methods) == 1:
            status = "Single predictive method"
        elif len(set(predictions)) == 1:
            status = "Agreement"
        else:
            status = "Disagreement"
        rows.append(AgreementRow(identifier, status))
    agreements = sum(row.status == "Agreement" for row in rows)
    disagreements = sum(row.status == "Disagreement" for row in rows)
    comparable = agreements + disagreements
    return AgreementSummary(
        rows=tuple(rows),
        agreement_count=agreements,
        disagreement_count=disagreements,
        single_method_count=sum(
            row.status == "Single predictive method" for row in rows
        ),
        unavailable_count=sum(row.status == "Unavailable" for row in rows),
        agreement_rate=None if comparable == 0 else agreements / comparable,
    )


def _markov_probabilities(
    identifier: str,
    predictive: FloatArray | None,
) -> tuple[str, float | None, float | None]:
    if predictive is None:
        return identifier, None, None
    probability_a, probability_b = float_values(predictive)
    return identifier, probability_a, probability_b
