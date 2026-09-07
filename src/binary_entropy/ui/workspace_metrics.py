"""Stored-value metric projections for the result workspace."""

from dataclasses import dataclass
from statistics import fmean, median
from typing import assert_never

from binary_entropy.domain import float_values
from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.ui.tokens import format_ui_decimal
from binary_entropy.ui.vmm_results import vmm_context_label
from binary_entropy.ui.workspace_agreement import prediction_label, prediction_values
from binary_entropy.vmm_types import VMMAnalysis
from binary_entropy.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One labeled value for a native Streamlit metric."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class EntropyPoint:
    """One stored entropy value identified by source record."""

    sequence_id: str
    entropy_bits: float


def single_metric_values(
    result: WorkbenchResult,
    labels: tuple[str, str],
) -> tuple[MetricValue, ...]:
    """Project the selected record's stored final values into metric labels."""
    match result:
        case VMMAnalysis(records=(record,)):
            context = (
                "Unavailable"
                if record.context_used is None
                else vmm_context_label(record.context_used, labels)
            )
            return (
                MetricValue(
                    "Effective context depth",
                    _display_integer(record.effective_context_depth),
                ),
                MetricValue(f"P(next {labels[0]})", _display(record.probability_a)),
                MetricValue(f"P(next {labels[1]})", _display(record.probability_b)),
                MetricValue(
                    "Predicted target",
                    _prediction(record.probability_a, record.probability_b, labels),
                ),
                MetricValue(
                    "Predictive entropy (bits)",
                    _display(record.predictive_entropy_bits),
                ),
                MetricValue("Context used", context),
            )
        case MarkovBatchAnalysis(records=(record,)):
            final = record.rows[-1]
            probabilities = (
                (None, None)
                if final.predictive is None
                else float_values(final.predictive)
            )
            context = (
                "Unavailable"
                if final.context is None
                else f"{labels[final.context[0]]} / available"
            )
            return (
                MetricValue(f"P(next {labels[0]})", _display(probabilities[0])),
                MetricValue(f"P(next {labels[1]})", _display(probabilities[1])),
                MetricValue(
                    "Prediction",
                    _prediction(probabilities[0], probabilities[1], labels),
                ),
                MetricValue("Predictive entropy (bits)", _display(final.entropy_bits)),
                MetricValue("Effective first-order context / status", context),
            )
        case HMMBatchAnalysis(records=(record,)):
            final = record.analysis.rows[-1]
            probability_a, probability_b = float_values(final.predictive)
            return (
                MetricValue(f"Final P(next {labels[0]})", _display(probability_a)),
                MetricValue(f"Final P(next {labels[1]})", _display(probability_b)),
                MetricValue(
                    "Prediction",
                    _prediction(probability_a, probability_b, labels),
                ),
                MetricValue("Predictive entropy (bits)", _display(final.entropy_bits)),
                MetricValue("Sequence depth", str(final.depth)),
            )
        case ShannonBatchAnalysis(records=(record,)):
            summary = record.summary
            probabilities = (
                (None, None)
                if summary.symbol_probabilities is None
                else float_values(summary.symbol_probabilities)
            )
            return (
                MetricValue("Observed entropy (bits)", _display(summary.entropy_bits)),
                MetricValue(f"Observed P({labels[0]})", _display(probabilities[0])),
                MetricValue(f"Observed P({labels[1]})", _display(probabilities[1])),
                MetricValue("Maximum binary entropy (bits)", "1.000"),
                MetricValue(
                    "Normalized entropy (H / 1 bit)",
                    _display(summary.entropy_bits),
                ),
            )
        case (
            VMMAnalysis()
            | MarkovBatchAnalysis()
            | HMMBatchAnalysis()
            | ShannonBatchAnalysis()
        ):
            return ()
    assert_never(result)


def aggregate_metric_values(
    result: WorkbenchResult,
    labels: tuple[str, str],
) -> tuple[MetricValue, ...]:
    """Summarize selected records without changing any stored result."""
    entropies = tuple(point.entropy_bits for point in entropy_points(result))
    predictions = prediction_values(result, labels)
    depths = _effective_depths(result)
    available_predictions = tuple(
        item for item in predictions if item.prediction is not None
    )
    confidences = tuple(
        item.confidence for item in predictions if item.confidence is not None
    )
    prediction_metrics = (
        MetricValue(
            f"Predicted {labels[0]}",
            str(sum(item.prediction == labels[0] for item in available_predictions)),
        ),
        MetricValue(
            f"Predicted {labels[1]}",
            str(sum(item.prediction == labels[1] for item in available_predictions)),
        ),
        MetricValue("Mean confidence", _mean_value(confidences)),
        MetricValue("Median confidence", _median_value(confidences)),
    )
    values = (
        MetricValue("Selected records", str(_record_count(result))),
        MetricValue("Mean entropy (bits)", _mean_value(entropies)),
        MetricValue("Median entropy (bits)", _median_value(entropies)),
    )
    match result:
        case ShannonBatchAnalysis():
            return (*values, MetricValue("Predictive confidence", "Not applicable"))
        case VMMAnalysis():
            return (
                *values,
                MetricValue("Mean effective depth", _mean_value(depths)),
                MetricValue("Median effective depth", _median_value(depths)),
                *prediction_metrics,
            )
        case MarkovBatchAnalysis() | HMMBatchAnalysis():
            return (*values, *prediction_metrics)
    assert_never(result)


def entropy_points(result: WorkbenchResult) -> tuple[EntropyPoint, ...]:
    """Return available stored final entropy values in record order."""
    match result:
        case VMMAnalysis(records=records):
            return _available_entropy_points(
                tuple(
                    (str(item.sequence_id), item.predictive_entropy_bits)
                    for item in records
                )
            )
        case MarkovBatchAnalysis(records=records):
            return _available_entropy_points(
                tuple(
                    (str(item.sequence_id), item.rows[-1].entropy_bits)
                    for item in records
                )
            )
        case HMMBatchAnalysis(records=records):
            return _available_entropy_points(
                tuple(
                    (str(item.sequence_id), item.analysis.rows[-1].entropy_bits)
                    for item in records
                )
            )
        case ShannonBatchAnalysis(records=records):
            return _available_entropy_points(
                tuple(
                    (str(item.sequence_id), item.summary.entropy_bits)
                    for item in records
                )
            )
    assert_never(result)


def _available_entropy_points(
    pairs: tuple[tuple[str, float | None], ...],
) -> tuple[EntropyPoint, ...]:
    return tuple(
        EntropyPoint(identifier, value)
        for identifier, value in pairs
        if value is not None
    )


def _record_count(result: WorkbenchResult) -> int:
    return len(result.records)


def _effective_depths(result: WorkbenchResult) -> tuple[int, ...]:
    match result:
        case VMMAnalysis(records=records):
            return tuple(
                item.effective_context_depth
                for item in records
                if item.effective_context_depth is not None
            )
        case MarkovBatchAnalysis() | HMMBatchAnalysis() | ShannonBatchAnalysis():
            return ()
    assert_never(result)


def _prediction(
    probability_a: float | None,
    probability_b: float | None,
    labels: tuple[str, str],
) -> str:
    if probability_a is None or probability_b is None:
        return "Unavailable"
    return prediction_label(probability_a, probability_b, labels)


def _display(value: float | None) -> str:
    return "Unavailable" if value is None else format_ui_decimal(value)


def _display_integer(value: int | None) -> str:
    return "Unavailable" if value is None else str(value)


def _mean_value(values: tuple[float, ...] | tuple[int, ...]) -> str:
    return "Unavailable" if not values else format_ui_decimal(float(fmean(values)))


def _median_value(values: tuple[float, ...] | tuple[int, ...]) -> str:
    return "Unavailable" if not values else format_ui_decimal(float(median(values)))
