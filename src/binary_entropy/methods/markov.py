"""First-order binary Markov fitting and prefix analysis."""

import math
from dataclasses import dataclass

import numpy as np

from binary_entropy.analysis import assess_target
from binary_entropy.domain import FloatArray, ObservableIndex, float_values
from binary_entropy.errors import InvalidSmoothingAlphaError
from binary_entropy.information import binary_entropy
from binary_entropy.markov_information import (
    empirical_conditional_entropy,
    entropy_rate,
    stationary_distribution,
)
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovContext,
    MarkovEstimation,
    MarkovModel,
    MarkovPredictionMode,
    MarkovPrefixResult,
    MarkovRecordAnalysis,
    MarkovResultScope,
    TransitionCounts,
)
from binary_entropy.records import SequenceDataset, SequenceRecord


@dataclass(frozen=True, slots=True)
class _MarkovAnalysisPlan:
    smoothing_alpha: float
    prediction_mode: MarkovPredictionMode
    result_scope: MarkovResultScope


@dataclass(frozen=True, slots=True)
class _MarkovAnalysisContext:
    dataset: SequenceDataset
    plan: _MarkovAnalysisPlan
    pooled_model: MarkovModel


def fit_markov(
    dataset: SequenceDataset,
    smoothing_alpha: float = 0.0,
) -> MarkovModel:
    """Fit a pooled first-order model without crossing record boundaries."""
    if not math.isfinite(smoothing_alpha) or smoothing_alpha < 0.0:
        raise InvalidSmoothingAlphaError(smoothing_alpha=smoothing_alpha)
    mutable_counts = [[0, 0], [0, 0]]
    start_counts = [0, 0]
    for record in dataset.records:
        if record.sequence:
            start_counts[record.sequence[0]] += 1
        for current, following in zip(
            record.sequence,
            record.sequence[1:],
            strict=False,
        ):
            mutable_counts[current][following] += 1
    counts: TransitionCounts = (
        (mutable_counts[0][0], mutable_counts[0][1]),
        (mutable_counts[1][0], mutable_counts[1][1]),
    )
    rows: list[FloatArray | None] = []
    for row_counts in counts:
        row_total = row_counts[0] + row_counts[1]
        if row_total == 0 and smoothing_alpha == 0.0:
            rows.append(None)
            continue
        denominator = row_total + 2.0 * smoothing_alpha
        row: FloatArray = np.array(
            [
                (row_counts[0] + smoothing_alpha) / denominator,
                (row_counts[1] + smoothing_alpha) / denominator,
            ],
            dtype=np.float64,
        )
        rows.append(row)
    nonempty_count = start_counts[0] + start_counts[1]
    starting_distribution: FloatArray | None = None
    if nonempty_count > 0:
        starting_distribution = np.array(
            [start_counts[0] / nonempty_count, start_counts[1] / nonempty_count],
            dtype=np.float64,
        )
    estimation_method = (
        MarkovEstimation.MAXIMUM_LIKELIHOOD
        if smoothing_alpha == 0.0
        else MarkovEstimation.ADDITIVE_SMOOTHING
    )
    source_transition_count = sum(sum(row_counts) for row_counts in counts)
    return MarkovModel(
        observable_labels=dataset.labels.observables,
        estimation_method=estimation_method,
        smoothing_alpha=smoothing_alpha,
        transition_counts=counts,
        transition_matrix=(rows[0], rows[1]),
        starting_distribution=starting_distribution,
        source_sequence_count=len(dataset.records),
        source_transition_count=source_transition_count,
    )


def predict_markov(
    model: MarkovModel,
    context: MarkovContext,
) -> FloatArray | None:
    """Select the transition row for the context's final observable."""
    return model.transition_matrix[context[-1]]


def analyze_markov(
    dataset: SequenceDataset,
    smoothing_alpha: float = 0.0,
    prediction_mode: MarkovPredictionMode = MarkovPredictionMode.FIXED_MODEL,
) -> MarkovBatchAnalysis:
    """Analyze records with one pooled full-dataset model."""
    return _analyze_markov(
        dataset,
        _MarkovAnalysisPlan(
            smoothing_alpha,
            prediction_mode,
            MarkovResultScope.POOLED,
        ),
    )


def analyze_markov_per_sequence(
    dataset: SequenceDataset,
    smoothing_alpha: float = 0.0,
    prediction_mode: MarkovPredictionMode = MarkovPredictionMode.FIXED_MODEL,
) -> MarkovBatchAnalysis:
    """Analyze records under independently fitted full-sequence models."""
    return _analyze_markov(
        dataset,
        _MarkovAnalysisPlan(
            smoothing_alpha,
            prediction_mode,
            MarkovResultScope.PER_SEQUENCE,
        ),
    )


def _analyze_markov(
    dataset: SequenceDataset,
    plan: _MarkovAnalysisPlan,
) -> MarkovBatchAnalysis:
    model = fit_markov(dataset, plan.smoothing_alpha)
    analysis_context = _MarkovAnalysisContext(dataset, plan, model)
    record_results: list[MarkovRecordAnalysis] = []
    for record in dataset.records:
        record_model = _record_model(analysis_context, record)
        prefix_results: list[MarkovPrefixResult] = []
        for depth in range(len(record.sequence) + 1):
            context: MarkovContext | None
            predictive: FloatArray | None
            entropy_bits: float | None
            predicted_index: ObservableIndex | None
            fitted_transition_count: int
            observed_next_index = (
                record.sequence[depth] if depth < len(record.sequence) else None
            )
            if depth == 0:
                context = None
                predictive = None
                entropy_bits = None
                predicted_index = None
                match plan.prediction_mode:
                    case MarkovPredictionMode.FIXED_MODEL:
                        fitted_transition_count = record_model.source_transition_count
                    case MarkovPredictionMode.CUMULATIVE_PREFIX:
                        fitted_transition_count = 0
            else:
                context = (record.sequence[depth - 1],)
                selected_model: MarkovModel
                match plan.prediction_mode:
                    case MarkovPredictionMode.FIXED_MODEL:
                        selected_model = record_model
                    case MarkovPredictionMode.CUMULATIVE_PREFIX:
                        prefix_record = SequenceRecord(
                            sequence_id=record.sequence_id,
                            sequence=record.sequence[:depth],
                        )
                        selected_model = fit_markov(
                            SequenceDataset(dataset.labels, (prefix_record,)),
                            plan.smoothing_alpha,
                        )
                fitted_transition_count = selected_model.source_transition_count
                predictive = predict_markov(selected_model, context)
                if predictive is None:
                    entropy_bits = None
                    predicted_index = None
                else:
                    probability_0, probability_1 = float_values(predictive)
                    entropy_bits = binary_entropy(probability_0)
                    predicted_index = 0 if probability_0 >= probability_1 else 1
            prefix_results.append(
                MarkovPrefixResult(
                    depth=depth,
                    context=context,
                    prediction_mode=plan.prediction_mode,
                    fitted_transition_count=fitted_transition_count,
                    observed_next_index=observed_next_index,
                    predictive=predictive,
                    entropy_bits=entropy_bits,
                    predicted_index=predicted_index,
                )
            )
        final_prediction = prefix_results[-1].predictive
        target_assessment = (
            assess_target(final_prediction, record.actual_target_index)
            if final_prediction is not None and record.actual_target_index is not None
            else None
        )
        record_results.append(
            MarkovRecordAnalysis(
                sequence_id=record.sequence_id,
                sequence=record.sequence,
                sequence_length=len(record.sequence),
                actual_target_index=record.actual_target_index,
                model=record_model,
                rows=tuple(prefix_results),
                target_assessment=target_assessment,
            )
        )
    stationary = stationary_distribution(model.transition_matrix)
    return MarkovBatchAnalysis(
        model=model,
        prediction_mode=plan.prediction_mode,
        records=tuple(record_results),
        empirical_conditional_entropy_bits=empirical_conditional_entropy(
            model.transition_counts
        ),
        stationary=stationary,
        entropy_rate_bits=entropy_rate(model.transition_matrix, stationary),
        result_scope=plan.result_scope,
    )


def _record_model(
    context: _MarkovAnalysisContext,
    record: SequenceRecord,
) -> MarkovModel:
    match context.plan.result_scope:
        case MarkovResultScope.POOLED:
            return context.pooled_model
        case MarkovResultScope.PER_SEQUENCE:
            return fit_markov(
                SequenceDataset(context.dataset.labels, (record,)),
                context.plan.smoothing_alpha,
            )
