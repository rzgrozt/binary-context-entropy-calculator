"""Stable JSON and high-precision CSV exports for first-order Markov results."""

import json
from typing import Literal, TypedDict

from binary_entropy.domain import float_values
from binary_entropy.markov_csv import CsvCell, markov_csv_text
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovResultScope,
    UnavailableStationaryDistribution,
    UniqueStationaryDistribution,
)

type JsonProbabilityRow = list[float] | None


class _MarkovModelJson(TypedDict):
    schema_version: Literal[1]
    method: Literal["markov"]
    observable_labels: list[str]
    markov_order: Literal[1]
    estimation_method: str
    smoothing_alpha: float
    transition_counts: list[list[int]]
    transition_matrix: list[JsonProbabilityRow]
    starting_distribution: list[float] | None
    stationary_distribution: list[float] | None
    source_sequence_count: int
    source_transition_count: int
    prediction_mode: str
    result_scope: str
    empirical_conditional_entropy_bits: float | None
    entropy_rate_bits: float | None
    stationary_status: str


class _PerSequenceModelJson(TypedDict):
    sequence_id: str
    estimation_method: str
    smoothing_alpha: float
    transition_counts: list[list[int]]
    transition_matrix: list[JsonProbabilityRow]
    starting_distribution: list[float] | None
    source_sequence_count: int
    source_transition_count: int


class _PerSequenceModelsJson(TypedDict):
    schema_version: Literal[1]
    method: Literal["markov"]
    observable_labels: list[str]
    markov_order: Literal[1]
    prediction_mode: str
    result_scope: str
    models: list[_PerSequenceModelJson]


def markov_model_json(analysis: MarkovBatchAnalysis) -> bytes:
    """Serialize fitted Markov models without source sequences."""
    model = analysis.model
    payload: _MarkovModelJson | _PerSequenceModelsJson
    match analysis.result_scope:
        case MarkovResultScope.POOLED:
            transition_matrix = [
                None if row is None else list(float_values(row))
                for row in model.transition_matrix
            ]
            starting_distribution = (
                None
                if model.starting_distribution is None
                else list(float_values(model.starting_distribution))
            )
            match analysis.stationary:
                case UniqueStationaryDistribution(distribution=distribution):
                    stationary = list(float_values(distribution))
                    stationary_status = "unique"
                case UnavailableStationaryDistribution(reason=reason):
                    stationary = None
                    stationary_status = reason.value
            payload = {
                "schema_version": 1,
                "method": "markov",
                "observable_labels": list(model.observable_labels),
                "markov_order": model.markov_order,
                "estimation_method": model.estimation_method.value,
                "smoothing_alpha": model.smoothing_alpha,
                "transition_counts": [
                    list(model.transition_counts[0]),
                    list(model.transition_counts[1]),
                ],
                "transition_matrix": transition_matrix,
                "starting_distribution": starting_distribution,
                "stationary_distribution": stationary,
                "source_sequence_count": model.source_sequence_count,
                "source_transition_count": model.source_transition_count,
                "prediction_mode": analysis.prediction_mode.value,
                "result_scope": analysis.result_scope.value,
                "empirical_conditional_entropy_bits": (
                    analysis.empirical_conditional_entropy_bits
                ),
                "entropy_rate_bits": analysis.entropy_rate_bits,
                "stationary_status": stationary_status,
            }
        case MarkovResultScope.PER_SEQUENCE:
            models: list[_PerSequenceModelJson] = []
            for record in analysis.records:
                record_model = record.model
                record_transition_matrix = [
                    None if row is None else list(float_values(row))
                    for row in record_model.transition_matrix
                ]
                record_starting_distribution = (
                    None
                    if record_model.starting_distribution is None
                    else list(float_values(record_model.starting_distribution))
                )
                models.append(
                    {
                        "sequence_id": record.sequence_id,
                        "estimation_method": record_model.estimation_method.value,
                        "smoothing_alpha": record_model.smoothing_alpha,
                        "transition_counts": [
                            list(record_model.transition_counts[0]),
                            list(record_model.transition_counts[1]),
                        ],
                        "transition_matrix": record_transition_matrix,
                        "starting_distribution": record_starting_distribution,
                        "source_sequence_count": record_model.source_sequence_count,
                        "source_transition_count": (
                            record_model.source_transition_count
                        ),
                    }
                )
            payload = {
                "schema_version": 1,
                "method": "markov",
                "observable_labels": list(model.observable_labels),
                "markov_order": model.markov_order,
                "prediction_mode": analysis.prediction_mode.value,
                "result_scope": analysis.result_scope.value,
                "models": models,
            }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    return (text + "\n").encode("utf-8", errors="strict")


def markov_sequence_csv(analysis: MarkovBatchAnalysis) -> str:
    """Serialize independent prefix rows with round-trip probability precision."""
    label_a, label_b = analysis.model.observable_labels
    columns = (
        "sequence_id",
        "method",
        "result_scope",
        "prediction_mode",
        "markov_order",
        "estimation_method",
        "smoothing_alpha",
        "source_sequence_count",
        "source_transition_count",
        "sequence_length",
        "depth",
        "context_symbol",
        "observed_next_symbol",
        "fitted_transition_count",
        f"predictive_probability_{label_a}",
        f"predictive_probability_{label_b}",
        "predictive_entropy_bits",
        "predicted_symbol",
        "actual_target_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "actual_target_classification",
    )
    rows: list[tuple[CsvCell, ...]] = []
    for record in analysis.records:
        model = record.model
        for prefix in record.rows:
            if prefix.predictive is None:
                probability_a = None
                probability_b = None
            else:
                probability_a, probability_b = float_values(prefix.predictive)
            context_symbol = (
                None
                if prefix.context is None
                else model.observable_labels[prefix.context[0]]
            )
            observed_next_symbol = (
                None
                if prefix.observed_next_index is None
                else model.observable_labels[prefix.observed_next_index]
            )
            predicted_symbol = (
                None
                if prefix.predicted_index is None
                else model.observable_labels[prefix.predicted_index]
            )
            target = (
                record.target_assessment
                if prefix.depth == record.sequence_length
                else None
            )
            rows.append(
                (
                    record.sequence_id,
                    analysis.method,
                    analysis.result_scope.value,
                    prefix.prediction_mode.value,
                    model.markov_order,
                    model.estimation_method.value,
                    model.smoothing_alpha,
                    model.source_sequence_count,
                    model.source_transition_count,
                    record.sequence_length,
                    prefix.depth,
                    context_symbol,
                    observed_next_symbol,
                    prefix.fitted_transition_count,
                    probability_a,
                    probability_b,
                    prefix.entropy_bits,
                    predicted_symbol,
                    (
                        None
                        if target is None
                        else model.observable_labels[target.actual_target_index]
                    ),
                    None if target is None else target.probability,
                    None if target is None else target.surprisal_bits,
                    None if target is None else target.classification.value,
                )
            )
    return markov_csv_text(columns, rows)
