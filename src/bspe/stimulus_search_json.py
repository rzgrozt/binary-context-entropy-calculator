"""Stable generator configuration JSON export."""

import json
import math
from collections.abc import Sequence
from typing import TypedDict, TypeIs, assert_never

from bspe.stimulus_search import evaluate_candidate
from bspe.stimulus_search_results import (
    MatchTolerances,
    MetricSummary,
    SearchResult,
    StimulusCandidate,
    ValidationReport,
)
from bspe.stimulus_search_types import InclusiveRange
from bspe.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMSmoothing,
)

type JsonMetric = float | str


class _MetricSummaryJson(TypedDict):
    minimum: JsonMetric
    mean: JsonMetric
    maximum: JsonMetric


class _MatchTolerancesJson(TypedDict):
    entropy: JsonMetric
    prediction_strength: JsonMetric
    a_proportion: JsonMetric
    switch_rate: JsonMetric
    run_structure: JsonMetric
    effective_depth: JsonMetric


class _ValidationReportJson(TypedDict):
    stimulus_count: int
    predicted_a_count: int
    predicted_b_count: int
    tie_count: int
    unavailable_count: int
    expected_target_count: int
    unexpected_target_count: int
    tie_target_count: int
    unassigned_target_count: int
    a_count: _MetricSummaryJson | None
    b_count: _MetricSummaryJson | None
    a_proportion: _MetricSummaryJson | None
    b_proportion: _MetricSummaryJson | None
    switches: _MetricSummaryJson | None
    switch_rate: _MetricSummaryJson | None
    longest_a_run: _MetricSummaryJson | None
    longest_b_run: _MetricSummaryJson | None
    longest_run: _MetricSummaryJson | None
    predicted_probability: _MetricSummaryJson | None
    predictive_entropy: _MetricSummaryJson | None
    effective_depth: _MetricSummaryJson | None
    context_support: _MetricSummaryJson | None
    target_surprisal: _MetricSummaryJson | None
    imbalance_flags: tuple[str, ...]


def stimulus_generator_config_json(
    result: SearchResult,
    *,
    match_tolerances: MatchTolerances | None = None,
    assignment_seed: int | None = None,
    validation_report: ValidationReport | None = None,
    final_candidates: Sequence[StimulusCandidate] | None = None,
) -> str:
    """Export the exact stored generator snapshot."""
    config = result.config
    constraints = config.constraints
    payload = {
        "package_version": result.package_version,
        "generation_strategy": "random_sample_without_replacement",
        "selection_rule": "full_sample_then_rank_score_then_stimulus_id",
        "matching_rule": "equal_length_symbol_exchanged_composition_and_runs",
        "selected_stimulus_ids": [str(row.stimulus_id) for row in result.selected],
        "final_stimuli": None
        if final_candidates is None
        else [
            {
                "stimulus_id": str(row.stimulus_id),
                "sequence": "".join("AB"[symbol] for symbol in row.sequence),
                "group_id": row.group_id,
                "pair_id": row.pair_id,
                "condition": row.condition,
                "symbol_mapping": row.symbol_mapping,
                "target": None if row.target_index is None else "AB"[row.target_index],
                "original_constraint_failures": evaluate_candidate(row, constraints),
            }
            for row in final_candidates
        ],
        "timestamp_utc": result.timestamp_utc,
        "sequence_length": config.sequence_length,
        "desired_stimuli": config.desired_stimuli,
        "seed": config.seed,
        "candidate_limit": config.candidate_limit,
        "symbol_mapping": config.symbol_mapping,
        "vmm_estimator": _estimator(config.vmm_config.smoothing),
        "smoothing_alpha": config.vmm_config.smoothing.alpha,
        "minimum_support": config.vmm_config.minimum_support,
        "result_scope": "per_sequence",
        "constraints": {
            "predicted_symbol": constraints.predicted_symbol.value,
            "predicted_probability": _range(constraints.predicted_probability),
            "predictive_entropy": _range(constraints.predictive_entropy),
            "effective_depth": _range(constraints.effective_depth),
            "context_support": _range(constraints.context_support),
            "a_count": _range(constraints.a_count),
            "a_proportion": _range(constraints.a_proportion),
            "switches": _range(constraints.switches),
            "switch_rate": _range(constraints.switch_rate),
            "longest_a_run": _range(constraints.longest_a_run),
            "longest_b_run": _range(constraints.longest_b_run),
            "longest_run": _range(constraints.longest_run),
        },
        "preferences": tuple(
            {
                "metric": preference.metric.value,
                "target": preference.target,
                "weight": preference.weight,
            }
            for preference in config.preferences
        ),
        "evaluated_count": result.evaluated_count,
        "accepted_count": result.accepted_count,
        "selected_count": result.selected_count,
        "status": result.status.value,
        "partial_reason": (
            None if result.partial_reason is None else result.partial_reason.value
        ),
        "match_tolerances": _match_tolerances(match_tolerances),
        "assignment_seed": assignment_seed,
        "validation_report": _validation_report(validation_report),
        "constraint_violations": tuple(
            {
                "constraint": violation.constraint,
                "count": violation.count,
                "frequency": violation.frequency,
            }
            for violation in result.violations
        ),
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _range(value: InclusiveRange | None) -> tuple[float, float] | None:
    return None if value is None else (value.minimum, value.maximum)


def _estimator(smoothing: VMMSmoothing) -> str:
    match smoothing:
        case KTSmoothing():
            return "krichevsky_trofimov"
        case MLESmoothing():
            return "maximum_likelihood"
        case unreachable if not _is_additive_smoothing(unreachable):
            assert_never(unreachable)
        case AdditiveSmoothing():
            return "additive_smoothing"


def _is_additive_smoothing(smoothing: VMMSmoothing) -> TypeIs[AdditiveSmoothing]:
    return isinstance(smoothing, AdditiveSmoothing)


def _match_tolerances(
    tolerances: MatchTolerances | None,
) -> _MatchTolerancesJson | None:
    if tolerances is None:
        return None
    return {
        "entropy": _json_metric(tolerances.entropy),
        "prediction_strength": _json_metric(tolerances.prediction_strength),
        "a_proportion": _json_metric(tolerances.a_proportion),
        "switch_rate": _json_metric(tolerances.switch_rate),
        "run_structure": _json_metric(tolerances.run_structure),
        "effective_depth": _json_metric(tolerances.effective_depth),
    }


def _validation_report(
    report: ValidationReport | None,
) -> _ValidationReportJson | None:
    if report is None:
        return None
    return {
        "stimulus_count": report.stimulus_count,
        "predicted_a_count": report.predicted_a_count,
        "predicted_b_count": report.predicted_b_count,
        "tie_count": report.tie_count,
        "unavailable_count": report.unavailable_count,
        "expected_target_count": report.expected_target_count,
        "unexpected_target_count": report.unexpected_target_count,
        "tie_target_count": report.tie_target_count,
        "unassigned_target_count": report.unassigned_target_count,
        "a_count": _metric_summary(report.a_count),
        "b_count": _metric_summary(report.b_count),
        "a_proportion": _metric_summary(report.a_proportion),
        "b_proportion": _metric_summary(report.b_proportion),
        "switches": _metric_summary(report.switches),
        "switch_rate": _metric_summary(report.switch_rate),
        "longest_a_run": _metric_summary(report.longest_a_run),
        "longest_b_run": _metric_summary(report.longest_b_run),
        "longest_run": _metric_summary(report.longest_run),
        "predicted_probability": _metric_summary(report.predicted_probability),
        "predictive_entropy": _metric_summary(report.predictive_entropy),
        "effective_depth": _metric_summary(report.effective_depth),
        "context_support": _metric_summary(report.context_support),
        "target_surprisal": _metric_summary(report.target_surprisal),
        "imbalance_flags": report.imbalance_flags,
    }


def _metric_summary(summary: MetricSummary | None) -> _MetricSummaryJson | None:
    if summary is None:
        return None
    return {
        "minimum": _json_metric(summary.minimum),
        "mean": _json_metric(summary.mean),
        "maximum": _json_metric(summary.maximum),
    }


def _json_metric(value: float) -> JsonMetric:
    if math.isfinite(value):
        return value
    return "infinity" if value > 0.0 else "-infinity"
