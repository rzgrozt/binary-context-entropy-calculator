"""Deterministic bounded orchestration for stimulus search."""

from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from itertools import batched
from typing import assert_never

from binary_entropy.stimulus_analysis import analyze_stimulus_records
from binary_entropy.stimulus_generation import generate_candidate_records
from binary_entropy.stimulus_search_results import (
    ConstraintViolation,
    SearchPartialReason,
    SearchResult,
    SearchStatus,
    StimulusCandidate,
)
from binary_entropy.stimulus_search_types import (
    InclusiveRange,
    InvalidStimulusSearchConfigurationError,
    PredictedSymbol,
    PreferenceMetric,
    SoftPreference,
    StimulusConstraints,
    StimulusSearchConfig,
)


def search_stimuli(
    config: StimulusSearchConfig,
    *,
    timestamp: datetime | None = None,
) -> SearchResult:
    """Search a bounded deterministic candidate sample."""
    created_at = timestamp or datetime.now(UTC)
    if created_at.utcoffset() is None:
        parameter = "timestamp"
        raise InvalidStimulusSearchConfigurationError(parameter, repr(created_at))
    candidates = (
        candidate
        for batch in batched(generate_candidate_records(config), 64, strict=False)
        for candidate in analyze_stimulus_records(batch, config)
    )
    evaluated_count = 0
    failures: Counter[str] = Counter()
    accepted: list[StimulusCandidate] = []
    for candidate in candidates:
        evaluated_count += 1
        failed = evaluate_candidate(candidate, config.constraints)
        failures.update(failed)
        if not failed:
            accepted.append(candidate)
    ranked = tuple(
        sorted(
            (
                replace(
                    candidate, rank_score=_rank_score(candidate, config.preferences)
                )
                for candidate in accepted
            ),
            key=lambda candidate: (candidate.rank_score, candidate.stimulus_id),
        )
    )
    selected = ranked[: config.desired_stimuli]
    violations = tuple(
        ConstraintViolation(name, count, count / evaluated_count)
        for name, count in sorted(failures.items())
    )
    status = (
        SearchStatus.COMPLETE
        if len(selected) == config.desired_stimuli
        else SearchStatus.PARTIAL
    )
    partial_reason = (
        None
        if status is SearchStatus.COMPLETE
        else SearchPartialReason.CANDIDATE_LIMIT
        if config.candidate_limit < 1 << config.sequence_length
        else SearchPartialReason.UNIVERSE_EXHAUSTED
    )
    return SearchResult(
        config=config,
        timestamp_utc=created_at.astimezone(UTC).isoformat(),
        package_version="0.1.0",
        evaluated_count=evaluated_count,
        accepted_count=len(ranked),
        selected_count=len(selected),
        status=status,
        partial_reason=partial_reason,
        accepted=ranked,
        selected=selected,
        violations=violations,
    )


def evaluate_candidate(
    candidate: StimulusCandidate,
    constraints: StimulusConstraints,
) -> tuple[str, ...]:
    """Return every failed hard filter, independently of ranking or generation."""
    failed: list[str] = []
    match constraints.predicted_symbol:
        case PredictedSymbol.A:
            if candidate.predicted_target_index != 0:
                failed.append("predicted_symbol")
        case PredictedSymbol.B:
            if candidate.predicted_target_index != 1:
                failed.append("predicted_symbol")
        case remaining if remaining is not PredictedSymbol.EITHER:
            assert_never(remaining)
        case PredictedSymbol.EITHER:
            pass
    predicted_probability = _predicted_probability(candidate)
    _check_range(
        failed,
        "predicted_probability",
        constraints.predicted_probability,
        predicted_probability,
    )
    _check_range(
        failed,
        "predictive_entropy",
        constraints.predictive_entropy,
        candidate.predictive_entropy_bits,
    )
    _check_range(
        failed,
        "effective_depth",
        constraints.effective_depth,
        candidate.effective_context_depth,
    )
    _check_range(
        failed, "context_support", constraints.context_support, candidate.support_count
    )
    _check_range(failed, "a_count", constraints.a_count, candidate.metrics.a_count)
    _check_range(
        failed, "a_proportion", constraints.a_proportion, candidate.metrics.a_proportion
    )
    _check_range(failed, "switches", constraints.switches, candidate.metrics.switches)
    _check_range(
        failed, "switch_rate", constraints.switch_rate, candidate.metrics.switch_rate
    )
    _check_range(
        failed,
        "longest_a_run",
        constraints.longest_a_run,
        candidate.metrics.longest_a_run,
    )
    _check_range(
        failed,
        "longest_b_run",
        constraints.longest_b_run,
        candidate.metrics.longest_b_run,
    )
    _check_range(
        failed, "longest_run", constraints.longest_run, candidate.metrics.longest_run
    )
    return tuple(failed)


def _check_range(
    failed: list[str],
    name: str,
    expected: InclusiveRange | None,
    value: float | None,
) -> None:
    if expected is not None and (value is None or not expected.contains(value)):
        failed.append(name)


def _predicted_probability(candidate: StimulusCandidate) -> float | None:
    if candidate.probability_a is None or candidate.probability_b is None:
        return None
    return max(candidate.probability_a, candidate.probability_b)


def _rank_score(
    candidate: StimulusCandidate,
    preferences: tuple[SoftPreference, ...],
) -> float:
    score = 0.0
    for preference in preferences:
        value = _preference_value(candidate, preference.metric)
        if value is None:
            return float("inf")
        score += preference.weight * abs(value - preference.target)
    return score


def _preference_value(
    candidate: StimulusCandidate,
    metric: PreferenceMetric,
) -> float | None:
    match metric:
        case PreferenceMetric.PREDICTED_PROBABILITY:
            value = _predicted_probability(candidate)
        case PreferenceMetric.PREDICTIVE_ENTROPY:
            value = candidate.predictive_entropy_bits
        case PreferenceMetric.EFFECTIVE_DEPTH:
            value = candidate.effective_context_depth
        case PreferenceMetric.CONTEXT_SUPPORT:
            value = candidate.support_count
        case PreferenceMetric.A_PROPORTION:
            value = candidate.metrics.a_proportion
        case PreferenceMetric.SWITCH_RATE:
            value = candidate.metrics.switch_rate
        case remaining:
            if remaining is PreferenceMetric.LONGEST_RUN:
                return candidate.metrics.longest_run
            assert_never(remaining)
    return value
