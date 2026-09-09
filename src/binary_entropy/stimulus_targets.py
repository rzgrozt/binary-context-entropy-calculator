"""Pure target assignment and descriptive validation."""

from __future__ import annotations

import random
from dataclasses import replace
from enum import StrEnum
from statistics import fmean
from typing import TYPE_CHECKING

from binary_entropy.stimulus_search_results import (
    MetricSummary,
    StimulusCandidate,
    TargetCongruency,
    ValidationReport,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from binary_entropy.domain import ObservableIndex


def assign_targets(
    candidates: Sequence[StimulusCandidate], seed: int
) -> tuple[StimulusCandidate, ...]:
    """Assign balanced expected/unexpected and A/B targets."""
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)  # noqa: S311
    predicting_a = [
        index for index in order if candidates[index].predicted_target_index == 0
    ]
    predicting_b = [
        index for index in order if candidates[index].predicted_target_index == 1
    ]
    ties = [
        index
        for index in order
        if candidates[index].predicted_target_index is None
        and candidates[index].probability_a is not None
    ]
    expected_a, expected_b, tie_a = _balanced_counts(
        len(predicting_a), len(predicting_b), len(ties)
    )
    expected_indices = set(predicting_a[:expected_a] + predicting_b[:expected_b])
    tie_a_indices = set(ties[:tie_a])
    assigned = list(candidates)
    for index in order:
        candidate = candidates[index]
        if candidate.probability_a is None or candidate.probability_b is None:
            continue
        if candidate.predicted_target_index is None:
            target: ObservableIndex = 0 if index in tie_a_indices else 1
            congruency = TargetCongruency.TIE
        else:
            choose_expected = index in expected_indices
            target = (
                candidate.predicted_target_index
                if choose_expected
                else 1
                if candidate.predicted_target_index == 0
                else 0
            )
            congruency = (
                TargetCongruency.EXPECTED
                if choose_expected
                else TargetCongruency.UNEXPECTED
            )
        probability = (
            candidate.probability_a if target == 0 else candidate.probability_b
        )
        surprisal = (
            candidate.surprisal_a_bits if target == 0 else candidate.surprisal_b_bits
        )
        assigned[index] = replace(
            candidate,
            target_index=target,
            target_probability=probability,
            target_surprisal_bits=surprisal,
            target_congruency=congruency,
            condition=congruency.value,
        )
    return tuple(assigned)


def validate_stimuli(candidates: Sequence[StimulusCandidate]) -> ValidationReport:
    """Summarize candidate quality-control values descriptively."""
    predicted_a = sum(candidate.predicted_target_index == 0 for candidate in candidates)
    predicted_b = sum(candidate.predicted_target_index == 1 for candidate in candidates)
    ties = sum(
        candidate.predicted_target_index is None and candidate.probability_a is not None
        for candidate in candidates
    )
    unavailable = sum(candidate.probability_a is None for candidate in candidates)
    expected = sum(
        candidate.target_congruency is TargetCongruency.EXPECTED
        for candidate in candidates
    )
    unexpected = sum(
        candidate.target_congruency is TargetCongruency.UNEXPECTED
        for candidate in candidates
    )
    target_ties = sum(
        candidate.target_congruency is TargetCongruency.TIE for candidate in candidates
    )
    unassigned = sum(candidate.target_index is None for candidate in candidates)
    target_a = sum(candidate.target_index == 0 for candidate in candidates)
    target_b = sum(candidate.target_index == 1 for candidate in candidates)
    flags: list[str] = []
    if abs(predicted_a - predicted_b) > 1:
        flags.append("predicted_symbol_imbalance")
    if abs(expected - unexpected) > 1:
        flags.append("target_congruency_imbalance")
    if abs(target_a - target_b) > 1:
        flags.append("target_symbol_imbalance")
    return ValidationReport(
        stimulus_count=len(candidates),
        predicted_a_count=predicted_a,
        predicted_b_count=predicted_b,
        tie_count=ties,
        unavailable_count=unavailable,
        expected_target_count=expected,
        unexpected_target_count=unexpected,
        tie_target_count=target_ties,
        unassigned_target_count=unassigned,
        a_count=_summary(tuple(float(row.metrics.a_count) for row in candidates)),
        b_count=_summary(tuple(float(row.metrics.b_count) for row in candidates)),
        a_proportion=_summary(tuple(row.metrics.a_proportion for row in candidates)),
        b_proportion=_summary(tuple(row.metrics.b_proportion for row in candidates)),
        switches=_summary(tuple(float(row.metrics.switches) for row in candidates)),
        switch_rate=_summary(tuple(row.metrics.switch_rate for row in candidates)),
        longest_a_run=_summary(
            tuple(float(row.metrics.longest_a_run) for row in candidates)
        ),
        longest_b_run=_summary(
            tuple(float(row.metrics.longest_b_run) for row in candidates)
        ),
        longest_run=_summary(
            tuple(float(row.metrics.longest_run) for row in candidates)
        ),
        predicted_probability=_summary(
            tuple(
                max(row.probability_a, row.probability_b)
                for row in candidates
                if row.probability_a is not None and row.probability_b is not None
            )
        ),
        predictive_entropy=_summary(
            tuple(
                row.predictive_entropy_bits
                for row in candidates
                if row.predictive_entropy_bits is not None
            )
        ),
        effective_depth=_summary(
            tuple(
                float(row.effective_context_depth)
                for row in candidates
                if row.effective_context_depth is not None
            )
        ),
        context_support=_summary(
            tuple(
                float(row.support_count)
                for row in candidates
                if row.support_count is not None
            )
        ),
        target_surprisal=_summary(
            tuple(
                row.target_surprisal_bits
                for row in candidates
                if row.target_surprisal_bits is not None
            )
        ),
        imbalance_flags=tuple(flags),
    )


def _balanced_counts(
    predicting_a: int,
    predicting_b: int,
    ties: int,
) -> tuple[int, int, int]:
    decisive = predicting_a + predicting_b
    total = decisive + ties
    options: list[tuple[int, int, int, int]] = []
    for expected in {decisive // 2, (decisive + 1) // 2}:
        minimum_a = max(0, expected - predicting_b)
        maximum_a = min(predicting_a, expected)
        for expected_a in range(minimum_a, maximum_a + 1):
            expected_b = expected - expected_a
            decisive_target_a = expected_a + predicting_b - expected_b
            desired_tie_a = round(total / 2) - decisive_target_a
            tie_a = min(ties, max(0, desired_tie_a))
            score = abs(2 * expected - decisive) + abs(
                2 * (decisive_target_a + tie_a) - total
            )
            options.append((score, expected_a, expected_b, tie_a))
    _, expected_a, expected_b, tie_a = min(options)
    return expected_a, expected_b, tie_a


def _summary(values: tuple[float, ...]) -> MetricSummary | None:
    if not values:
        return None
    return MetricSummary(min(values), fmean(values), max(values))


class TargetAssignmentMode(StrEnum):
    """Explicit experimental outcome policies, separate from context fitting."""

    BALANCED = "Balanced expected / unexpected"
    EXPECTED = "Expected outcomes"
    UNEXPECTED = "Unexpected outcomes"
    A = "Always A"
    B = "Always B"


def assign_outcomes(
    candidates: Sequence[StimulusCandidate],
    mode: TargetAssignmentMode,
    seed: int,
) -> tuple[StimulusCandidate, ...]:
    """Assign outcomes using existing probabilities and symbol surprisals.

    Expected/unexpected policies leave ties unassigned because no unique
    prediction exists. Explicit A/B policies retain the tie congruency label.
    """
    if mode is TargetAssignmentMode.BALANCED:
        return assign_targets(candidates, seed)
    assigned: list[StimulusCandidate] = []
    for candidate in candidates:
        target: ObservableIndex | None = None
        predicted = candidate.predicted_target_index
        if mode is TargetAssignmentMode.A:
            target = 0
        elif mode is TargetAssignmentMode.B:
            target = 1
        elif predicted is not None:
            target = (
                predicted
                if mode is TargetAssignmentMode.EXPECTED
                else (1 if predicted == 0 else 0)
            )
        if candidate.probability_a is None or candidate.probability_b is None:
            target = None
        congruency = (
            None
            if target is None
            else TargetCongruency.TIE
            if predicted is None
            else TargetCongruency.EXPECTED
            if target == predicted
            else TargetCongruency.UNEXPECTED
        )
        assigned.append(
            replace(
                candidate,
                target_index=target,
                target_probability=None
                if target is None
                else (
                    candidate.probability_a if target == 0 else candidate.probability_b
                ),
                target_surprisal_bits=None
                if target is None
                else (
                    candidate.surprisal_a_bits
                    if target == 0
                    else candidate.surprisal_b_bits
                ),
                target_congruency=congruency,
                condition=None if congruency is None else congruency.value,
            )
        )
    return tuple(assigned)
