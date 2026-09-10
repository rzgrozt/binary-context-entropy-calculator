"""Independent complement analysis and tolerance matching."""

from collections.abc import Sequence
from dataclasses import replace

from bspe.records import SequenceRecord
from bspe.stimulus_analysis import analyze_stimulus_records
from bspe.stimulus_search_results import (
    MatchingResult,
    MatchTolerances,
    StimulusCandidate,
    StimulusPair,
)
from bspe.stimulus_search_types import StimulusSearchConfig


def create_complement_candidates(
    candidates: Sequence[StimulusCandidate],
    config: StimulusSearchConfig,
) -> tuple[StimulusCandidate, ...]:
    """Analyze explicit complements as independent records."""
    records = tuple(
        SequenceRecord(
            f"{candidate.stimulus_id}-complement",
            tuple(1 if symbol == 0 else 0 for symbol in candidate.sequence),
        )
        for candidate in candidates
    )
    if not records:
        return ()
    return analyze_stimulus_records(records, config)


def match_stimuli(
    candidates: Sequence[StimulusCandidate],
    tolerances: MatchTolerances,
) -> MatchingResult:
    """Greedily match opposite predictions within all tolerances."""
    predicting_a = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.predicted_target_index == 0
        ),
        key=lambda candidate: candidate.stimulus_id,
    )
    predicting_b = {
        candidate.stimulus_id: candidate
        for candidate in candidates
        if candidate.predicted_target_index == 1
    }
    pairs: list[StimulusPair] = []
    paired_ids: set[str] = set()
    for first in predicting_a:
        options = tuple(
            (second, _distance(first, second, tolerances))
            for second in predicting_b.values()
            if second.stimulus_id not in paired_ids
        )
        qualified = tuple(
            (second, distance) for second, distance in options if distance is not None
        )
        if not qualified:
            continue
        second, distance = min(
            qualified,
            key=lambda item: (
                not _are_explicit_complements(first, item[0]),
                item[1],
                item[0].stimulus_id,
            ),
        )
        pair_id = f"pair-{len(pairs) + 1:06d}"
        paired_first = replace(first, pair_id=pair_id, group_id=pair_id)
        paired_second = replace(second, pair_id=pair_id, group_id=pair_id)
        pairs.append(StimulusPair(pair_id, paired_first, paired_second, distance))
        paired_ids.update((str(first.stimulus_id), str(second.stimulus_id)))
    unmatched = tuple(
        candidate
        for candidate in candidates
        if str(candidate.stimulus_id) not in paired_ids
    )
    return MatchingResult(tuple(pairs), unmatched)


def _distance(
    first: StimulusCandidate,
    second: StimulusCandidate,
    tolerances: MatchTolerances,
) -> float | None:
    if (
        first.metrics.length != second.metrics.length
        or first.predictive_entropy_bits is None
        or second.predictive_entropy_bits is None
        or first.effective_context_depth is None
        or second.effective_context_depth is None
        or first.probability_a is None
        or first.probability_b is None
        or second.probability_a is None
        or second.probability_b is None
    ):
        return None
    differences = (
        abs(first.predictive_entropy_bits - second.predictive_entropy_bits),
        abs(
            max(first.probability_a, first.probability_b)
            - max(second.probability_a, second.probability_b)
        ),
        abs(first.metrics.a_proportion - second.metrics.b_proportion),
        abs(first.metrics.switch_rate - second.metrics.switch_rate),
        max(
            abs(first.metrics.longest_a_run - second.metrics.longest_b_run),
            abs(first.metrics.longest_b_run - second.metrics.longest_a_run),
        ),
        abs(first.effective_context_depth - second.effective_context_depth),
    )
    limits = (
        tolerances.entropy,
        tolerances.prediction_strength,
        tolerances.a_proportion,
        tolerances.switch_rate,
        tolerances.run_structure,
        tolerances.effective_depth,
    )
    if any(
        difference > limit
        for difference, limit in zip(differences, limits, strict=True)
    ):
        return None
    return sum(differences)


def _are_explicit_complements(
    first: StimulusCandidate,
    second: StimulusCandidate,
) -> bool:
    return (
        f"{first.stimulus_id}-complement" == second.stimulus_id
        or f"{second.stimulus_id}-complement" == first.stimulus_id
    )
