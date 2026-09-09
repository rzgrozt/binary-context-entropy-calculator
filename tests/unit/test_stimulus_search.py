from datetime import UTC, datetime

import pytest

from binary_entropy import (
    BinaryLabels,
    KTSmoothing,
    SequenceDataset,
    SequenceRecord,
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
)
from binary_entropy.stimulus_search import search_stimuli
from binary_entropy.stimulus_search_results import SearchPartialReason, SearchStatus
from binary_entropy.stimulus_search_types import (
    InclusiveRange,
    PredictedSymbol,
    PreferenceMetric,
    SoftPreference,
    StimulusConstraints,
    StimulusSearchConfig,
)

TIMESTAMP = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def _config(
    *,
    desired: int = 5,
    limit: int = 24,
    constraints: StimulusConstraints | None = None,
) -> StimulusSearchConfig:
    return StimulusSearchConfig(
        sequence_length=8,
        desired_stimuli=desired,
        seed=91,
        candidate_limit=limit,
        vmm_config=VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
        constraints=constraints or StimulusConstraints(),
        preferences=(SoftPreference(PreferenceMetric.PREDICTIVE_ENTROPY, 0.5, 2.0),),
    )


def test_search_when_seed_and_timestamp_repeat_returns_identical_snapshot() -> None:
    # Given
    config = _config()

    # When
    first = search_stimuli(config, timestamp=TIMESTAMP)
    second = search_stimuli(config, timestamp=TIMESTAMP)

    # Then
    assert first == second
    assert first.timestamp_utc == "2026-09-08T12:00:00+00:00"
    assert first.package_version == "0.1.0"


def test_search_when_desired_is_small_still_evaluates_complete_sample() -> None:
    # Given
    config = _config(desired=1, limit=19)

    # When
    result = search_stimuli(config, timestamp=TIMESTAMP)

    # Then
    assert result.evaluated_count == 19
    assert result.accepted_count == 19
    assert result.selected_count == 1
    assert result.status is SearchStatus.COMPLETE


def test_search_when_hard_filters_are_set_accepts_only_satisfying_candidates() -> None:
    # Given
    constraints = StimulusConstraints(
        predicted_symbol=PredictedSymbol.A,
        predicted_probability=InclusiveRange(0.5, 1.0),
        predictive_entropy=InclusiveRange(0.0, 1.0),
        effective_depth=InclusiveRange(0, 8),
        context_support=InclusiveRange(1, 8),
        a_count=InclusiveRange(2, 6),
        a_proportion=InclusiveRange(0.25, 0.75),
        switches=InclusiveRange(2, 7),
        switch_rate=InclusiveRange(2 / 7, 1.0),
        longest_a_run=InclusiveRange(1, 5),
        longest_b_run=InclusiveRange(1, 5),
        longest_run=InclusiveRange(1, 5),
    )

    # When
    result = search_stimuli(
        _config(limit=128, constraints=constraints), timestamp=TIMESTAMP
    )

    # Then
    assert result.accepted
    assert constraints.predicted_probability is not None
    assert constraints.a_count is not None
    assert constraints.switches is not None
    assert constraints.longest_run is not None
    for candidate in result.accepted:
        assert candidate.predicted_target_index == 0
        assert candidate.probability_a is not None
        assert constraints.predicted_probability.contains(candidate.probability_a)
        assert constraints.a_count.contains(candidate.metrics.a_count)
        assert constraints.switches.contains(candidate.metrics.switches)
        assert constraints.longest_run.contains(candidate.metrics.longest_run)


def test_search_when_constraints_are_impossible_returns_partial_with_frequencies() -> (
    None
):
    # Given
    constraints = StimulusConstraints(a_count=InclusiveRange(9, 9))

    # When
    result = search_stimuli(
        _config(desired=3, limit=10, constraints=constraints), timestamp=TIMESTAMP
    )

    # Then
    assert result.status is SearchStatus.PARTIAL
    assert result.partial_reason is SearchPartialReason.CANDIDATE_LIMIT
    assert result.selected_count == 0
    assert result.violations[0].constraint == "a_count"
    assert result.violations[0].frequency == 1.0


def test_search_when_analyzed_copies_vmm_values_and_preserves_boundaries() -> None:
    # Given
    config = _config(desired=3, limit=3)

    # When
    result = search_stimuli(config, timestamp=TIMESTAMP)

    # Then
    labels = BinaryLabels(states=("internal-0", "internal-1"), observables=("A", "B"))
    for candidate in result.selected:
        direct = analyze_dataset(
            SequenceDataset(
                labels,
                (SequenceRecord(str(candidate.stimulus_id), candidate.sequence),),
            ),
            VMMAnalysisRequest(config.vmm_config, VMMResultScope.PER_SEQUENCE),
        ).records[0]
        assert candidate.probability_a == direct.probability_a
        assert candidate.probability_b == direct.probability_b
        assert candidate.predictive_entropy_bits == direct.predictive_entropy_bits
        assert candidate.depth_rows == direct.depth_rows
        assert direct.model.source_sequence_count == 1
        if candidate.probability_a is not None and candidate.probability_b is not None:
            assert candidate.probability_a + candidate.probability_b == pytest.approx(
                1.0
            )


def test_search_when_limit_is_reached_reports_candidate_limit_partial() -> None:
    # Given
    constraints = StimulusConstraints(a_count=InclusiveRange(9, 9))

    # When
    result = search_stimuli(
        _config(desired=5, limit=5, constraints=constraints), timestamp=TIMESTAMP
    )

    # Then
    assert result.status is SearchStatus.PARTIAL
    assert result.partial_reason is SearchPartialReason.CANDIDATE_LIMIT


def test_search_when_entire_universe_is_exhausted_reports_universe_partial() -> None:
    # Given
    config = StimulusSearchConfig(
        sequence_length=2,
        desired_stimuli=5,
        seed=1,
        candidate_limit=5,
        vmm_config=VMMConfig(),
    )

    # When
    result = search_stimuli(config, timestamp=TIMESTAMP)

    # Then
    assert result.evaluated_count == 4
    assert result.status is SearchStatus.PARTIAL
    assert result.partial_reason is SearchPartialReason.UNIVERSE_EXHAUSTED
