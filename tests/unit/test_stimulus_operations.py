from datetime import UTC, datetime

from bspe import (
    BinaryLabels,
    KTSmoothing,
    SequenceDataset,
    SequenceRecord,
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
)
from bspe.stimulus_matching import (
    create_complement_candidates,
    match_stimuli,
)
from bspe.stimulus_search import search_stimuli
from bspe.stimulus_search_results import MatchTolerances, TargetCongruency
from bspe.stimulus_search_types import StimulusSearchConfig
from bspe.stimulus_targets import assign_targets, validate_stimuli


def _search_config() -> StimulusSearchConfig:
    return StimulusSearchConfig(
        sequence_length=8,
        desired_stimuli=8,
        seed=17,
        candidate_limit=32,
        vmm_config=VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )


def test_complements_when_created_are_independently_analyzed() -> None:
    # Given
    config = _search_config()
    selected = search_stimuli(
        config, timestamp=datetime(2026, 9, 8, tzinfo=UTC)
    ).selected

    # When
    complements = create_complement_candidates(selected, config)

    # Then
    assert tuple(candidate.sequence for candidate in complements) == tuple(
        tuple(1 - symbol for symbol in original.sequence) for original in selected
    )
    assert all(row.depth_rows for row in complements)
    labels = BinaryLabels(states=("internal-0", "internal-1"), observables=("A", "B"))
    for complement in complements:
        direct = analyze_dataset(
            SequenceDataset(
                labels,
                (SequenceRecord(str(complement.stimulus_id), complement.sequence),),
            ),
            VMMAnalysisRequest(config.vmm_config, VMMResultScope.PER_SEQUENCE),
        ).records[0]
        assert complement.probability_a == direct.probability_a
        assert complement.probability_b == direct.probability_b
        assert complement.predicted_target_index == direct.predicted_target_index
        assert complement.predictive_entropy_bits == direct.predictive_entropy_bits
        assert complement.effective_context_depth == direct.effective_context_depth
        assert complement.context_used == direct.context_used
        assert complement.support_count == direct.support_count
        assert complement.surprisal_a_bits == direct.surprisal_a_bits
        assert complement.surprisal_b_bits == direct.surprisal_b_bits
        assert complement.depth_rows == direct.depth_rows


def test_matching_when_repeated_is_deterministic_and_never_fabricates_pairs() -> None:
    # Given
    config = _search_config()
    originals = search_stimuli(
        config, timestamp=datetime(2026, 9, 8, tzinfo=UTC)
    ).selected
    candidates = originals + create_complement_candidates(originals, config)
    tolerances = MatchTolerances(
        entropy=0.2,
        prediction_strength=0.2,
        a_proportion=1.0,
        switch_rate=0.2,
        run_structure=3.0,
        effective_depth=3.0,
    )

    # When
    first = match_stimuli(candidates, tolerances)
    second = match_stimuli(candidates, tolerances)

    # Then
    assert first == second
    paired_ids = tuple(
        stimulus.stimulus_id
        for pair in first.pairs
        for stimulus in (pair.first, pair.second)
    )
    assert len(set(paired_ids)) == len(paired_ids)
    assert len(paired_ids) + len(first.unmatched) == len(candidates)


def test_target_assignment_when_prediction_available_is_balanced_and_copied() -> None:
    # Given
    selected = search_stimuli(
        _search_config(), timestamp=datetime(2026, 9, 8, tzinfo=UTC)
    ).selected

    # When
    assigned = assign_targets(selected, seed=5)

    # Then
    assert assigned == assign_targets(selected, seed=5)
    assert (
        abs(
            sum(row.target_index == 0 for row in assigned)
            - sum(row.target_index == 1 for row in assigned)
        )
        <= 1
    )
    expected = sum(
        row.target_congruency is TargetCongruency.EXPECTED for row in assigned
    )
    unexpected = sum(
        row.target_congruency is TargetCongruency.UNEXPECTED for row in assigned
    )
    assert abs(expected - unexpected) <= 1
    for row in assigned:
        if row.target_index == 0:
            assert row.target_probability == row.probability_a
            assert row.target_surprisal_bits == row.surprisal_a_bits
        elif row.target_index == 1:
            assert row.target_probability == row.probability_b
            assert row.target_surprisal_bits == row.surprisal_b_bits


def test_validation_when_targets_are_assigned_reports_descriptive_qc() -> None:
    # Given
    selected = search_stimuli(
        _search_config(), timestamp=datetime(2026, 9, 8, tzinfo=UTC)
    ).selected
    assigned = assign_targets(selected, seed=5)

    # When
    report = validate_stimuli(assigned)

    # Then
    assert report.stimulus_count == len(assigned)
    assert report.a_proportion is not None
    assert report.switches is not None
    assert report.longest_run is not None
    assert report.predictive_entropy is not None
    assert report.unassigned_target_count == 0
