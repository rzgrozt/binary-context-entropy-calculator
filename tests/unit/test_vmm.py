import math

import pytest

from binary_entropy.domain import BinaryLabels, TargetClassification
from binary_entropy.information import binary_entropy, surprisal
from binary_entropy.methods.vmm import (
    analyze_vmm,
    analyze_vmm_per_sequence,
    fit_vmm,
)
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    InvalidVMMConfigurationError,
    KTSmoothing,
    MLESmoothing,
    VMMConfig,
    VMMDepthStatus,
    VMMResultScope,
)


def _dataset(records: tuple[SequenceRecord, ...]) -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(labels, records)


def test_vmm_when_recurrent_aa_context_is_supported_uses_depth_two() -> None:
    # Given
    record = SequenceRecord(
        "recurrent-aa",
        (0, 0, 1, 0, 0, 1, 0, 0),
        actual_target_index=1,
    )
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=2)

    # When
    analysis = analyze_vmm(_dataset((record,)), config)
    result = analysis.records[0]

    # Then
    assert analysis.result_scope is VMMResultScope.POOLED
    assert result.sequence_id == "recurrent-aa"
    assert result.sequence == record.sequence
    assert result.effective_context_depth == 2
    assert result.context_used == (0, 0)
    assert result.support_count == 2
    assert result.probability_a == 0.5 / 3
    assert result.probability_b == 2.5 / 3
    assert result.predicted_target_index == 1
    assert result.predictive_entropy_bits == pytest.approx(
        binary_entropy(0.5 / 3),
        abs=1e-15,
    )
    assert result.surprisal_a_bits == pytest.approx(surprisal(0.5 / 3), abs=1e-15)
    assert result.surprisal_b_bits == pytest.approx(surprisal(2.5 / 3), abs=1e-15)
    assert result.target_assessment is not None
    assert result.target_assessment.classification is TargetClassification.MODAL
    assert result.target_assessment.probability == 2.5 / 3
    assert tuple(row.depth for row in result.depth_rows) == tuple(range(9))
    assert result.depth_rows[0].matched_suffix == ()
    assert (
        result.depth_rows[1].count_a,
        result.depth_rows[1].count_b,
        result.depth_rows[1].support,
        result.depth_rows[1].probability_b,
    ) == (3, 2, 5, 2.5 / 6)
    assert result.depth_rows[1].status is VMMDepthStatus.ACCEPTED
    assert result.depth_rows[2].status is VMMDepthStatus.ACCEPTED


def test_vmm_when_deepest_suffix_has_low_support_backs_off() -> None:
    # Given
    record = SequenceRecord("backoff", (0, 0, 1, 0, 0))
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=2)

    # When
    result = analyze_vmm(_dataset((record,)), config).records[0]

    # Then
    assert result.effective_context_depth == 1
    assert result.context_used == (0,)
    assert result.depth_rows[1].status is VMMDepthStatus.ACCEPTED
    assert (
        result.depth_rows[2].matched_suffix,
        result.depth_rows[2].support,
        result.depth_rows[2].count_a,
        result.depth_rows[2].count_b,
    ) == ((0, 0), 1, 0, 1)
    assert result.depth_rows[2].probability_a == 0.25
    assert result.depth_rows[2].probability_b == 0.75
    assert result.depth_rows[2].status is VMMDepthStatus.LOW_SUPPORT
    assert result.depth_rows[3].status is VMMDepthStatus.UNAVAILABLE
    assert result.depth_rows[3].probability_a is None


def test_vmm_when_pooling_records_preserves_sequence_boundaries() -> None:
    # Given
    dataset = _dataset(
        (
            SequenceRecord("aa", (0, 0)),
            SequenceRecord("ba", (1, 0)),
        )
    )

    # When
    model = fit_vmm(dataset)
    analysis = analyze_vmm(
        dataset,
        VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )

    # Then
    assert tuple(
        (row.context, row.count_a, row.count_b) for row in model.context_counts
    ) == (
        ((), 3, 1),
        ((0,), 1, 0),
        ((1,), 1, 0),
    )
    assert len(analysis.records) == 2
    assert all(record.model == model for record in analysis.records)
    assert all(record.effective_context_depth == 1 for record in analysis.records)


def test_vmm_when_scope_is_per_sequence_uses_only_each_records_counts() -> None:
    # Given
    dataset = _dataset(
        (
            SequenceRecord("aa", (0, 0)),
            SequenceRecord("ba", (1, 0)),
        )
    )
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=1)

    # When
    analysis = analyze_vmm_per_sequence(dataset, config)

    # Then
    first, second = analysis.records
    assert analysis.result_scope is VMMResultScope.PER_SEQUENCE
    assert first.effective_context_depth == 1
    assert first.context_used == (0,)
    assert second.effective_context_depth == 0
    assert second.context_used == ()
    assert tuple(
        (row.context, row.count_a, row.count_b) for row in second.model.context_counts
    ) == (((), 1, 1), ((1,), 1, 0))


def test_vmm_when_record_is_empty_returns_explicit_unavailable_prediction() -> None:
    # Given
    dataset = _dataset((SequenceRecord("empty", ()),))
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=1)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    assert len(result.depth_rows) == 1
    assert result.depth_rows[0].status is VMMDepthStatus.UNAVAILABLE
    assert result.effective_context_depth is None
    assert result.context_used is None
    assert result.support_count is None
    assert result.probability_a is None
    assert result.probability_b is None
    assert result.predicted_target_index is None
    assert result.predictive_entropy_bits is None
    assert result.surprisal_a_bits is None
    assert result.surprisal_b_bits is None


def test_vmm_when_minimum_support_is_unmet_returns_unavailable_final() -> None:
    # Given
    dataset = _dataset((SequenceRecord("single", (0,)),))
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=2)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    assert result.depth_rows[0].status is VMMDepthStatus.LOW_SUPPORT
    assert result.depth_rows[0].support == 1
    assert result.depth_rows[0].probability_a == 0.75
    assert result.depth_rows[0].probability_b == 0.25
    assert result.depth_rows[1].status is VMMDepthStatus.UNAVAILABLE
    assert result.effective_context_depth is None
    assert result.context_used is None
    assert result.support_count is None
    assert result.probability_a is None


def test_vmm_when_custom_additive_smoothing_is_selected_uses_its_alpha() -> None:
    # Given
    dataset = _dataset((SequenceRecord("additive", (0, 0, 0, 1, 0)),))
    config = VMMConfig(smoothing=AdditiveSmoothing(alpha=2.0), minimum_support=2)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    assert result.effective_context_depth == 1
    assert result.support_count == 3
    assert result.probability_a == 4 / 7
    assert result.probability_b == 3 / 7
    assert result.predictive_entropy_bits == pytest.approx(
        binary_entropy(4 / 7),
        abs=1e-15,
    )


def test_mle_smoothing_when_constructed_exposes_zero_alpha() -> None:
    # Given / When
    smoothing = MLESmoothing()

    # Then
    assert smoothing.alpha == 0.0


def test_vmm_when_mle_context_is_seen_uses_raw_count_probabilities() -> None:
    # Given
    dataset = _dataset((SequenceRecord("mle-seen", (0, 0, 1, 0, 0, 1, 0, 0)),))
    config = VMMConfig(smoothing=MLESmoothing(), minimum_support=2)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    selected = result.depth_rows[2]
    assert result.effective_context_depth == 2
    assert (selected.count_a, selected.count_b, selected.support) == (0, 2, 2)
    assert selected.probability_a == 0.0
    assert selected.probability_b == 1.0
    assert selected.predictive_entropy_bits == 0.0


def test_vmm_when_mle_context_is_unseen_keeps_evidence_unavailable() -> None:
    # Given
    dataset = _dataset((SequenceRecord("mle-unseen", (0, 1, 1)),))
    config = VMMConfig(smoothing=MLESmoothing(), minimum_support=1)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    unseen = result.depth_rows[2]
    assert unseen.matched_suffix == (1, 1)
    assert unseen.status is VMMDepthStatus.UNAVAILABLE
    assert (unseen.count_a, unseen.count_b, unseen.support) == (0, 0, 0)
    assert unseen.probability_a is None
    assert unseen.probability_b is None
    assert unseen.predictive_entropy_bits is None


def test_vmm_when_mle_deepest_context_has_low_support_backs_off() -> None:
    # Given
    dataset = _dataset((SequenceRecord("mle-backoff", (0, 0, 1, 0, 0)),))
    config = VMMConfig(smoothing=MLESmoothing(), minimum_support=2)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    assert result.effective_context_depth == 1
    assert result.context_used == (0,)
    assert result.probability_a == 2 / 3
    assert result.probability_b == 1 / 3
    assert result.depth_rows[2].status is VMMDepthStatus.LOW_SUPPORT
    assert result.depth_rows[2].probability_a == 0.0
    assert result.depth_rows[2].probability_b == 1.0


def test_vmm_when_selected_distribution_is_tied_has_no_predicted_target() -> None:
    # Given
    dataset = _dataset((SequenceRecord("tie", (0, 1)),))
    config = VMMConfig(smoothing=KTSmoothing(), minimum_support=2)

    # When
    result = analyze_vmm_per_sequence(dataset, config).records[0]

    # Then
    assert result.effective_context_depth == 0
    assert result.probability_a == 0.5
    assert result.probability_b == 0.5
    assert result.predicted_target_index is None


@pytest.mark.parametrize("alpha", [0.0, -0.5, math.inf, math.nan])
def test_additive_smoothing_when_alpha_is_not_positive_finite_rejects_it(
    alpha: float,
) -> None:
    # Given / When / Then
    with pytest.raises(InvalidVMMConfigurationError):
        _ = AdditiveSmoothing(alpha=alpha)


@pytest.mark.parametrize("minimum_support", [0, -1])
def test_vmm_config_when_minimum_support_is_not_positive_rejects_it(
    minimum_support: int,
) -> None:
    # Given / When / Then
    with pytest.raises(InvalidVMMConfigurationError):
        _ = VMMConfig(smoothing=KTSmoothing(), minimum_support=minimum_support)
