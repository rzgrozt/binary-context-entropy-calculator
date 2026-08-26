import math

import numpy as np
import pytest

from binary_entropy.domain import BinaryLabels, ObservableIndex
from binary_entropy.errors import InvalidSmoothingAlphaError
from binary_entropy.markov_types import MarkovEstimation, MarkovPredictionMode
from binary_entropy.methods.markov import analyze_markov, fit_markov, predict_markov
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord


def _dataset(
    sequences: tuple[BinarySequence, ...],
    targets: tuple[ObservableIndex | None, ...] | None = None,
) -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    selected_targets = targets or tuple(None for _ in sequences)
    records = tuple(
        SequenceRecord(f"record-{index}", sequence, target)
        for index, (sequence, target) in enumerate(
            zip(sequences, selected_targets, strict=True),
            start=1,
        )
    )
    return SequenceDataset(labels, records)


def test_fit_markov_when_sequence_is_aababb_counts_and_estimates_exactly() -> None:
    # Given
    dataset = _dataset(((0, 0, 1, 0, 1, 1),))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.transition_counts == ((1, 2), (1, 1))
    assert result.source_transition_count == 5
    row_a, row_b = result.transition_matrix
    assert row_a is not None
    assert row_b is not None
    np.testing.assert_allclose(row_a, [1 / 3, 2 / 3], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(row_b, [1 / 2, 1 / 2], atol=0.0, rtol=0.0)


def test_fit_markov_when_records_are_aab_and_ba_does_not_count_boundary() -> None:
    # Given
    dataset = _dataset(((0, 0, 1), (1, 0)))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.transition_counts == ((1, 1), (1, 0))
    assert result.source_transition_count == 3


def test_fit_markov_when_mle_row_is_unseen_marks_row_unavailable() -> None:
    # Given
    dataset = _dataset(((0, 0),))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.estimation_method is MarkovEstimation.MAXIMUM_LIKELIHOOD
    assert result.transition_matrix[0] is not None
    assert result.transition_matrix[1] is None


def test_fit_markov_when_alpha_is_positive_applies_additive_smoothing() -> None:
    # Given
    dataset = _dataset(((0, 0, 0),))

    # When
    result = fit_markov(dataset, smoothing_alpha=1.0)

    # Then
    assert result.estimation_method is MarkovEstimation.ADDITIVE_SMOOTHING
    row_a, row_b = result.transition_matrix
    assert row_a is not None
    assert row_b is not None
    np.testing.assert_allclose(row_a, [3 / 4, 1 / 4], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(row_b, [1 / 2, 1 / 2], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("alpha", [-0.1, math.nan, math.inf])
def test_fit_markov_when_alpha_is_outside_domain_rejects_value(alpha: float) -> None:
    # Given
    dataset = _dataset(((0,),))

    # When / Then
    with pytest.raises(InvalidSmoothingAlphaError):
        _ = fit_markov(dataset, smoothing_alpha=alpha)


def test_fit_markov_when_probabilities_are_created_owns_read_only_float64_arrays() -> (
    None
):
    # Given
    dataset = _dataset(((0, 1), (1, 0)))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.starting_distribution is not None
    assert result.starting_distribution.dtype == np.float64
    assert not result.starting_distribution.flags.writeable
    for row in result.transition_matrix:
        assert row is not None
        assert row.dtype == np.float64
        assert not row.flags.writeable


def test_fit_markov_when_records_have_distinct_starts_uses_nonempty_starts_only() -> (
    None
):
    # Given
    dataset = _dataset(((0, 1), (1,), ()))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.starting_distribution is not None
    np.testing.assert_array_equal(result.starting_distribution, [0.5, 0.5])
    assert result.source_sequence_count == 3


def test_fit_markov_when_all_records_are_empty_has_unavailable_start_distribution() -> (
    None
):
    # Given
    dataset = _dataset(((), ()))

    # When
    result = fit_markov(dataset)

    # Then
    assert result.starting_distribution is None


def test_analyze_markov_when_fixed_uses_final_state_only_for_every_prefix() -> None:
    # Given
    dataset = _dataset(((0, 0, 1, 0, 1, 1),))

    # When
    result = analyze_markov(dataset, prediction_mode=MarkovPredictionMode.FIXED_MODEL)

    # Then
    rows = result.records[0].rows
    assert len(rows) == 7
    assert rows[1].context == (0,)
    assert rows[4].context == (0,)
    assert rows[1].predictive is not None
    assert rows[4].predictive is not None
    np.testing.assert_array_equal(rows[1].predictive, rows[4].predictive)
    final_predictive = rows[-1].predictive
    assert final_predictive is not None
    np.testing.assert_allclose(final_predictive, [0.5, 0.5], atol=0.0, rtol=0.0)


def test_predict_markov_when_context_has_history_uses_only_final_state() -> None:
    # Given
    model = fit_markov(_dataset(((0, 0, 1, 0, 1, 1),)))

    # When
    result = predict_markov(model, (1, 1, 0))

    # Then
    assert result is not None
    np.testing.assert_allclose(result, [1 / 3, 2 / 3], atol=0.0, rtol=0.0)


def test_analyze_markov_when_modes_differ_preserves_fixed_vs_cumulative_fit() -> None:
    # Given
    dataset = _dataset(((0, 0, 1, 0, 1, 1),))

    # When
    fixed = analyze_markov(dataset, prediction_mode=MarkovPredictionMode.FIXED_MODEL)
    cumulative = analyze_markov(
        dataset,
        prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
    )

    # Then
    fixed_depth_two = fixed.records[0].rows[2]
    cumulative_depth_two = cumulative.records[0].rows[2]
    assert fixed_depth_two.prediction_mode is MarkovPredictionMode.FIXED_MODEL
    assert (
        cumulative_depth_two.prediction_mode is MarkovPredictionMode.CUMULATIVE_PREFIX
    )
    assert fixed_depth_two.predictive is not None
    assert cumulative_depth_two.predictive is not None
    np.testing.assert_allclose(
        fixed_depth_two.predictive,
        [1 / 3, 2 / 3],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(cumulative_depth_two.predictive, [1.0, 0.0])
    assert cumulative.records[0].rows[1].predictive is None


def test_analyze_markov_when_actual_target_changes_keeps_fit_and_predictions() -> None:
    # Given
    target_a = _dataset(((0, 0, 1, 0, 1, 1),), (0,))
    target_b = _dataset(((0, 0, 1, 0, 1, 1),), (1,))

    # When
    result_a = analyze_markov(target_a)
    result_b = analyze_markov(target_b)

    # Then
    assert result_a.model.transition_counts == result_b.model.transition_counts
    for row_a, row_b in zip(
        result_a.records[0].rows,
        result_b.records[0].rows,
        strict=True,
    ):
        if row_a.predictive is None:
            assert row_b.predictive is None
        else:
            assert row_b.predictive is not None
            np.testing.assert_array_equal(row_a.predictive, row_b.predictive)
    assert result_a.records[0].target_assessment is not None
    assert result_b.records[0].target_assessment is not None
    assert (
        result_a.records[0].target_assessment.actual_target_index
        != result_b.records[0].target_assessment.actual_target_index
    )
