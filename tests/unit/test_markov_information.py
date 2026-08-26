import numpy as np
import pytest

from binary_entropy.domain import BinaryLabels
from binary_entropy.markov_information import (
    empirical_conditional_entropy,
    entropy_rate,
    stationary_distribution,
)
from binary_entropy.markov_types import (
    StationaryUnavailableReason,
    TransitionMatrix,
    UnavailableStationaryDistribution,
    UniqueStationaryDistribution,
)
from binary_entropy.methods.markov import analyze_markov
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord


def _dataset(sequences: tuple[BinarySequence, ...]) -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    records = tuple(
        SequenceRecord(f"record-{index}", sequence)
        for index, sequence in enumerate(sequences, start=1)
    )
    return SequenceDataset(labels, records)


def _matrix(
    row_a: tuple[float, float] | None,
    row_b: tuple[float, float] | None,
) -> TransitionMatrix:
    first = None if row_a is None else np.array(row_a, dtype=np.float64)
    second = None if row_b is None else np.array(row_b, dtype=np.float64)
    return first, second


def test_empirical_conditional_entropy_when_counts_exist_weights_raw_rows() -> None:
    # Given
    counts = ((1, 2), (1, 1))
    expected = (3 / 5) * 0.9182958340544896 + (2 / 5)

    # When
    result = empirical_conditional_entropy(counts)

    # Then
    assert result == pytest.approx(expected, abs=1e-15)


def test_empirical_conditional_entropy_when_no_transitions_is_unavailable() -> None:
    # Given
    counts = ((0, 0), (0, 0))

    # When
    result = empirical_conditional_entropy(counts)

    # Then
    assert result is None


def test_stationary_distribution_when_matrix_is_complete_and_unique_is_exact() -> None:
    # Given
    matrix = _matrix((2 / 3, 1 / 3), (1 / 2, 1 / 2))

    # When
    result = stationary_distribution(matrix)

    # Then
    match result:
        case UniqueStationaryDistribution(distribution=distribution):
            np.testing.assert_allclose(distribution, [0.6, 0.4], atol=1e-15, rtol=0.0)
            assert not distribution.flags.writeable
        case UnavailableStationaryDistribution():
            pytest.fail("unique matrix was reported unavailable")


def test_stationary_distribution_when_matrix_is_identity_reports_non_unique() -> None:
    # Given
    matrix = _matrix((1.0, 0.0), (0.0, 1.0))

    # When
    result = stationary_distribution(matrix)

    # Then
    match result:
        case UnavailableStationaryDistribution(reason=reason):
            assert reason is StationaryUnavailableReason.NON_UNIQUE
        case UniqueStationaryDistribution():
            pytest.fail("identity matrix was reported unique")


def test_stationary_distribution_when_matrix_is_periodic_two_cycle_is_unique() -> None:
    # Given
    matrix = _matrix((0.0, 1.0), (1.0, 0.0))

    # When
    stationary = stationary_distribution(matrix)
    result = entropy_rate(matrix, stationary)

    # Then
    match stationary:
        case UniqueStationaryDistribution(distribution=distribution):
            np.testing.assert_array_equal(distribution, [0.5, 0.5])
        case UnavailableStationaryDistribution():
            pytest.fail("periodic two-cycle was reported unavailable")
    assert result == 0.0


def test_stationary_distribution_when_matrix_has_missing_row_reports_incomplete() -> (
    None
):
    # Given
    matrix = _matrix((0.5, 0.5), None)

    # When
    result = stationary_distribution(matrix)

    # Then
    match result:
        case UnavailableStationaryDistribution(reason=reason):
            assert reason is StationaryUnavailableReason.INCOMPLETE_MATRIX
        case UniqueStationaryDistribution():
            pytest.fail("incomplete matrix was reported unique")


def test_entropy_rate_when_stationary_is_unique_weights_row_entropies() -> None:
    # Given
    matrix = _matrix((2 / 3, 1 / 3), (1 / 2, 1 / 2))
    stationary = stationary_distribution(matrix)
    expected = 0.6 * 0.9182958340544896 + 0.4

    # When
    result = entropy_rate(matrix, stationary)

    # Then
    assert result == pytest.approx(expected, abs=1e-15)


def test_analyze_markov_when_smoothing_is_selected_keeps_empirical_entropy_raw() -> (
    None
):
    # Given
    dataset: SequenceDataset = _dataset(((0, 0, 1, 0, 1, 1),))
    expected = (3 / 5) * 0.9182958340544896 + (2 / 5)

    # When
    result = analyze_markov(dataset, smoothing_alpha=1.0)

    # Then
    assert result.empirical_conditional_entropy_bits == expected
