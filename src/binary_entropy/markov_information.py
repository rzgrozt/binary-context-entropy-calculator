"""Information measures for first-order binary Markov models."""

import numpy as np

from binary_entropy.domain import FloatArray, float_values
from binary_entropy.information import binary_entropy
from binary_entropy.markov_types import (
    StationaryDistributionResult,
    StationaryUnavailableReason,
    TransitionCounts,
    TransitionMatrix,
    UnavailableStationaryDistribution,
    UniqueStationaryDistribution,
)


def empirical_conditional_entropy(counts: TransitionCounts) -> float | None:
    """Measure raw empirical H(X[t+1] | X[t]) without smoothing."""
    transition_total = sum(sum(row) for row in counts)
    if transition_total == 0:
        return None
    result = 0.0
    for row in counts:
        row_total = row[0] + row[1]
        if row_total > 0:
            result += (row_total / transition_total) * binary_entropy(
                row[0] / row_total
            )
    return result


def stationary_distribution(
    matrix: TransitionMatrix,
) -> StationaryDistributionResult:
    """Return the unique binary stationary distribution when identifiable."""
    row_a, row_b = matrix
    if row_a is None or row_b is None:
        return UnavailableStationaryDistribution(
            StationaryUnavailableReason.INCOMPLETE_MATRIX
        )
    _, a_to_b = float_values(row_a)
    b_to_a, _ = float_values(row_b)
    cross_transition_total = a_to_b + b_to_a
    if cross_transition_total == 0.0:
        return UnavailableStationaryDistribution(StationaryUnavailableReason.NON_UNIQUE)
    distribution: FloatArray = np.array(
        [
            b_to_a / cross_transition_total,
            a_to_b / cross_transition_total,
        ],
        dtype=np.float64,
    )
    return UniqueStationaryDistribution(distribution)


def entropy_rate(
    matrix: TransitionMatrix,
    stationary: StationaryDistributionResult,
) -> float | None:
    """Return stationary-weighted transition entropy when uniquely defined."""
    row_a, row_b = matrix
    if row_a is None or row_b is None:
        return None
    match stationary:
        case UniqueStationaryDistribution(distribution=distribution):
            stationary_a, stationary_b = float_values(distribution)
            probability_aa, _ = float_values(row_a)
            probability_ba, _ = float_values(row_b)
            return stationary_a * binary_entropy(
                probability_aa
            ) + stationary_b * binary_entropy(probability_ba)
        case UnavailableStationaryDistribution():
            return None
