"""Information-theoretic primitives."""

import math

from binary_entropy.constants import PROBABILITY_TOLERANCE
from binary_entropy.errors import NumericalInvariantError, ProbabilityRangeError


def binary_entropy(probability: float) -> float:
    """Return binary Shannon entropy in bits."""
    _validate_probability(probability)
    if probability in {0.0, 1.0}:
        return 0.0
    complement = 1.0 - probability
    result = -(
        probability * math.log2(probability) + complement * math.log2(complement)
    )
    if result < -PROBABILITY_TOLERANCE or result > 1.0 + PROBABILITY_TOLERANCE:
        raise NumericalInvariantError(quantity="binary entropy", value=result)
    return min(1.0, max(0.0, result))


def surprisal(probability: float) -> float:
    """Return self-information in bits."""
    _validate_probability(probability)
    if probability == 0.0:
        return math.inf
    return -math.log2(probability)


def _validate_probability(probability: float) -> None:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ProbabilityRangeError(
            field="probability",
            index=(),
            value=probability,
        )
