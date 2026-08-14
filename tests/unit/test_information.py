import math

import pytest

from binary_entropy.errors import ProbabilityRangeError
from binary_entropy.information import binary_entropy, surprisal


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 1.0),
        (0.25, 0.8112781244591328),
    ],
)
def test_binary_entropy_when_probability_is_valid(
    probability: float,
    expected: float,
) -> None:
    result = binary_entropy(probability)

    assert result == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("probability", [0.01, 0.1, 0.33, 0.9])
def test_binary_entropy_when_probability_is_reflected(
    probability: float,
) -> None:
    forward = binary_entropy(probability)
    reflected = binary_entropy(1.0 - probability)

    assert forward == pytest.approx(reflected, abs=1e-15)


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_binary_entropy_when_probability_is_in_unit_interval(
    probability: float,
) -> None:
    result = binary_entropy(probability)

    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("probability", [-0.01, 1.01, math.nan, math.inf])
def test_information_functions_when_probability_is_invalid(
    probability: float,
) -> None:
    with pytest.raises(ProbabilityRangeError):
        _ = binary_entropy(probability)

    with pytest.raises(ProbabilityRangeError):
        _ = surprisal(probability)


def test_surprisal_when_probability_is_zero() -> None:
    result = surprisal(0.0)

    assert result == math.inf


def test_surprisal_when_probability_is_positive() -> None:
    result = surprisal(0.25)

    assert result == 2.0
