import math

import numpy as np
import pytest

from binary_entropy.domain import BinaryHMM, BinaryLabels
from binary_entropy.errors import (
    DuplicateLabelError,
    InvalidLabelError,
    ProbabilityRangeError,
    ProbabilityShapeError,
    ProbabilitySumError,
)
from tests.unit.helpers import hand_model


def test_labels_when_values_have_surrounding_whitespace() -> None:
    labels = BinaryLabels(
        states=(" latent A ", "latent B"),
        observables=(" left ", "right"),
    )

    assert labels.states == ("latent A", "latent B")
    assert labels.observables == ("left", "right")


@pytest.mark.parametrize("value", ["", "   ", "bad,label", "bad\nlabel"])
def test_labels_when_value_is_invalid(value: str) -> None:
    with pytest.raises(InvalidLabelError):
        _ = BinaryLabels(states=(value, "other"), observables=("A", "B"))


def test_labels_when_trimmed_values_are_duplicates() -> None:
    with pytest.raises(DuplicateLabelError):
        _ = BinaryLabels(states=("same", " same "), observables=("A", "B"))


def test_model_when_values_are_valid_copies_read_only_float64_arrays() -> None:
    initial = np.array([0.5, 0.5], dtype=np.float32)
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])

    model = BinaryHMM(
        labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
        initial=initial,
        transition=transition,
        emission=emission,
    )
    initial[0] = 0.1

    assert model.initial.dtype == np.float64
    assert model.initial.tolist() == [0.5, 0.5]
    assert not model.initial.flags.writeable
    assert not model.transition.flags.writeable
    assert not model.emission.flags.writeable


@pytest.mark.parametrize(
    ("field", "initial", "transition", "emission"),
    [
        ("initial", [1.0], [[0.7, 0.3], [0.2, 0.8]], [[0.9, 0.1], [0.2, 0.8]]),
        ("transition", [0.6, 0.4], [[1.0, 0.0]], [[0.9, 0.1], [0.2, 0.8]]),
        ("emission", [0.6, 0.4], [[0.7, 0.3], [0.2, 0.8]], [[1.0, 0.0]]),
    ],
)
def test_model_when_probability_shape_is_invalid(
    field: str,
    initial: list[float],
    transition: list[list[float]],
    emission: list[list[float]],
) -> None:
    with pytest.raises(ProbabilityShapeError) as captured:
        _ = BinaryHMM(
            labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
            initial=initial,
            transition=transition,
            emission=emission,
        )

    assert captured.value.field == field


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, math.nan, math.inf])
def test_model_when_probability_is_out_of_range(bad_value: float) -> None:
    with pytest.raises(ProbabilityRangeError):
        _ = BinaryHMM(
            labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
            initial=[bad_value, 1.0],
            transition=[[0.7, 0.3], [0.2, 0.8]],
            emission=[[0.9, 0.1], [0.2, 0.8]],
        )


def test_model_when_probability_sum_exceeds_tolerance() -> None:
    with pytest.raises(ProbabilitySumError):
        _ = BinaryHMM(
            labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
            initial=[0.6, 0.400000000002],
            transition=[[0.7, 0.3], [0.2, 0.8]],
            emission=[[0.9, 0.1], [0.2, 0.8]],
        )


def test_model_when_probability_sum_is_within_tolerance_is_not_normalized() -> None:
    model = BinaryHMM(
        labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
        initial=[0.6, 0.4000000000005],
        transition=[[0.7, 0.3], [0.2, 0.8]],
        emission=[[0.9, 0.1], [0.2, 0.8]],
    )

    assert model.initial[1] == 0.4000000000005


def test_model_arrays_when_mutation_is_attempted() -> None:
    model = hand_model()

    with pytest.raises(ValueError, match="read-only"):
        model.initial[0] = 0.5
