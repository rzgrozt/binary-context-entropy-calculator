"""Immutable domain values for the binary entropy core."""

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from binary_entropy.constants import PROBABILITY_TOLERANCE
from binary_entropy.errors import (
    DuplicateLabelError,
    InvalidLabelError,
    ProbabilityRangeError,
    ProbabilityShapeError,
    ProbabilitySumError,
)

type FloatArray = NDArray[np.float64]
type ObservableIndex = Literal[0, 1]
type LabelPair = tuple[str, str]
type FloatInputArray = NDArray[np.float32] | NDArray[np.float64]
type ProbabilityVectorInput = Sequence[float] | FloatInputArray
type ProbabilityMatrixInput = Sequence[Sequence[float]] | FloatInputArray


class TargetClassification(StrEnum):
    """Relationship between an actual target and its predictive distribution."""

    MODAL = "modal"
    LOWER_PROBABILITY = "lower_probability"
    TIED = "tied"


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    """Metrics for one actual target under a predictive distribution."""

    actual_target_index: ObservableIndex
    probability: float
    surprisal_bits: float
    classification: TargetClassification


@dataclass(frozen=True, slots=True, init=False)
class BinaryLabels:
    """Exactly two hidden-state and observable labels."""

    states: LabelPair
    observables: LabelPair

    def __init__(self, states: LabelPair, observables: LabelPair) -> None:
        """Trim and validate exactly two labels in each category."""
        object.__setattr__(self, "states", _validated_labels("state", states))
        object.__setattr__(
            self,
            "observables",
            _validated_labels("observable", observables),
        )


@dataclass(frozen=True, slots=True, init=False)
class BinaryHMM:
    """Validated two-state, two-observable hidden Markov model."""

    labels: BinaryLabels
    initial: FloatArray
    transition: FloatArray
    emission: FloatArray

    def __init__(
        self,
        labels: BinaryLabels,
        initial: ProbabilityVectorInput,
        transition: ProbabilityMatrixInput,
        emission: ProbabilityMatrixInput,
    ) -> None:
        """Validate and own read-only float64 copies of all probabilities."""
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "initial", _validated_vector("initial", initial))
        object.__setattr__(
            self,
            "transition",
            _validated_matrix("transition", transition),
        )
        object.__setattr__(
            self,
            "emission",
            _validated_matrix("emission", emission),
        )


@dataclass(frozen=True, slots=True)
class PrefixResult:
    """Prediction after consuming a prefix of the sequence.

    At depth zero, ``posterior`` is absent and ``next_hidden`` is the initial
    hidden-state distribution because no observation or transition occurred.
    """

    depth: int
    observed_index: ObservableIndex | None
    posterior: FloatArray | None
    next_hidden: FloatArray
    predictive: FloatArray
    entropy_bits: float
    predicted_index: ObservableIndex
    actual_target_index: ObservableIndex | None
    target_classification: TargetClassification | None
    actual_target_probability: float | None
    actual_target_surprisal_bits: float | None

    def __post_init__(self) -> None:
        """Own read-only copies of every stored distribution."""
        if self.posterior is not None:
            object.__setattr__(self, "posterior", readonly_vector(self.posterior))
        object.__setattr__(self, "next_hidden", readonly_vector(self.next_hidden))
        object.__setattr__(self, "predictive", readonly_vector(self.predictive))


@dataclass(frozen=True, slots=True)
class SequenceAnalysis:
    """All deterministic prefix results for one parsed sequence."""

    sequence: tuple[ObservableIndex, ...]
    rows: tuple[PrefixResult, ...]
    observed_entropy_bits: float | None


def readonly_vector(values: FloatArray) -> FloatArray:
    """Copy a float64 vector and prevent mutation of the owned array."""
    result: FloatArray = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def float_values(values: FloatArray) -> tuple[float, ...]:
    """Return typed Python floats from a contiguous float64 array."""
    scalars = array("d")
    scalars.frombytes(values.tobytes(order="C"))
    return tuple(scalars)


def _validated_labels(category: str, values: LabelPair) -> LabelPair:
    first = values[0].strip()
    second = values[1].strip()
    for index, value in enumerate((first, second)):
        if not value or "," in value or "\n" in value or "\r" in value:
            raise InvalidLabelError(category=category, index=index, value=value)
    if first == second:
        raise DuplicateLabelError(category=category, value=first)
    return first, second


def _validated_vector(
    field: str,
    values: ProbabilityVectorInput,
) -> FloatArray:
    result: FloatArray = np.array(values, dtype=np.float64, copy=True)
    expected = (2,)
    if result.shape != expected:
        raise ProbabilityShapeError(field=field, expected=expected, actual=result.shape)
    scalars = float_values(result)
    for index, value in enumerate(scalars):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProbabilityRangeError(field=field, index=(index,), value=value)
    total = math.fsum(scalars)
    if not math.isclose(
        total,
        1.0,
        abs_tol=PROBABILITY_TOLERANCE,
        rel_tol=0.0,
    ):
        raise ProbabilitySumError(field=field, row=None, total=total)
    result.setflags(write=False)
    return result


def _validated_matrix(
    field: str,
    values: ProbabilityMatrixInput,
) -> FloatArray:
    result: FloatArray = np.array(values, dtype=np.float64, copy=True)
    expected = (2, 2)
    if result.shape != expected:
        raise ProbabilityShapeError(field=field, expected=expected, actual=result.shape)
    scalars = float_values(result)
    for row in range(2):
        row_values = scalars[row * 2 : row * 2 + 2]
        for column, value in enumerate(row_values):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ProbabilityRangeError(
                    field=field,
                    index=(row, column),
                    value=value,
                )
        total = math.fsum(row_values)
        if not math.isclose(
            total,
            1.0,
            abs_tol=PROBABILITY_TOLERANCE,
            rel_tol=0.0,
        ):
            raise ProbabilitySumError(field=field, row=row, total=total)
    result.setflags(write=False)
    return result
