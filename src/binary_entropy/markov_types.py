"""Immutable result values for first-order binary Markov analysis."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from binary_entropy.domain import (
    FloatArray,
    LabelPair,
    ObservableIndex,
    TargetAssessment,
    readonly_vector,
)
from binary_entropy.records import BinarySequence, SequenceId

type MarkovOrder = Literal[1]
type MarkovContext = tuple[ObservableIndex, *tuple[ObservableIndex, ...]]
type TransitionCounts = tuple[tuple[int, int], tuple[int, int]]
type TransitionMatrix = tuple[FloatArray | None, FloatArray | None]


class MarkovEstimation(StrEnum):
    """Supported first-order transition estimators."""

    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    ADDITIVE_SMOOTHING = "additive_smoothing"


class MarkovPredictionMode(StrEnum):
    """Whether prefix predictions share one fit or refit cumulatively."""

    FIXED_MODEL = "fixed_model"
    CUMULATIVE_PREFIX = "cumulative_prefix"


class MarkovResultScope(StrEnum):
    """Whether full-model results are pooled or independently fitted."""

    POOLED = "pooled"
    PER_SEQUENCE = "per_sequence"


class StationaryUnavailableReason(StrEnum):
    """Why a unique stationary distribution cannot be reported."""

    INCOMPLETE_MATRIX = "incomplete_matrix"
    NON_UNIQUE = "non_unique"


@dataclass(frozen=True, slots=True)
class UniqueStationaryDistribution:
    """The selected transition matrix has exactly one stationary distribution."""

    distribution: FloatArray

    def __post_init__(self) -> None:
        """Own a read-only float64 stationary distribution."""
        object.__setattr__(self, "distribution", readonly_vector(self.distribution))


@dataclass(frozen=True, slots=True)
class UnavailableStationaryDistribution:
    """The selected matrix cannot identify one stationary distribution."""

    reason: StationaryUnavailableReason


type StationaryDistributionResult = (
    UniqueStationaryDistribution | UnavailableStationaryDistribution
)


@dataclass(frozen=True, slots=True)
class MarkovModel:
    """A fitted first-order model with explicit unavailable transition rows."""

    observable_labels: LabelPair
    estimation_method: MarkovEstimation
    smoothing_alpha: float
    transition_counts: TransitionCounts
    transition_matrix: TransitionMatrix
    starting_distribution: FloatArray | None
    source_sequence_count: int
    source_transition_count: int
    markov_order: MarkovOrder = field(default=1, init=False)

    def __post_init__(self) -> None:
        """Own read-only float64 copies of all available distributions."""
        row_a, row_b = self.transition_matrix
        owned_matrix = (
            readonly_vector(row_a) if row_a is not None else None,
            readonly_vector(row_b) if row_b is not None else None,
        )
        owned_start = (
            readonly_vector(self.starting_distribution)
            if self.starting_distribution is not None
            else None
        )
        object.__setattr__(self, "transition_matrix", owned_matrix)
        object.__setattr__(self, "starting_distribution", owned_start)


@dataclass(frozen=True, slots=True)
class MarkovPrefixResult:
    """Prediction selected after one sequence prefix."""

    depth: int
    context: MarkovContext | None
    prediction_mode: MarkovPredictionMode
    fitted_transition_count: int
    observed_next_index: ObservableIndex | None
    predictive: FloatArray | None
    entropy_bits: float | None
    predicted_index: ObservableIndex | None

    def __post_init__(self) -> None:
        """Own a read-only float64 prediction when one is available."""
        if self.predictive is not None:
            object.__setattr__(self, "predictive", readonly_vector(self.predictive))


@dataclass(frozen=True, slots=True)
class MarkovRecordAnalysis:
    """One record's selected full model, prefixes, and target assessment."""

    sequence_id: SequenceId
    sequence: BinarySequence
    sequence_length: int
    actual_target_index: ObservableIndex | None
    model: MarkovModel
    rows: tuple[MarkovPrefixResult, ...]
    target_assessment: TargetAssessment | None


@dataclass(frozen=True, slots=True)
class MarkovBatchAnalysis:
    """Pooled aggregate model and explicitly scoped per-record analyses."""

    model: MarkovModel
    prediction_mode: MarkovPredictionMode
    records: tuple[MarkovRecordAnalysis, ...]
    empirical_conditional_entropy_bits: float | None
    stationary: StationaryDistributionResult
    entropy_rate_bits: float | None
    result_scope: MarkovResultScope = MarkovResultScope.POOLED
    method: Literal["markov"] = field(default="markov", init=False)
