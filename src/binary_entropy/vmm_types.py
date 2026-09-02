"""Immutable values for binary variable-order Markov analysis."""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

from binary_entropy.domain import ObservableIndex, TargetAssessment
from binary_entropy.errors import BinaryEntropyError
from binary_entropy.records import BinarySequence, SequenceId

_ALPHA_PARAMETER: Final = "alpha"
_MINIMUM_SUPPORT_PARAMETER: Final = "minimum_support"


class InvalidVMMConfigurationError(BinaryEntropyError):
    """A VMM smoothing or support parameter is outside its domain."""

    parameter: str
    value: float

    def __init__(self, parameter: str, value: float) -> None:
        """Retain the invalid parameter and value for boundary reporting."""
        self.parameter = parameter
        self.value = value
        super().__init__(f"{parameter} must be finite and positive; got {value}")


@dataclass(frozen=True, slots=True)
class KTSmoothing:
    """Krichevsky-Trofimov smoothing with fixed alpha one half."""

    alpha: float = field(default=0.5, init=False)


@dataclass(frozen=True, slots=True)
class AdditiveSmoothing:
    """Custom positive additive smoothing."""

    alpha: float

    def __post_init__(self) -> None:
        """Require a finite, strictly positive additive alpha."""
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise InvalidVMMConfigurationError(_ALPHA_PARAMETER, self.alpha)


@dataclass(frozen=True, slots=True)
class MLESmoothing(AdditiveSmoothing):
    """Maximum-likelihood estimation with fixed alpha zero."""

    alpha: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Treat the fixed MLE boundary separately from custom additive alpha."""


type VMMSmoothing = KTSmoothing | AdditiveSmoothing


@dataclass(frozen=True, slots=True)
class VMMConfig:
    """Suffix-selection threshold and binary smoothing choice."""

    smoothing: VMMSmoothing = field(default_factory=KTSmoothing)
    minimum_support: int = 1

    def __post_init__(self) -> None:
        """Require at least one observation before accepting a context."""
        if self.minimum_support <= 0:
            raise InvalidVMMConfigurationError(
                _MINIMUM_SUPPORT_PARAMETER,
                self.minimum_support,
            )


class VMMDepthStatus(StrEnum):
    """Selection status for one candidate suffix depth."""

    ACCEPTED = "accepted"
    LOW_SUPPORT = "low_support"
    UNAVAILABLE = "unavailable"


class VMMResultScope(StrEnum):
    """Whether records share pooled counts or use independent counts."""

    POOLED = "pooled"
    PER_SEQUENCE = "per_sequence"


@dataclass(frozen=True, slots=True)
class VMMContextCount:
    """Raw binary continuation counts for one observed context."""

    context: BinarySequence
    count_a: int
    count_b: int

    @property
    def support(self) -> int:
        """Return the total observed outcomes after this context."""
        return self.count_a + self.count_b


@dataclass(frozen=True, slots=True)
class VMMModel:
    """Boundary-preserving continuation counts fitted from records."""

    context_counts: tuple[VMMContextCount, ...]
    source_sequence_count: int


@dataclass(frozen=True, slots=True)
class VMMDepthAnalysis:
    """Evidence and smoothed prediction for one final suffix depth."""

    depth: int
    matched_suffix: BinarySequence
    support: int
    count_a: int
    count_b: int
    probability_a: float | None
    probability_b: float | None
    predictive_entropy_bits: float | None
    status: VMMDepthStatus


@dataclass(frozen=True, slots=True)
class VMMRecordAnalysis:
    """Final backoff prediction and all depth evidence for one record."""

    sequence_id: SequenceId
    sequence: BinarySequence
    model: VMMModel
    effective_context_depth: int | None
    context_used: BinarySequence | None
    support_count: int | None
    probability_a: float | None
    probability_b: float | None
    predicted_target_index: ObservableIndex | None
    predictive_entropy_bits: float | None
    surprisal_a_bits: float | None
    surprisal_b_bits: float | None
    depth_rows: tuple[VMMDepthAnalysis, ...]
    target_assessment: TargetAssessment | None


@dataclass(frozen=True, slots=True)
class VMMAnalysis:
    """Explicitly scoped VMM analysis for every dataset record."""

    config: VMMConfig
    records: tuple[VMMRecordAnalysis, ...]
    result_scope: VMMResultScope
    method: Literal["vmm"] = field(default="vmm", init=False)
