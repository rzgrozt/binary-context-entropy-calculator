"""Validated immutable configuration for deterministic stimulus search."""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NewType, override

from binary_entropy.errors import BinaryEntropyError
from binary_entropy.vmm_types import VMMConfig

StimulusId = NewType("StimulusId", str)
_BINARY_LABEL_COUNT: Final = 2
_MAX_SEQUENCE_LENGTH: Final = 32
_MAX_CANDIDATE_LIMIT: Final = 5_000
_MAX_SEED: Final = 2**63 - 1
_UNSAFE_LABEL_CHARACTERS: Final = frozenset({",", "|", "\n", "\r"})


@dataclass(frozen=True, slots=True)
class InvalidStimulusSearchConfigurationError(BinaryEntropyError):
    """A stimulus-search setting is outside its supported domain."""

    parameter: str
    value: str | int | float

    @override
    def __str__(self) -> str:
        return f"invalid stimulus search {self.parameter}: {self.value}"


@dataclass(frozen=True, slots=True)
class InclusiveRange:
    """A finite inclusive numeric interval."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Reject non-finite or reversed bounds."""
        parameter = "range"
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise InvalidStimulusSearchConfigurationError(parameter, repr(self))
        if self.minimum > self.maximum:
            raise InvalidStimulusSearchConfigurationError(parameter, repr(self))

    def contains(self, value: float) -> bool:
        """Return whether value lies inside both inclusive bounds."""
        return self.minimum <= value <= self.maximum


class PredictedSymbol(StrEnum):
    """Allowed modal prediction for a candidate."""

    A = "A"
    B = "B"
    EITHER = "either"


class PreferenceMetric(StrEnum):
    """Candidate quantities supported by transparent distance ranking."""

    PREDICTED_PROBABILITY = "predicted_probability"
    PREDICTIVE_ENTROPY = "predictive_entropy"
    EFFECTIVE_DEPTH = "effective_depth"
    CONTEXT_SUPPORT = "context_support"
    A_PROPORTION = "a_proportion"
    SWITCH_RATE = "switch_rate"
    LONGEST_RUN = "longest_run"


@dataclass(frozen=True, slots=True)
class SoftPreference:
    """Weighted absolute-distance preference for one candidate quantity."""

    metric: PreferenceMetric
    target: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject non-finite targets and invalid weights."""
        if not math.isfinite(self.target):
            parameter = "preference target"
            raise InvalidStimulusSearchConfigurationError(parameter, self.target)
        if not math.isfinite(self.weight) or self.weight < 0.0:
            parameter = "preference weight"
            raise InvalidStimulusSearchConfigurationError(parameter, self.weight)


@dataclass(frozen=True, slots=True)
class StimulusConstraints:
    """Optional hard filters applied without implicit relaxation."""

    predicted_symbol: PredictedSymbol = PredictedSymbol.EITHER
    predicted_probability: InclusiveRange | None = None
    predictive_entropy: InclusiveRange | None = None
    effective_depth: InclusiveRange | None = None
    context_support: InclusiveRange | None = None
    a_count: InclusiveRange | None = None
    a_proportion: InclusiveRange | None = None
    switches: InclusiveRange | None = None
    switch_rate: InclusiveRange | None = None
    longest_a_run: InclusiveRange | None = None
    longest_b_run: InclusiveRange | None = None
    longest_run: InclusiveRange | None = None


@dataclass(frozen=True, slots=True)
class StimulusSearchConfig:
    """Complete reproducible search and VMM configuration."""

    sequence_length: int
    desired_stimuli: int
    seed: int
    candidate_limit: int
    vmm_config: VMMConfig
    constraints: StimulusConstraints = field(default_factory=StimulusConstraints)
    preferences: tuple[SoftPreference, ...] = ()
    symbol_mapping: tuple[str, str] = ("A", "B")

    def __post_init__(self) -> None:
        """Reject non-positive limits and ambiguous presentation labels."""
        for parameter, value in (
            ("sequence_length", self.sequence_length),
            ("desired_stimuli", self.desired_stimuli),
            ("candidate_limit", self.candidate_limit),
        ):
            if value <= 0:
                raise InvalidStimulusSearchConfigurationError(parameter, value)
        for parameter, value, maximum in (
            ("sequence_length", self.sequence_length, _MAX_SEQUENCE_LENGTH),
            ("candidate_limit", self.candidate_limit, _MAX_CANDIDATE_LIMIT),
            ("seed", self.seed, _MAX_SEED),
        ):
            if not 0 <= value <= maximum:
                raise InvalidStimulusSearchConfigurationError(parameter, value)
        if self.desired_stimuli > self.candidate_limit:
            parameter = "desired_stimuli"
            raise InvalidStimulusSearchConfigurationError(
                parameter, self.desired_stimuli
            )
        if len(self.symbol_mapping) != _BINARY_LABEL_COUNT:
            parameter = "symbol_mapping"
            raise InvalidStimulusSearchConfigurationError(
                parameter, repr(self.symbol_mapping)
            )
        first, second = (label.strip() for label in self.symbol_mapping)
        labels_are_safe = (
            bool(first)
            and bool(second)
            and first != second
            and all(
                character not in label
                for label in self.symbol_mapping
                for character in _UNSAFE_LABEL_CHARACTERS
            )
        )
        if not labels_are_safe:
            parameter = "symbol_mapping"
            raise InvalidStimulusSearchConfigurationError(
                parameter, repr(self.symbol_mapping)
            )
        object.__setattr__(self, "symbol_mapping", (first, second))
        _validate_constraint_ranges(self.constraints)


def _validate_constraint_ranges(constraints: StimulusConstraints) -> None:
    for name, value in (
        ("predicted_probability", constraints.predicted_probability),
        ("predictive_entropy", constraints.predictive_entropy),
        ("a_proportion", constraints.a_proportion),
        ("switch_rate", constraints.switch_rate),
    ):
        if value is not None and (value.minimum < 0.0 or value.maximum > 1.0):
            raise InvalidStimulusSearchConfigurationError(name, repr(value))
    for name, value in (
        ("effective_depth", constraints.effective_depth),
        ("context_support", constraints.context_support),
        ("a_count", constraints.a_count),
        ("switches", constraints.switches),
        ("longest_a_run", constraints.longest_a_run),
        ("longest_b_run", constraints.longest_b_run),
        ("longest_run", constraints.longest_run),
    ):
        if value is not None and value.minimum < 0.0:
            raise InvalidStimulusSearchConfigurationError(name, repr(value))
