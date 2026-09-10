"""Immutable records and reports produced by stimulus search."""

import math
from dataclasses import dataclass
from enum import StrEnum

from bspe.domain import ObservableIndex
from bspe.records import BinarySequence
from bspe.stimulus_search_types import (
    InvalidStimulusSearchConfigurationError,
    StimulusId,
    StimulusSearchConfig,
)
from bspe.vmm_types import VMMDepthAnalysis


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    """Exact descriptive metrics for one generated binary sequence."""

    length: int
    a_count: int
    b_count: int
    a_proportion: float
    b_proportion: float
    switches: int
    switch_rate: float
    longest_a_run: int
    longest_b_run: int
    longest_run: int
    alternation_tendency: float


class TargetCongruency(StrEnum):
    """Relationship between an assigned target and modal prediction."""

    EXPECTED = "expected"
    UNEXPECTED = "unexpected"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class StimulusCandidate:
    """One independently analyzed canonical 0/1 stimulus."""

    stimulus_id: StimulusId
    sequence: BinarySequence
    metrics: SequenceMetrics
    probability_a: float | None
    probability_b: float | None
    predicted_target_index: ObservableIndex | None
    predictive_entropy_bits: float | None
    effective_context_depth: int | None
    context_used: BinarySequence | None
    support_count: int | None
    surprisal_a_bits: float | None
    surprisal_b_bits: float | None
    depth_rows: tuple[VMMDepthAnalysis, ...]
    rank_score: float = 0.0
    group_id: str | None = None
    pair_id: str | None = None
    condition: str | None = None
    symbol_mapping: tuple[str, str] = ("A", "B")
    target_index: ObservableIndex | None = None
    target_probability: float | None = None
    target_surprisal_bits: float | None = None
    target_congruency: TargetCongruency | None = None


class SearchStatus(StrEnum):
    """Whether the requested selection size was reached."""

    COMPLETE = "complete"
    PARTIAL = "partial"


class SearchPartialReason(StrEnum):
    """Bound that ended a partial search before the requested selection."""

    CANDIDATE_LIMIT = "candidate_limit"
    UNIVERSE_EXHAUSTED = "universe_exhausted"


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """Frequency of one failed hard constraint."""

    constraint: str
    count: int
    frequency: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Stored reproducible search snapshot."""

    config: StimulusSearchConfig
    timestamp_utc: str
    package_version: str
    evaluated_count: int
    accepted_count: int
    selected_count: int
    status: SearchStatus
    partial_reason: SearchPartialReason | None
    accepted: tuple[StimulusCandidate, ...]
    selected: tuple[StimulusCandidate, ...]
    violations: tuple[ConstraintViolation, ...]


@dataclass(frozen=True, slots=True)
class MatchTolerances:
    """Maximum absolute differences allowed for complementary matching."""

    entropy: float = 1.0
    prediction_strength: float = 1.0
    a_proportion: float = 1.0
    switch_rate: float = 1.0
    run_structure: float = 1.0
    effective_depth: float = float("inf")

    def __post_init__(self) -> None:
        """Reject negative and undefined tolerances."""
        for name, value in (
            ("entropy", self.entropy),
            ("prediction_strength", self.prediction_strength),
            ("a_proportion", self.a_proportion),
            ("switch_rate", self.switch_rate),
            ("run_structure", self.run_structure),
            ("effective_depth", self.effective_depth),
        ):
            if math.isnan(value) or value < 0.0:
                raise InvalidStimulusSearchConfigurationError(name, value)


@dataclass(frozen=True, slots=True)
class StimulusPair:
    """One stable tolerance-qualified A/B prediction pair."""

    pair_id: str
    first: StimulusCandidate
    second: StimulusCandidate
    distance: float


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """Matched pairs and explicitly retained unmatched stimuli."""

    pairs: tuple[StimulusPair, ...]
    unmatched: tuple[StimulusCandidate, ...]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Minimum, mean, and maximum for one available numeric quantity."""

    minimum: float
    mean: float
    maximum: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Descriptive quality-control summary without inferential claims."""

    stimulus_count: int
    predicted_a_count: int
    predicted_b_count: int
    tie_count: int
    unavailable_count: int
    expected_target_count: int
    unexpected_target_count: int
    tie_target_count: int
    unassigned_target_count: int
    a_count: MetricSummary | None
    b_count: MetricSummary | None
    a_proportion: MetricSummary | None
    b_proportion: MetricSummary | None
    switches: MetricSummary | None
    switch_rate: MetricSummary | None
    longest_a_run: MetricSummary | None
    longest_b_run: MetricSummary | None
    longest_run: MetricSummary | None
    predicted_probability: MetricSummary | None
    predictive_entropy: MetricSummary | None
    effective_depth: MetricSummary | None
    context_support: MetricSummary | None
    target_surprisal: MetricSummary | None
    imbalance_flags: tuple[str, ...]
