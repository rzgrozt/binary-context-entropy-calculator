"""Descriptive observed-symbol Shannon analysis."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from binary_entropy.domain import (
    FloatArray,
    LabelPair,
    ObservableIndex,
    readonly_vector,
)
from binary_entropy.information import binary_entropy
from binary_entropy.records import SequenceDataset, SequenceId


@dataclass(frozen=True, slots=True)
class ShannonSummary:
    """Observed binary counts, frequencies, and entropy for one collection."""

    observation_count: int
    symbol_counts: tuple[int, int]
    symbol_probabilities: FloatArray | None
    entropy_bits: float | None

    def __post_init__(self) -> None:
        """Own a read-only float64 frequency vector when observations exist."""
        if self.symbol_probabilities is not None:
            object.__setattr__(
                self,
                "symbol_probabilities",
                readonly_vector(self.symbol_probabilities),
            )


@dataclass(frozen=True, slots=True)
class ShannonPrefixResult:
    """Observed frequencies after one nonempty sequence prefix."""

    depth: int
    count_a: int
    count_b: int
    probability_a: float
    probability_b: float
    entropy_bits: float


@dataclass(frozen=True, slots=True)
class ShannonRecordAnalysis:
    """Descriptive result for one independent record."""

    sequence_id: SequenceId
    summary: ShannonSummary
    prefixes: tuple[ShannonPrefixResult, ...] = ()


@dataclass(frozen=True, slots=True)
class ShannonBatchAnalysis:
    """Pooled and per-record descriptive Shannon results."""

    observable_labels: LabelPair
    pooled: ShannonSummary
    records: tuple[ShannonRecordAnalysis, ...]
    method: Literal["observed_shannon"] = field(
        default="observed_shannon",
        init=False,
    )


def analyze_shannon(dataset: SequenceDataset) -> ShannonBatchAnalysis:
    """Describe pooled and independent observed-symbol frequencies."""
    pooled_symbols: list[ObservableIndex] = []
    for record in dataset.records:
        pooled_symbols.extend(record.sequence)
    return ShannonBatchAnalysis(
        observable_labels=dataset.labels.observables,
        pooled=_summarize(pooled_symbols),
        records=tuple(
            ShannonRecordAnalysis(
                sequence_id=record.sequence_id,
                summary=_summarize(record.sequence),
                prefixes=_prefixes(record.sequence),
            )
            for record in dataset.records
        ),
    )


def _summarize(sequence: Sequence[ObservableIndex]) -> ShannonSummary:
    observation_count = len(sequence)
    counts = sequence.count(0), sequence.count(1)
    if observation_count == 0:
        return ShannonSummary(0, counts, None, None)
    probability_0 = counts[0] / observation_count
    probabilities: FloatArray = np.array(
        [probability_0, counts[1] / observation_count],
        dtype=np.float64,
    )
    return ShannonSummary(
        observation_count,
        counts,
        probabilities,
        binary_entropy(probability_0),
    )


def _prefixes(
    sequence: Sequence[ObservableIndex],
) -> tuple[ShannonPrefixResult, ...]:
    count_a = 0
    count_b = 0
    results: list[ShannonPrefixResult] = []
    for depth, symbol in enumerate(sequence, start=1):
        match symbol:
            case 0:
                count_a += 1
            case 1:
                count_b += 1
        probability_a = count_a / depth
        results.append(
            ShannonPrefixResult(
                depth=depth,
                count_a=count_a,
                count_b=count_b,
                probability_a=probability_a,
                probability_b=count_b / depth,
                entropy_bits=binary_entropy(probability_a),
            )
        )
    return tuple(results)
