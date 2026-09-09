"""Deterministic candidate sequence generation."""

import random

from binary_entropy.domain import ObservableIndex
from binary_entropy.records import SequenceRecord
from binary_entropy.stimulus_search_types import StimulusSearchConfig


def generate_candidate_records(
    config: StimulusSearchConfig,
) -> tuple[SequenceRecord, ...]:
    """Generate a bounded sample of independent candidate records."""
    universe_size = 1 << config.sequence_length
    sample_size = min(config.candidate_limit, universe_size)
    values = random.Random(config.seed).sample(  # noqa: S311
        range(universe_size), sample_size
    )
    return tuple(
        SequenceRecord(
            f"stimulus-{index:06d}",
            _binary_sequence(value, config.sequence_length),
        )
        for index, value in enumerate(values, start=1)
    )


def _binary_sequence(value: int, length: int) -> tuple[ObservableIndex, ...]:
    return tuple(
        1 if value & (1 << shift) else 0 for shift in range(length - 1, -1, -1)
    )
