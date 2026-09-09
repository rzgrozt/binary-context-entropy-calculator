"""Descriptive metrics for canonical binary stimuli."""

from itertools import pairwise

from binary_entropy.records import BinarySequence
from binary_entropy.stimulus_search_results import SequenceMetrics


def sequence_metrics(sequence: BinarySequence) -> SequenceMetrics:
    """Measure composition, switching, and run structure."""
    length = len(sequence)
    a_count = sequence.count(0)
    b_count = length - a_count
    switches = sum(left != right for left, right in pairwise(sequence))
    switch_rate = switches / (length - 1) if length > 1 else 0.0
    longest_a_run = _longest_run(sequence, 0)
    longest_b_run = _longest_run(sequence, 1)
    return SequenceMetrics(
        length=length,
        a_count=a_count,
        b_count=b_count,
        a_proportion=a_count / length if length else 0.0,
        b_proportion=b_count / length if length else 0.0,
        switches=switches,
        switch_rate=switch_rate,
        longest_a_run=longest_a_run,
        longest_b_run=longest_b_run,
        longest_run=max(longest_a_run, longest_b_run),
        alternation_tendency=switch_rate,
    )


def _longest_run(sequence: BinarySequence, symbol: int) -> int:
    longest = 0
    current = 0
    for observed in sequence:
        current = current + 1 if observed == symbol else 0
        longest = max(longest, current)
    return longest
