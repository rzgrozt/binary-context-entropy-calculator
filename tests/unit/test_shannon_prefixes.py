import pytest

from binary_entropy.domain import BinaryLabels
from binary_entropy.methods.shannon import analyze_shannon
from binary_entropy.records import SequenceDataset, SequenceRecord


def test_analyze_shannon_when_sequence_is_aab_has_exact_nonempty_prefixes() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(labels, (SequenceRecord("candidate", (0, 0, 1)),))

    # When
    result = analyze_shannon(dataset)

    # Then
    prefixes = result.records[0].prefixes
    assert tuple(prefix.depth for prefix in prefixes) == (1, 2, 3)
    assert tuple((prefix.count_a, prefix.count_b) for prefix in prefixes) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    probabilities = tuple(
        (prefix.probability_a, prefix.probability_b) for prefix in prefixes
    )
    assert probabilities == (
        (1.0, 0.0),
        (1.0, 0.0),
        (2 / 3, 1 / 3),
    )
    assert tuple(prefix.entropy_bits for prefix in prefixes) == pytest.approx(
        (0.0, 0.0, 0.9182958340544896),
        abs=1e-15,
    )
    assert result.records[0].summary.symbol_counts == (2, 1)


def test_analyze_shannon_when_sequence_is_empty_omits_unavailable_depth_zero() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(labels, (SequenceRecord("empty", ()),))

    # When
    result = analyze_shannon(dataset)

    # Then
    assert result.records[0].prefixes == ()
