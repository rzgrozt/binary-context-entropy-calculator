import pytest

from binary_entropy.domain import BinaryLabels
from binary_entropy.errors import DatasetValidationError
from binary_entropy.records import SequenceDataset, SequenceRecord


def _labels() -> BinaryLabels:
    return BinaryLabels(states=("S1", "S2"), observables=("A", "B"))


def test_sequence_record_when_id_has_whitespace_trims_and_allows_empty_sequence() -> (
    None
):
    # Given
    sequence_id = "  candidate-1  "

    # When
    result = SequenceRecord(sequence_id, ())

    # Then
    assert result.sequence_id == "candidate-1"
    assert result.sequence == ()


def test_sequence_dataset_when_no_records_rejects_empty_collection() -> None:
    # Given
    records: tuple[SequenceRecord, ...] = ()

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceDataset(_labels(), records)


def test_sequence_dataset_when_trimmed_ids_repeat_rejects_duplicate() -> None:
    # Given
    records = (
        SequenceRecord("candidate-1", (0,)),
        SequenceRecord(" candidate-1 ", (1,)),
    )

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceDataset(_labels(), records)


def test_sequence_record_when_id_is_blank_rejects_record() -> None:
    # Given
    sequence_id = " \t "

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceRecord(sequence_id, ())
