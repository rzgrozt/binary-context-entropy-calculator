"""Canonical independent binary sequence records."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from binary_entropy.domain import BinaryLabels, ObservableIndex
from binary_entropy.errors import DatasetErrorCode, DatasetValidationError

SequenceId = NewType("SequenceId", str)
type BinarySequence = tuple[ObservableIndex, ...]


@dataclass(frozen=True, slots=True, init=False)
class SequenceRecord:
    """One independently observed binary sequence and optional evaluation target."""

    sequence_id: SequenceId
    sequence: BinarySequence
    actual_target_index: ObservableIndex | None

    def __init__(
        self,
        sequence_id: str,
        sequence: BinarySequence,
        actual_target_index: ObservableIndex | None = None,
    ) -> None:
        """Trim and validate the identifier while preserving an empty sequence."""
        normalized_id = sequence_id.strip()
        if not normalized_id:
            raise DatasetValidationError(
                code=DatasetErrorCode.INVALID_RECORD_ID,
                value=sequence_id,
            )
        object.__setattr__(self, "sequence_id", SequenceId(normalized_id))
        object.__setattr__(self, "sequence", tuple(sequence))
        object.__setattr__(self, "actual_target_index", actual_target_index)


@dataclass(frozen=True, slots=True, init=False)
class SequenceDataset:
    """At least one uniquely identified independent sequence."""

    labels: BinaryLabels
    records: tuple[SequenceRecord, ...]

    def __init__(
        self,
        labels: BinaryLabels,
        records: Sequence[SequenceRecord],
    ) -> None:
        """Own records after requiring a nonempty collection of unique IDs."""
        owned_records = tuple(records)
        if not owned_records:
            raise DatasetValidationError(
                code=DatasetErrorCode.EMPTY_DATASET,
                value="records",
            )
        seen_ids: set[SequenceId] = set()
        for record in owned_records:
            if record.sequence_id in seen_ids:
                raise DatasetValidationError(
                    code=DatasetErrorCode.DUPLICATE_RECORD_ID,
                    value=record.sequence_id,
                )
            seen_ids.add(record.sequence_id)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "records", owned_records)
