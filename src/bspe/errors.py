"""Typed application errors."""

from dataclasses import FrozenInstanceError, dataclass
from enum import StrEnum
from types import TracebackType
from typing import override


class BinaryEntropyError(ValueError):
    """Base class for calculator failures."""


class _FrozenMetadataError(BinaryEntropyError):
    """Permit exception metadata mutation while freezing domain fields."""

    @override
    def __setattr__(
        self,
        name: str,
        value: str
        | float
        | bool
        | tuple[int, ...]
        | BaseException
        | TracebackType
        | None,
    ) -> None:
        if name in {
            "__traceback__",
            "__cause__",
            "__context__",
            "__suppress_context__",
        }:
            BaseException.__setattr__(self, name, value)
            return
        message = f"cannot assign to field {name!r}"
        raise FrozenInstanceError(message)


@dataclass(frozen=True, slots=True)
class _ProbabilityRangeBaseError(BinaryEntropyError):
    field: str
    index: tuple[int, ...]
    value: float

    @override
    def __str__(self) -> str:
        return (
            f"{self.field}{self.index} must be finite and in [0, 1]; got {self.value}"
        )


class ProbabilityRangeError(_FrozenMetadataError, _ProbabilityRangeBaseError):
    """A probability is non-finite or outside the closed unit interval."""


@dataclass(frozen=True, slots=True)
class _ProbabilitySumBaseError(BinaryEntropyError):
    field: str
    row: int | None
    total: float

    @override
    def __str__(self) -> str:
        location = self.field if self.row is None else f"{self.field} row {self.row}"
        return f"{location} must sum to 1; got {self.total}"


class ProbabilitySumError(_FrozenMetadataError, _ProbabilitySumBaseError):
    """A probability vector does not sum to one within tolerance."""


@dataclass(frozen=True, slots=True)
class _ProbabilityShapeBaseError(BinaryEntropyError):
    field: str
    expected: tuple[int, ...]
    actual: tuple[int, ...]

    @override
    def __str__(self) -> str:
        return f"{self.field} must have shape {self.expected}; got {self.actual}"


class ProbabilityShapeError(_FrozenMetadataError, _ProbabilityShapeBaseError):
    """A probability array has an unsupported shape."""


@dataclass(frozen=True, slots=True)
class _InvalidLabelBaseError(BinaryEntropyError):
    category: str
    index: int
    value: str

    @override
    def __str__(self) -> str:
        return f"{self.category} label {self.index} is invalid: {self.value!r}"


class InvalidLabelError(_FrozenMetadataError, _InvalidLabelBaseError):
    """A state or observable label violates the label grammar."""


@dataclass(frozen=True, slots=True)
class _DuplicateLabelBaseError(BinaryEntropyError):
    category: str
    value: str

    @override
    def __str__(self) -> str:
        return f"{self.category} labels must be distinct; duplicate {self.value!r}"


class DuplicateLabelError(_FrozenMetadataError, _DuplicateLabelBaseError):
    """Two labels in one category are equal after trimming."""


@dataclass(frozen=True, slots=True)
class _InvalidSequenceTokenBaseError(BinaryEntropyError):
    token: str
    position: int

    @override
    def __str__(self) -> str:
        return f"invalid sequence token {self.token!r} at position {self.position}"


class InvalidSequenceTokenError(_FrozenMetadataError, _InvalidSequenceTokenBaseError):
    """A sequence token does not match either observable label."""


@dataclass(frozen=True, slots=True)
class _PresetDecodeBaseError(BinaryEntropyError):
    detail: str

    @override
    def __str__(self) -> str:
        return f"preset JSON could not be decoded: {self.detail}"


class PresetDecodeError(_FrozenMetadataError, _PresetDecodeBaseError):
    """Preset bytes are not valid strict UTF-8 JSON."""


@dataclass(frozen=True, slots=True)
class _PresetSchemaBaseError(BinaryEntropyError):
    detail: str

    @override
    def __str__(self) -> str:
        return f"preset JSON does not match schema version 1: {self.detail}"


class PresetSchemaError(_FrozenMetadataError, _PresetSchemaBaseError):
    """Decoded preset JSON does not satisfy the version-one schema."""


@dataclass(frozen=True, slots=True)
class _ZeroLikelihoodBaseError(BinaryEntropyError):
    observable_index: int
    position: int

    @override
    def __str__(self) -> str:
        return (
            f"observable {self.observable_index} at position {self.position} "
            "has zero likelihood"
        )


class ZeroLikelihoodError(_FrozenMetadataError, _ZeroLikelihoodBaseError):
    """An observed symbol has zero likelihood under the current prior."""


@dataclass(frozen=True, slots=True)
class _NumericalInvariantBaseError(BinaryEntropyError):
    quantity: str
    value: float

    @override
    def __str__(self) -> str:
        return f"calculated {self.quantity} violates its invariant: {self.value}"


class NumericalInvariantError(_FrozenMetadataError, _NumericalInvariantBaseError):
    """A calculated value violates a mathematical output invariant."""


class DatasetErrorCode(StrEnum):
    """Closed set of canonical dataset validation failures."""

    EMPTY_DATASET = "empty_dataset"
    INVALID_RECORD_ID = "invalid_record_id"
    DUPLICATE_RECORD_ID = "duplicate_record_id"


@dataclass(frozen=True, slots=True)
class _DatasetValidationBaseError(BinaryEntropyError):
    code: DatasetErrorCode
    value: str

    @override
    def __str__(self) -> str:
        return f"dataset validation failed ({self.code.value}): {self.value!r}"


class DatasetValidationError(_FrozenMetadataError, _DatasetValidationBaseError):
    """Canonical independent-record dataset invariants were violated."""


class BatchParseErrorCode(StrEnum):
    """Closed set of batch input boundary failures."""

    INVALID_UTF8 = "invalid_utf8"
    INVALID_COLUMNS = "invalid_columns"
    MISSING_COLUMN = "missing_column"
    MALFORMED_ROW = "malformed_row"
    INVALID_SEQUENCE = "invalid_sequence"
    INVALID_TARGET = "invalid_target"
    INVALID_RECORD_ID = "invalid_record_id"
    DUPLICATE_RECORD_ID = "duplicate_record_id"


@dataclass(frozen=True, slots=True)
class BatchRecordIssue:
    """One recoverable invalid-record diagnostic at a batch boundary."""

    row: int
    record_id: str | None
    token_position: int | None
    code: BatchParseErrorCode
    detail: str

    @override
    def __str__(self) -> str:
        record = "<unavailable>" if self.record_id is None else repr(self.record_id)
        token = "" if self.token_position is None else f", token {self.token_position}"
        return (
            f"row {self.row}, record {record}{token} ({self.code.value}): {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class _BatchRecordBaseError(BinaryEntropyError):
    issues: tuple[BatchRecordIssue, ...]

    @override
    def __str__(self) -> str:
        issue_label = "issue" if len(self.issues) == 1 else "issues"
        rendered_issues = "\n".join(f"- {issue}" for issue in self.issues)
        return (
            f"batch parse failed with {len(self.issues)} record {issue_label}:\n"
            f"{rendered_issues}"
        )


class BatchRecordError(_FrozenMetadataError, _BatchRecordBaseError):
    """All recoverable invalid-record diagnostics from one atomic batch parse."""


@dataclass(frozen=True, slots=True)
class _BatchParseBaseError(BinaryEntropyError):
    code: BatchParseErrorCode
    detail: str
    row: int | None = None

    @override
    def __str__(self) -> str:
        location = "" if self.row is None else f" at CSV row {self.row}"
        return f"batch parse failed ({self.code.value}){location}: {self.detail}"


class BatchParseError(_FrozenMetadataError, _BatchParseBaseError):
    """Manual, TXT, or CSV batch input could not be parsed atomically."""


@dataclass(frozen=True, slots=True)
class _InvalidSmoothingAlphaBaseError(BinaryEntropyError):
    smoothing_alpha: float

    @override
    def __str__(self) -> str:
        return (
            "smoothing alpha must be finite and greater than or equal to zero; "
            f"got {self.smoothing_alpha}"
        )


class InvalidSmoothingAlphaError(
    _FrozenMetadataError,
    _InvalidSmoothingAlphaBaseError,
):
    """A Markov additive-smoothing parameter is outside its domain."""
