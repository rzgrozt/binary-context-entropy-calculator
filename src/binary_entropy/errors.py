"""Typed application errors."""

from dataclasses import FrozenInstanceError, dataclass
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
