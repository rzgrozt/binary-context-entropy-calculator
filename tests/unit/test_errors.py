import pytest

from binary_entropy.errors import (
    BinaryEntropyError,
    DuplicateLabelError,
    InvalidLabelError,
    InvalidSequenceTokenError,
    NumericalInvariantError,
    PresetDecodeError,
    PresetSchemaError,
    ProbabilityRangeError,
    ProbabilityShapeError,
    ProbabilitySumError,
    ZeroLikelihoodError,
)


@pytest.mark.parametrize(
    ("error", "expected_fragment"),
    [
        (ProbabilityRangeError("initial", (0,), -0.1), "finite and in [0, 1]"),
        (ProbabilitySumError("initial", None, 0.9), "initial must sum to 1"),
        (ProbabilitySumError("transition", 1, 0.9), "transition row 1"),
        (ProbabilityShapeError("emission", (2, 2), (1, 2)), "shape (2, 2)"),
        (InvalidLabelError("state", 0, ""), "state label 0"),
        (DuplicateLabelError("observable", "A"), "duplicate 'A'"),
        (InvalidSequenceTokenError("C", 2), "at position 2"),
        (PresetDecodeError("bad bytes"), "could not be decoded"),
        (PresetSchemaError("missing field"), "schema version 1"),
        (ZeroLikelihoodError(1, 3), "zero likelihood"),
        (NumericalInvariantError("entropy", 1.1), "violates its invariant"),
    ],
)
def test_typed_error_when_rendered_has_reproducible_detail(
    error: BinaryEntropyError,
    expected_fragment: str,
) -> None:
    result = str(error)

    assert expected_fragment in result
