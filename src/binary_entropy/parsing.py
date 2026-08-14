"""Text parsing boundaries."""

from dataclasses import dataclass

from binary_entropy.domain import BinaryLabels, ObservableIndex
from binary_entropy.errors import InvalidSequenceTokenError


@dataclass(frozen=True, slots=True)
class _LabelCandidate:
    text: str
    index: ObservableIndex


def parse_sequence(
    text: str,
    labels: BinaryLabels,
) -> tuple[ObservableIndex, ...]:
    """Parse observable labels separated by commas or Unicode whitespace."""
    if not text.strip():
        return ()
    candidates = tuple(
        sorted(
            (
                _LabelCandidate(labels.observables[0], 0),
                _LabelCandidate(labels.observables[1], 1),
            ),
            key=lambda candidate: len(candidate.text),
            reverse=True,
        )
    )
    sequence: list[ObservableIndex] = []
    cursor = _skip_outer_separators(text, 0)
    while cursor < len(text):
        candidate = next(
            (
                item
                for item in candidates
                if text.startswith(item.text, cursor)
                and _has_end_boundary(text, cursor + len(item.text))
            ),
            None,
        )
        if candidate is None:
            raise InvalidSequenceTokenError(
                token=_invalid_token(text, cursor),
                position=len(sequence) + 1,
            )
        sequence.append(candidate.index)
        cursor += len(candidate.text)
        if cursor == len(text):
            break
        cursor = _skip_inner_separators(text, cursor, len(sequence) + 1)
    return tuple(sequence)


def _has_end_boundary(text: str, cursor: int) -> bool:
    return cursor == len(text) or text[cursor] == "," or text[cursor].isspace()


def _skip_outer_separators(text: str, cursor: int) -> int:
    while cursor < len(text) and (text[cursor] == "," or text[cursor].isspace()):
        cursor += 1
    return cursor


def _skip_inner_separators(text: str, cursor: int, position: int) -> int:
    comma_count = 0
    start = cursor
    while cursor < len(text) and (text[cursor] == "," or text[cursor].isspace()):
        comma_count += text[cursor] == ","
        cursor += 1
    if cursor == len(text):
        return cursor
    if comma_count > 1:
        raise InvalidSequenceTokenError(token="", position=position)
    if cursor == start:
        raise InvalidSequenceTokenError(
            token=_invalid_token(text, cursor), position=position
        )
    return cursor


def _invalid_token(text: str, cursor: int) -> str:
    end = cursor
    while end < len(text) and text[end] != "," and not text[end].isspace():
        end += 1
    return text[cursor:end]
