import pytest

from binary_entropy.domain import BinaryLabels
from binary_entropy.errors import InvalidSequenceTokenError
from binary_entropy.parsing import parse_sequence


def test_parse_sequence_when_text_has_surrounding_whitespace() -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))

    result = parse_sequence(" A, B ,B,A ", labels)

    assert result == (0, 1, 1, 0)


@pytest.mark.parametrize(
    "text",
    [
        "A B B A",
        "A\nB\nB\nA",
        " A,\n B\tB, A ",
    ],
)
def test_parse_sequence_when_separator_styles_vary(text: str) -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))

    result = parse_sequence(text, labels)

    assert result == (0, 1, 1, 0)


def test_parse_sequence_when_labels_contain_spaces_matches_each_whole_label() -> None:
    labels = BinaryLabels(
        states=("S1", "S2"),
        observables=("light red", "deep blue"),
    )

    result = parse_sequence("light red deep blue, light red", labels)

    assert result == (0, 1, 0)


def test_parse_sequence_when_one_label_prefixes_another_prefers_longest_match() -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("light", "light red"))

    result = parse_sequence("light red light", labels)

    assert result == (1, 0)


def test_parse_sequence_when_outer_separators_repeat_ignores_them() -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))

    result = parse_sequence(",, \nA B\t,,", labels)

    assert result == (0, 1)


@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_parse_sequence_when_text_is_blank(text: str) -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))

    result = parse_sequence(text, labels)

    assert result == ()


def test_parse_sequence_when_token_is_invalid_reports_first_one_based_position() -> (
    None
):
    labels = BinaryLabels(states=("S1", "S2"), observables=("yes", "no"))
    expected = "maybe"

    with pytest.raises(InvalidSequenceTokenError) as captured:
        _ = parse_sequence("yes, maybe, unknown", labels)

    assert captured.value.token == expected
    assert captured.value.position == 2


@pytest.mark.parametrize("text", ["A,,B", "A, ,B"])
def test_parse_sequence_when_empty_token_occurs(text: str) -> None:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))

    with pytest.raises(InvalidSequenceTokenError) as captured:
        _ = parse_sequence(text, labels)

    assert captured.value.token == ""
    assert captured.value.position == 2
