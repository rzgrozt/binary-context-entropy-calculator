from pathlib import Path
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict

from binary_entropy.domain import BinaryHMM, BinaryLabels, ObservableIndex

FIXTURE_PATH: Final = Path(__file__).parents[1] / "fixtures" / "hand_sequence.json"


class ExpectedRow(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    depth: int
    context: str
    observed_index: Literal[0, 1] | None
    posterior: tuple[float, float] | None
    next_hidden: tuple[float, float]
    predictive: tuple[float, float]
    entropy_bits: float
    predicted_index: Literal[0, 1]
    surprisal_bits: tuple[float, float]


class HandFixture(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    preset_name: str
    state_labels: tuple[str, str]
    observable_labels: tuple[str, str]
    initial: tuple[float, float]
    transition: tuple[tuple[float, float], tuple[float, float]]
    emission: tuple[tuple[float, float], tuple[float, float]]
    sequence: tuple[Literal[0, 1], ...]
    observed_entropy_bits: float
    rows: tuple[ExpectedRow, ...]


def load_hand_fixture() -> HandFixture:
    payload = FIXTURE_PATH.read_text(encoding="utf-8")
    return HandFixture.model_validate_json(payload, strict=True)


def hand_model() -> BinaryHMM:
    fixture = load_hand_fixture()
    return BinaryHMM(
        labels=BinaryLabels(
            states=fixture.state_labels,
            observables=fixture.observable_labels,
        ),
        initial=fixture.initial,
        transition=fixture.transition,
        emission=fixture.emission,
    )


def hand_sequence() -> tuple[ObservableIndex, ...]:
    fixture = load_hand_fixture()
    return fixture.sequence
