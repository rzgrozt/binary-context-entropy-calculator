"""Presentation metadata applied after statistical analysis."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from bspe.stimulus_search_results import StimulusCandidate
from bspe.stimulus_search_types import InvalidStimulusSearchConfigurationError


@dataclass(frozen=True, slots=True)
class PresentationMetadata:
    """Labels and optional condition; never inputs to probability estimation."""

    symbol_mapping: tuple[str, str] = ("A", "B")
    condition: str | None = None

    def __post_init__(self) -> None:
        """Require distinct nonempty labels and an unambiguous export separator."""
        labels = tuple(label.strip() for label in self.symbol_mapping)
        if (
            len(labels) != 2
            or not all(labels)
            or labels[0] == labels[1]
            or any(character in label for label in labels for character in "|\n\r")
        ):
            raise InvalidStimulusSearchConfigurationError(
                "symbol_mapping", repr(self.symbol_mapping)
            )
        object.__setattr__(self, "symbol_mapping", labels)
        object.__setattr__(self, "condition", (self.condition or "").strip() or None)


def apply_presentation_metadata(
    candidates: Sequence[StimulusCandidate], metadata: PresentationMetadata
) -> tuple[StimulusCandidate, ...]:
    """Copy metadata while preserving canonical sequences and analysis values."""
    return tuple(
        replace(
            candidate,
            symbol_mapping=metadata.symbol_mapping,
            condition=metadata.condition or candidate.condition,
        )
        for candidate in candidates
    )
