"""Pydantic boundary schemas for versioned presets."""

import math
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from binary_entropy.constants import PROBABILITY_TOLERANCE
from binary_entropy.errors import DuplicateLabelError, ProbabilitySumError

type SchemaLabel = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[^,\r\n]+$"),
]
type SchemaProbability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
type SchemaPair = tuple[SchemaProbability, SchemaProbability]
type SchemaMatrix = tuple[SchemaPair, SchemaPair]


class PresetV1(BaseModel):
    """Strict version-one model preset."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1
    preset_name: SchemaLabel
    state_labels: tuple[SchemaLabel, SchemaLabel]
    observable_labels: tuple[SchemaLabel, SchemaLabel]
    initial: SchemaPair
    transition: SchemaMatrix
    emission: SchemaMatrix

    @model_validator(mode="after")
    def validate_model_invariants(self) -> "PresetV1":
        """Require distinct labels and stochastic probability rows."""
        if self.state_labels[0] == self.state_labels[1]:
            raise DuplicateLabelError(category="state", value=self.state_labels[0])
        if self.observable_labels[0] == self.observable_labels[1]:
            raise DuplicateLabelError(
                category="observable",
                value=self.observable_labels[0],
            )
        probability_rows = (
            ("initial", self.initial),
            ("transition row 0", self.transition[0]),
            ("transition row 1", self.transition[1]),
            ("emission row 0", self.emission[0]),
            ("emission row 1", self.emission[1]),
        )
        for name, row in probability_rows:
            total = row[0] + row[1]
            if not math.isclose(
                total,
                1.0,
                abs_tol=PROBABILITY_TOLERANCE,
                rel_tol=0.0,
            ):
                raise ProbabilitySumError(field=name, row=None, total=total)
        return self
