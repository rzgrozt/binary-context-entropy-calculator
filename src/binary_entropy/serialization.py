"""Strict preset and deterministic CSV serialization boundaries."""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from binary_entropy.analysis import assess_target
from binary_entropy.domain import (
    BinaryHMM,
    BinaryLabels,
    ObservableIndex,
    SequenceAnalysis,
    float_values,
)
from binary_entropy.errors import PresetDecodeError, PresetSchemaError
from binary_entropy.presentation import CellValue, analysis_table, format_decimal
from binary_entropy.schemas import PresetV1


@dataclass(frozen=True, slots=True)
class CandidateMetadata:
    """Identifiers and optional next target for one candidate export."""

    sequence_id: str
    preset_name: str
    actual_target_index: ObservableIndex | None = None


def parse_preset_json(payload: str | bytes) -> PresetV1:
    """Decode and validate one UTF-8 version-one preset transactionally."""
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PresetDecodeError(detail=str(error)) from error
    else:
        text = payload
    try:
        return PresetV1.model_validate_json(text, strict=True)
    except ValidationError as error:
        detail = str(error)
        if "Invalid JSON" in detail:
            raise PresetDecodeError(detail=detail) from error
        raise PresetSchemaError(detail=detail) from error


def preset_json(preset: PresetV1) -> bytes:
    """Serialize one preset as stable UTF-8 JSON."""
    text = preset.model_dump_json(indent=2) + "\n"
    return text.encode("utf-8", errors="strict")


def model_from_preset(preset: PresetV1) -> BinaryHMM:
    """Convert a validated boundary preset to an immutable model."""
    return BinaryHMM(
        labels=BinaryLabels(
            states=preset.state_labels,
            observables=preset.observable_labels,
        ),
        initial=preset.initial,
        transition=preset.transition,
        emission=preset.emission,
    )


def preset_from_model(model: BinaryHMM, preset_name: str) -> PresetV1:
    """Convert an immutable model to a version-one boundary preset."""
    initial_0, initial_1 = float_values(model.initial)
    transition_00, transition_01, transition_10, transition_11 = float_values(
        model.transition
    )
    emission_00, emission_01, emission_10, emission_11 = float_values(model.emission)
    return PresetV1(
        preset_name=preset_name,
        state_labels=model.labels.states,
        observable_labels=model.labels.observables,
        initial=(initial_0, initial_1),
        transition=(
            (transition_00, transition_01),
            (transition_10, transition_11),
        ),
        emission=(
            (emission_00, emission_01),
            (emission_10, emission_11),
        ),
    )


def prefix_csv(analysis: SequenceAnalysis, model: BinaryHMM) -> str:
    """Serialize deterministic prefix results as CSV."""
    table = analysis_table(analysis, model)
    return _csv_text(table.columns, table.rows)


def candidate_summary_csv(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
    metadata: CandidateMetadata,
) -> str:
    """Serialize one deterministic candidate summary row."""
    final = analysis.rows[-1]
    initial_0, initial_1 = float_values(model.initial)
    transition_values = float_values(model.transition)
    emission_values = float_values(model.emission)
    final_probability_0, final_probability_1 = float_values(final.predictive)
    target = (
        assess_target(final.predictive, metadata.actual_target_index)
        if metadata.actual_target_index is not None
        else None
    )
    columns = (
        "sequence_id",
        "preset_name",
        "state_label_0",
        "state_label_1",
        "observable_label_0",
        "observable_label_1",
        "sequence",
        "initial_0",
        "initial_1",
        "transition_00",
        "transition_01",
        "transition_10",
        "transition_11",
        "emission_00",
        "emission_01",
        "emission_10",
        "emission_11",
        "sequence_length",
        "observed_entropy_bits",
        "final_predictive_probability_0",
        "final_predictive_probability_1",
        "final_predictive_entropy_bits",
        "final_predicted_symbol",
        "actual_target_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "actual_target_classification",
    )
    target_label = (
        model.labels.observables[target.actual_target_index]
        if target is not None
        else None
    )
    row: tuple[CellValue, ...] = (
        metadata.sequence_id,
        metadata.preset_name,
        *model.labels.states,
        *model.labels.observables,
        ",".join(model.labels.observables[index] for index in analysis.sequence),
        initial_0,
        initial_1,
        *transition_values,
        *emission_values,
        len(analysis.sequence),
        analysis.observed_entropy_bits,
        final_probability_0,
        final_probability_1,
        final.entropy_bits,
        model.labels.observables[final.predicted_index],
        target_label,
        target.probability if target is not None else None,
        target.surprisal_bits if target is not None else None,
        target.classification.value if target is not None else None,
    )
    return _csv_text(columns, (row,))


def _csv_text(
    columns: Sequence[str],
    rows: Sequence[Sequence[CellValue]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(tuple(_formatted_cell(cell) for cell in row))
    return buffer.getvalue()


def _formatted_cell(value: CellValue) -> str:
    if value is None:
        return ""
    if type(value) is float:
        return format_decimal(value)
    return str(value)
