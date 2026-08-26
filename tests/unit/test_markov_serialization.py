import csv
import hashlib
import io
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict

from binary_entropy.domain import BinaryLabels
from binary_entropy.markov_serialization import (
    markov_model_json,
    markov_sequence_csv,
)
from binary_entropy.markov_types import MarkovPredictionMode
from binary_entropy.methods.markov import analyze_markov, analyze_markov_per_sequence
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord


class _MarkovJsonView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal[1]
    method: Literal["markov"]
    observable_labels: tuple[str, str]
    markov_order: Literal[1]
    estimation_method: str
    smoothing_alpha: float
    transition_counts: tuple[tuple[int, int], tuple[int, int]]
    transition_matrix: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]
    starting_distribution: tuple[float, float] | None
    stationary_distribution: tuple[float, float] | None
    source_sequence_count: int
    source_transition_count: int


class _PerSequenceMarkovModelJsonView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    sequence_id: str
    estimation_method: str
    smoothing_alpha: float
    transition_counts: tuple[tuple[int, int], tuple[int, int]]
    transition_matrix: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]
    starting_distribution: tuple[float, float] | None
    source_sequence_count: int
    source_transition_count: int


class _PerSequenceMarkovJsonView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    method: Literal["markov"]
    observable_labels: tuple[str, str]
    markov_order: Literal[1]
    prediction_mode: str
    result_scope: Literal["per_sequence"]
    models: tuple[_PerSequenceMarkovModelJsonView, ...]


def _dataset(
    sequences: tuple[BinarySequence, ...],
    actual_target: Literal[0, 1] | None = None,
) -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    records = tuple(
        SequenceRecord(
            f"record-{index}",
            sequence,
            actual_target if index == 1 else None,
        )
        for index, sequence in enumerate(sequences, start=1)
    )
    return SequenceDataset(labels, records)


def test_markov_model_json_when_fit_is_complete_contains_required_fields() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 0, 1, 0, 1, 1),)))

    # When
    payload = markov_model_json(analysis)
    result = _MarkovJsonView.model_validate_json(payload, strict=True)

    # Then
    assert result.observable_labels == ("A", "B")
    assert result.transition_counts == ((1, 2), (1, 1))
    assert result.transition_matrix == ((1 / 3, 2 / 3), (1 / 2, 1 / 2))
    assert result.starting_distribution == (1.0, 0.0)
    assert result.stationary_distribution is not None
    assert result.stationary_distribution == pytest.approx((3 / 7, 4 / 7), abs=1e-15)
    assert result.source_sequence_count == 1
    assert result.source_transition_count == 5


def test_markov_model_json_when_serialized_is_stable_and_has_no_raw_sequences() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 1), (1, 0))))

    # When
    first = markov_model_json(analysis)
    second = markov_model_json(analysis)

    # Then
    assert first == second
    assert first.endswith(b"\n")
    assert b'"sequence":' not in first


def test_markov_model_json_when_scope_is_pooled_preserves_schema_v1_bytes() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 0),)))

    # When
    payload = markov_model_json(analysis)

    # Then
    assert (
        hashlib.sha256(payload).hexdigest()
        == "c9b31aa32de4cfd75dfc30ceee2aa784fb4f6b26c1cf81832043268984c9d00f"
    )


def test_markov_model_json_when_scope_is_per_sequence_uses_ordered_record_models() -> (
    None
):
    # Given
    analysis = analyze_markov_per_sequence(_dataset(((0, 0, 0), (1, 0))))

    # When
    payload = markov_model_json(analysis)
    result = _PerSequenceMarkovJsonView.model_validate_json(payload, strict=True)

    # Then
    assert tuple(model.sequence_id for model in result.models) == (
        "record-1",
        "record-2",
    )
    assert tuple(model.transition_counts for model in result.models) == (
        ((2, 0), (0, 0)),
        ((0, 0), (1, 0)),
    )
    assert tuple(model.transition_matrix for model in result.models) == (
        ((1.0, 0.0), None),
        (None, (1.0, 0.0)),
    )
    assert tuple(model.starting_distribution for model in result.models) == (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    assert tuple(model.source_sequence_count for model in result.models) == (1, 1)
    assert tuple(model.source_transition_count for model in result.models) == (2, 1)
    assert payload == markov_model_json(analysis)
    assert payload.endswith(b"\n")
    assert b'"sequence":' not in payload
    assert b'"stationary_distribution"' not in payload
    top_level_keys = (
        b'"schema_version"',
        b'"method"',
        b'"observable_labels"',
        b'"markov_order"',
        b'"prediction_mode"',
        b'"result_scope"',
        b'"models"',
    )
    positions = tuple(payload.index(key) for key in top_level_keys)
    assert positions == tuple(sorted(positions))


def test_markov_model_json_when_mle_row_is_missing_uses_json_null() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 0),)))

    # When
    payload = markov_model_json(analysis)
    result = _MarkovJsonView.model_validate_json(payload, strict=True)

    # Then
    assert result.transition_matrix[0] == (1.0, 0.0)
    assert result.transition_matrix[1] is None
    assert result.stationary_distribution is None


def test_markov_sequence_csv_when_batch_is_analyzed_has_stable_requested_columns() -> (
    None
):
    # Given
    analysis = analyze_markov(_dataset(((0, 1), (1,))))

    # When
    payload = markov_sequence_csv(analysis)
    rows = list(csv.reader(io.StringIO(payload)))

    # Then
    assert rows[0] == [
        "sequence_id",
        "method",
        "result_scope",
        "prediction_mode",
        "markov_order",
        "estimation_method",
        "smoothing_alpha",
        "source_sequence_count",
        "source_transition_count",
        "sequence_length",
        "depth",
        "context_symbol",
        "observed_next_symbol",
        "fitted_transition_count",
        "predictive_probability_A",
        "predictive_probability_B",
        "predictive_entropy_bits",
        "predicted_symbol",
        "actual_target_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "actual_target_classification",
    ]
    assert len(rows) == 6


def test_markov_sequence_csv_when_scope_is_per_sequence_uses_each_record_model() -> (
    None
):
    # Given
    analysis = analyze_markov_per_sequence(
        _dataset(((0, 0, 0), (1, 0))),
        prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
    )

    # When
    payload = markov_sequence_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    first_rows = tuple(row for row in rows if row["sequence_id"] == "record-1")
    second_rows = tuple(row for row in rows if row["sequence_id"] == "record-2")
    assert {row["result_scope"] for row in rows} == {"per_sequence"}
    assert {row["markov_order"] for row in rows} == {"1"}
    assert {row["estimation_method"] for row in rows} == {"maximum_likelihood"}
    assert {row["smoothing_alpha"] for row in rows} == {"0.000000000000"}
    assert {row["source_sequence_count"] for row in first_rows} == {"1"}
    assert {row["source_transition_count"] for row in first_rows} == {"2"}
    assert tuple(row["fitted_transition_count"] for row in first_rows) == (
        "0",
        "0",
        "1",
        "2",
    )
    assert {row["source_sequence_count"] for row in second_rows} == {"1"}
    assert {row["source_transition_count"] for row in second_rows} == {"1"}
    assert tuple(row["fitted_transition_count"] for row in second_rows) == (
        "0",
        "0",
        "1",
    )


def test_markov_sequence_csv_when_values_are_available_round_trips_float64() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 0, 1, 0, 1, 1),)))

    # When
    payload = markov_sequence_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))
    probability_text = rows[1]["predictive_probability_A"]

    # Then
    assert probability_text is not None
    assert float(probability_text) == 1 / 3
    assert len(probability_text.split(".", maxsplit=1)[1]) >= 12


def test_markov_sequence_csv_when_prediction_is_unavailable_uses_blank_not_nan() -> (
    None
):
    # Given
    analysis = analyze_markov(
        _dataset(((0, 0),)),
        prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
    )

    # When
    payload = markov_sequence_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert rows[0]["predictive_probability_A"] == ""
    assert rows[1]["predictive_probability_A"] == ""
    assert "nan" not in payload.lower()


def test_markov_sequence_csv_when_target_exists_only_final_row_evaluates() -> None:
    # Given
    analysis = analyze_markov(_dataset(((0, 0, 1, 0, 1, 1),), actual_target=0))

    # When
    payload = markov_sequence_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert all(row["actual_target_symbol"] == "" for row in rows[:-1])
    assert rows[-1]["actual_target_symbol"] == "A"
    assert rows[-1]["actual_target_probability"] == "0.500000000000"
    assert rows[-1]["actual_target_classification"] == "tied"
