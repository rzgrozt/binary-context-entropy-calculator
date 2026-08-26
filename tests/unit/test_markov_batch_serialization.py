import csv
import io

from binary_entropy.domain import BinaryLabels
from binary_entropy.markov_batch_serialization import markov_batch_summary_csv
from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.markov import analyze_markov, analyze_markov_per_sequence
from binary_entropy.records import SequenceDataset, SequenceRecord


def _analysis_with_target_b() -> MarkovBatchAnalysis:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(
        labels,
        (
            SequenceRecord("first", (0, 0, 0, 1, 0), actual_target_index=1),
            SequenceRecord("second", (0, 1, 1, 1, 0)),
        ),
    )
    return analyze_markov_per_sequence(dataset)


def _dataset_with_divergent_transition_counts() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (
            SequenceRecord("first", (0, 0, 0)),
            SequenceRecord("second", (1, 0)),
        ),
    )


def test_markov_batch_summary_csv_when_serialized_has_stable_requested_columns() -> (
    None
):
    # Given
    analysis = _analysis_with_target_b()

    # When
    payload = markov_batch_summary_csv(analysis)
    rows = list(csv.reader(io.StringIO(payload)))

    # Then
    assert rows[0] == [
        "sequence_id",
        "sequence",
        "length",
        "count_A",
        "count_B",
        "AA_count",
        "AB_count",
        "BA_count",
        "BB_count",
        "P_A_given_A",
        "P_B_given_A",
        "P_A_given_B",
        "P_B_given_B",
        "last_symbol",
        "P_next_A",
        "P_next_B",
        "predicted_target",
        "predictive_entropy",
        "surprisal_A",
        "surprisal_B",
        "observed_shannon_entropy",
        "actual_target",
        "actual_target_probability",
        "actual_target_surprisal",
        "modal_or_nonmodal",
        "method",
        "result_scope",
        "prediction_mode",
        "markov_order",
        "estimation_method",
        "smoothing_alpha",
        "source_sequence_count",
        "source_transition_count",
    ]


def test_markov_batch_summary_csv_when_batch_has_two_records_writes_one_each() -> None:
    # Given
    analysis = _analysis_with_target_b()

    # When
    payload = markov_batch_summary_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert len(rows) == 2
    assert tuple(row["sequence_id"] for row in rows) == ("first", "second")


def test_markov_batch_summary_csv_when_record_is_complete_has_experimental_values() -> (
    None
):
    # Given
    analysis = _analysis_with_target_b()

    # When
    payload = markov_batch_summary_csv(analysis)
    row = next(iter(csv.DictReader(io.StringIO(payload))))

    # Then
    assert row["sequence"] == "A,A,A,B,A"
    assert (row["length"], row["count_A"], row["count_B"]) == ("5", "4", "1")
    assert (
        row["AA_count"],
        row["AB_count"],
        row["BA_count"],
        row["BB_count"],
    ) == ("2", "1", "1", "0")
    assert row["last_symbol"] == "A"
    assert row["predicted_target"] == "A"
    assert row["result_scope"] == "per_sequence"
    assert row["source_sequence_count"] == "1"
    assert row["source_transition_count"] == "4"


def test_markov_batch_summary_csv_when_per_sequence_uses_record_source_counts() -> None:
    # Given
    analysis = analyze_markov_per_sequence(_dataset_with_divergent_transition_counts())

    # When
    payload = markov_batch_summary_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert tuple(
        (
            row["sequence_id"],
            row["source_sequence_count"],
            row["source_transition_count"],
        )
        for row in rows
    ) == (("first", "1", "2"), ("second", "1", "1"))


def test_markov_batch_summary_csv_when_pooled_uses_dataset_source_counts() -> None:
    # Given
    analysis = analyze_markov(_dataset_with_divergent_transition_counts())

    # When
    payload = markov_batch_summary_csv(analysis)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert tuple(
        (
            row["sequence_id"],
            row["source_sequence_count"],
            row["source_transition_count"],
        )
        for row in rows
    ) == (("first", "2", "3"), ("second", "2", "3"))


def test_markov_batch_summary_csv_when_float_is_available_round_trips_precisely() -> (
    None
):
    # Given
    analysis = _analysis_with_target_b()

    # When
    payload = markov_batch_summary_csv(analysis)
    row = next(iter(csv.DictReader(io.StringIO(payload))))
    probability_text = row["P_next_A"]

    # Then
    assert probability_text is not None
    assert float(probability_text) == 2 / 3
    assert len(probability_text.split(".", maxsplit=1)[1]) >= 12
    assert float(row["P_A_given_A"] or "") == 2 / 3


def test_markov_batch_summary_csv_when_target_is_nonmodal_exports_evaluation() -> None:
    # Given
    analysis = _analysis_with_target_b()

    # When
    payload = markov_batch_summary_csv(analysis)
    row = next(iter(csv.DictReader(io.StringIO(payload))))

    # Then
    assert row["actual_target"] == "B"
    assert float(row["actual_target_probability"] or "") == 1 / 3
    assert row["modal_or_nonmodal"] == "nonmodal"


def test_markov_batch_summary_csv_when_model_row_is_unavailable_uses_blanks() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(labels, (SequenceRecord("short", (0,)),))
    analysis = analyze_markov_per_sequence(dataset)

    # When
    payload = markov_batch_summary_csv(analysis)
    row = next(iter(csv.DictReader(io.StringIO(payload))))

    # Then
    unavailable_columns = (
        "P_A_given_A",
        "P_B_given_A",
        "P_A_given_B",
        "P_B_given_B",
        "P_next_A",
        "P_next_B",
        "predicted_target",
        "predictive_entropy",
    )
    assert all(row[column] == "" for column in unavailable_columns)
    assert "nan" not in payload.lower()
