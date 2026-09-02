import csv
import io

from binary_entropy import (
    BinaryLabels,
    MarkovAnalysisRequest,
    analyze_dataset,
    markov_model_json,
    markov_sequence_csv,
    parse_csv_batch,
)
from binary_entropy.batch_parsing import CsvBatchColumns


def test_workbench_when_csv_batch_routes_to_markov_exports_end_to_end() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    payload = b'id,sequence,target\nalpha,"A,A,B",A\nbeta,"B,A",B\n'
    dataset = parse_csv_batch(
        payload,
        labels,
        CsvBatchColumns("id", "sequence", "target"),
    )

    # When
    routed = analyze_dataset(dataset, MarkovAnalysisRequest())

    # Then
    model_json = markov_model_json(routed)
    sequence_csv = markov_sequence_csv(routed)
    assert routed.model.transition_counts == ((1, 1), (1, 0))
    assert routed.model.source_transition_count == 3
    assert b'"source_sequence_count": 2' in model_json
    rows = list(csv.DictReader(io.StringIO(sequence_csv)))
    assert len(rows) == 7
    assert {row["sequence_id"] for row in rows} == {"alpha", "beta"}
    assert "nan" not in sequence_csv.lower()
