import csv
import io

import pytest

from binary_entropy import (
    BinaryLabels,
    MarkovAnalysisRequest,
    MarkovBatchAnalysis,
    analyze_dataset,
    markov_model_json,
    markov_sequence_csv,
    parse_csv_batch,
)
from binary_entropy.batch_parsing import CsvBatchColumns
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis


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
    match routed:
        case MarkovBatchAnalysis() as result:
            model_json = markov_model_json(result)
            sequence_csv = markov_sequence_csv(result)
        case HMMBatchAnalysis() | ShannonBatchAnalysis():
            pytest.fail("Markov request routed to a different scientific method")
    assert result.model.transition_counts == ((1, 1), (1, 0))
    assert result.model.source_transition_count == 3
    assert b'"source_sequence_count": 2' in model_json
    rows = list(csv.DictReader(io.StringIO(sequence_csv)))
    assert len(rows) == 7
    assert {row["sequence_id"] for row in rows} == {"alpha", "beta"}
    assert "nan" not in sequence_csv.lower()
