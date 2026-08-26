import numpy as np
import pytest

from binary_entropy.domain import BinaryLabels
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovPredictionMode,
    MarkovResultScope,
)
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.markov import (
    analyze_markov,
    analyze_markov_per_sequence,
)
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.workbench import MarkovAnalysisRequest, analyze_dataset


def _different_transition_dataset() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (
            SequenceRecord("first", (0, 0, 0, 1, 0)),
            SequenceRecord("second", (0, 1, 1, 1, 0)),
        ),
    )


def test_analyze_markov_per_sequence_when_selected_labels_independent_scope() -> None:
    # Given
    dataset = _different_transition_dataset()

    # When
    result = analyze_markov_per_sequence(dataset)

    # Then
    assert result.result_scope is MarkovResultScope.PER_SEQUENCE


def test_analyze_markov_per_sequence_when_records_differ_fits_each_record() -> None:
    # Given
    dataset = _different_transition_dataset()

    # When
    result = analyze_markov_per_sequence(dataset)

    # Then
    assert result.records[0].model.transition_counts == ((2, 1), (1, 0))
    assert result.records[1].model.transition_counts == ((0, 1), (1, 2))
    assert tuple(record.model.source_transition_count for record in result.records) == (
        4,
        4,
    )


def test_analyze_markov_per_sequence_when_fixed_uses_each_full_record_model() -> None:
    # Given
    dataset = _different_transition_dataset()

    # When
    result = analyze_markov_per_sequence(
        dataset,
        prediction_mode=MarkovPredictionMode.FIXED_MODEL,
    )

    # Then
    first_prediction = result.records[0].rows[-1].predictive
    second_prediction = result.records[1].rows[-1].predictive
    assert first_prediction is not None
    assert second_prediction is not None
    np.testing.assert_allclose(first_prediction, [2 / 3, 1 / 3], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(second_prediction, [0.0, 1.0])


def test_analyze_markov_per_sequence_when_cumulative_refits_each_record_prefix() -> (
    None
):
    # Given
    dataset = _different_transition_dataset()

    # When
    result = analyze_markov_per_sequence(
        dataset,
        prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
    )

    # Then
    first_depth_three = result.records[0].rows[3]
    second_depth_three = result.records[1].rows[3]
    assert first_depth_three.fitted_transition_count == 2
    assert second_depth_three.fitted_transition_count == 2
    assert first_depth_three.predictive is not None
    assert second_depth_three.predictive is not None
    np.testing.assert_array_equal(first_depth_three.predictive, [1.0, 0.0])
    np.testing.assert_array_equal(second_depth_three.predictive, [0.0, 1.0])


def test_analyze_markov_when_default_scope_pools_without_record_boundary() -> None:
    # Given
    dataset = _different_transition_dataset()

    # When
    result = analyze_markov(dataset)

    # Then
    assert result.result_scope is MarkovResultScope.POOLED
    assert result.model.transition_counts == ((2, 2), (2, 2))
    assert result.model.source_transition_count == 8
    for record in result.records:
        assert record.model.transition_counts == ((2, 2), (2, 2))
        final_prediction = record.rows[-1].predictive
        assert final_prediction is not None
        np.testing.assert_array_equal(final_prediction, [0.5, 0.5])


def test_analyze_dataset_when_per_sequence_scope_requested_routes_independent_fit() -> (
    None
):
    # Given
    dataset = _different_transition_dataset()
    request = MarkovAnalysisRequest(result_scope=MarkovResultScope.PER_SEQUENCE)

    # When
    result = analyze_dataset(dataset, request)

    # Then
    match result:
        case MarkovBatchAnalysis(result_scope=result_scope):
            assert result_scope is MarkovResultScope.PER_SEQUENCE
        case HMMBatchAnalysis() | ShannonBatchAnalysis():
            pytest.fail("per-sequence Markov request routed to a different method")
