import numpy as np
import pytest

from binary_entropy.analysis import analyze_sequence
from binary_entropy.domain import BinaryLabels, ObservableIndex
from binary_entropy.markov_types import MarkovBatchAnalysis, MarkovPredictionMode
from binary_entropy.methods.hmm import HMMBatchAnalysis, analyze_hmm
from binary_entropy.methods.shannon import ShannonBatchAnalysis, analyze_shannon
from binary_entropy.records import BinarySequence, SequenceDataset, SequenceRecord
from binary_entropy.workbench import (
    AnalysisMethod,
    HMMAnalysisRequest,
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
    analyze_dataset,
    compare_methods,
)
from tests.unit.helpers import hand_model


def _dataset(
    sequences: tuple[BinarySequence, ...],
    targets: tuple[ObservableIndex | None, ...] | None = None,
) -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    selected_targets = targets or tuple(None for _ in sequences)
    records = tuple(
        SequenceRecord(f"record-{index}", sequence, target)
        for index, (sequence, target) in enumerate(
            zip(sequences, selected_targets, strict=True),
            start=1,
        )
    )
    return SequenceDataset(labels, records)


def test_analyze_shannon_when_records_vary_returns_pooled_and_per_record() -> None:
    # Given
    dataset = _dataset(((0, 0, 1), (1,), ()))

    # When
    result = analyze_shannon(dataset)

    # Then
    assert result.pooled.observation_count == 4
    assert result.pooled.symbol_counts == (2, 2)
    assert result.pooled.entropy_bits == 1.0
    assert len(result.records) == 3
    assert result.records[0].summary.symbol_counts == (2, 1)
    assert result.records[0].summary.entropy_bits == pytest.approx(
        0.9182958340544896,
        abs=1e-15,
    )
    assert result.records[2].summary.symbol_probabilities is None
    assert result.records[2].summary.entropy_bits is None


def test_analyze_shannon_when_descriptive_has_no_predictive_fields() -> None:
    # Given
    dataset = _dataset(((0, 1),))

    # When
    result = analyze_shannon(dataset)

    # Then
    assert not hasattr(result.pooled, "predictive")
    assert not hasattr(result.records[0].summary, "predicted_index")


def test_analyze_shannon_when_probabilities_exist_owns_read_only_float64() -> None:
    # Given
    dataset = _dataset(((0, 0, 1),))

    # When
    result = analyze_shannon(dataset)

    # Then
    probabilities = result.records[0].summary.symbol_probabilities
    assert probabilities is not None
    assert probabilities.dtype == np.float64
    assert not probabilities.flags.writeable
    np.testing.assert_allclose(probabilities, [2 / 3, 1 / 3], atol=0.0, rtol=0.0)


def test_analyze_shannon_when_target_changes_keeps_descriptive_results() -> None:
    # Given
    target_a = _dataset(((0, 0, 1),), (0,))
    target_b = _dataset(((0, 0, 1),), (1,))

    # When
    result_a = analyze_shannon(target_a)
    result_b = analyze_shannon(target_b)

    # Then
    assert result_a.pooled.symbol_counts == result_b.pooled.symbol_counts == (2, 1)
    assert result_a.pooled.entropy_bits == result_b.pooled.entropy_bits


def test_analyze_hmm_when_dataset_has_two_records_calls_legacy_per_sequence() -> None:
    # Given
    model = hand_model()
    dataset = _dataset(((0,), (1, 0)))
    expected = (
        analyze_sequence(model, (0,)),
        analyze_sequence(model, (1, 0)),
    )

    # When
    result = analyze_hmm(dataset, model)

    # Then
    assert tuple(len(record.analysis.rows) for record in result.records) == (2, 3)
    for actual, expected_analysis in zip(result.records, expected, strict=True):
        assert actual.analysis.sequence == expected_analysis.sequence
        np.testing.assert_array_equal(
            actual.analysis.rows[-1].predictive,
            expected_analysis.rows[-1].predictive,
        )


def test_analyze_hmm_when_target_changes_only_external_assessment_changes() -> None:
    # Given
    model = hand_model()
    target_a = _dataset(((0, 1),), (0,))
    target_b = _dataset(((0, 1),), (1,))

    # When
    result_a = analyze_hmm(target_a, model)
    result_b = analyze_hmm(target_b, model)

    # Then
    np.testing.assert_array_equal(
        result_a.records[0].analysis.rows[-1].predictive,
        result_b.records[0].analysis.rows[-1].predictive,
    )
    assert result_a.records[0].target_assessment is not None
    assert result_b.records[0].target_assessment is not None
    assert (
        result_a.records[0].target_assessment.actual_target_index
        != result_b.records[0].target_assessment.actual_target_index
    )


@pytest.mark.parametrize(
    ("analysis_request", "expected_method", "expected_type"),
    [
        (HMMAnalysisRequest(hand_model()), AnalysisMethod.HMM, HMMBatchAnalysis),
        (
            MarkovAnalysisRequest(
                smoothing_alpha=1.0,
                prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
            ),
            AnalysisMethod.MARKOV,
            MarkovBatchAnalysis,
        ),
        (
            ShannonAnalysisRequest(),
            AnalysisMethod.OBSERVED_SHANNON,
            ShannonBatchAnalysis,
        ),
    ],
)
def test_analyze_dataset_when_request_is_typed_routes_matching_method(
    analysis_request: HMMAnalysisRequest
    | MarkovAnalysisRequest
    | ShannonAnalysisRequest,
    expected_method: AnalysisMethod,
    expected_type: type[HMMBatchAnalysis | MarkovBatchAnalysis | ShannonBatchAnalysis],
) -> None:
    # Given
    dataset = _dataset(((0, 1, 0),))

    # When
    result = analyze_dataset(dataset, analysis_request)

    # Then
    assert result.method == expected_method
    assert type(result) is expected_type


def test_compare_methods_when_three_requests_preserves_labeled_order() -> None:
    # Given
    dataset = _dataset(((0, 1, 0),))
    requests = (
        HMMAnalysisRequest(hand_model()),
        MarkovAnalysisRequest(),
        ShannonAnalysisRequest(),
    )

    # When
    result = compare_methods(dataset, requests)

    # Then
    assert tuple(item.method for item in result.results) == (
        AnalysisMethod.HMM,
        AnalysisMethod.MARKOV,
        AnalysisMethod.OBSERVED_SHANNON,
    )
