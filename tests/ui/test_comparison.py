from dataclasses import dataclass

import pytest

from binary_entropy.domain import BinaryHMM, BinaryLabels
from binary_entropy.markov_types import MarkovResultScope
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.comparison import comparison_dataframe
from binary_entropy.vmm_types import VMMConfig, VMMResultScope
from binary_entropy.workbench import (
    HMMAnalysisRequest,
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
    VMMAnalysisRequest,
    WorkbenchResult,
    compare_methods,
)


@dataclass(frozen=True, slots=True)
class VMMComparisonCase:
    scope: VMMResultScope
    scope_label: str
    probability_a: str
    probability_b: str
    prediction: str
    entropy: str


def _batch_inputs() -> tuple[SequenceDataset, BinaryHMM]:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(
        labels,
        (
            SequenceRecord("record-zeta", (0, 0, 0, 1, 0), 1),
            SequenceRecord("record-alpha", (0, 1, 1, 1, 1, 1), 1),
        ),
    )
    model = BinaryHMM(
        labels,
        initial=(0.6, 0.4),
        transition=((0.7, 0.3), (0.2, 0.8)),
        emission=((0.9, 0.1), (0.2, 0.8)),
    )
    return dataset, model


@pytest.fixture
def three_method_results() -> tuple[WorkbenchResult, ...]:
    dataset, model = _batch_inputs()
    return compare_methods(
        dataset,
        (
            MarkovAnalysisRequest(),
            HMMAnalysisRequest(model),
            ShannonAnalysisRequest(),
        ),
    ).results


def test_comparison_dataframe_when_batch_maps_each_method_to_each_record(
    three_method_results: tuple[WorkbenchResult, ...],
) -> None:
    # Given
    expected_columns = (
        "Sequence ID",
        "Method",
        "Scope",
        "P(next A)",
        "P(next B)",
        "Prediction",
        "Target probability",
        "Target surprisal (bits)",
        "Predictive entropy (bits)",
        "Observed entropy (bits)",
    )
    expected_rows = (
        (
            "record-zeta",
            "Markov Chain",
            "Pooled fit; per-sequence prediction",
            "0.500",
            "0.500",
            "A",
            "0.500",
            "1.000",
            "1.000",
            "N/A",
        ),
        (
            "record-zeta",
            "Hidden Markov Model",
            "Configured model; per-sequence filtering",
            "0.568",
            "0.432",
            "A",
            "0.432",
            "1.210",
            "0.987",
            "N/A",
        ),
        (
            "record-zeta",
            "Observed Shannon Entropy",
            "Per-sequence descriptive",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "0.722",
        ),
        (
            "record-alpha",
            "Markov Chain",
            "Pooled fit; per-sequence prediction",
            "0.200",
            "0.800",
            "B",
            "0.800",
            "0.322",
            "0.722",
            "N/A",
        ),
        (
            "record-alpha",
            "Hidden Markov Model",
            "Configured model; per-sequence filtering",
            "0.352",
            "0.648",
            "B",
            "0.648",
            "0.625",
            "0.936",
            "N/A",
        ),
        (
            "record-alpha",
            "Observed Shannon Entropy",
            "Per-sequence descriptive",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "0.650",
        ),
    )

    # When
    frame = comparison_dataframe(three_method_results, ("A", "B"))

    # Then
    assert tuple(frame.columns) == expected_columns
    assert tuple(frame.itertuples(index=False, name=None)) == expected_rows


def test_comparison_dataframe_when_markov_fit_is_per_sequence_labels_scope() -> None:
    # Given
    dataset, model = _batch_inputs()
    results = compare_methods(
        dataset,
        (
            MarkovAnalysisRequest(result_scope=MarkovResultScope.PER_SEQUENCE),
            HMMAnalysisRequest(model),
            ShannonAnalysisRequest(),
        ),
    ).results

    # When
    frame = comparison_dataframe(results, ("A", "B"))

    # Then
    assert tuple(frame["Scope"]) == (
        "Per-sequence fit and prediction",
        "Configured model; per-sequence filtering",
        "Per-sequence descriptive",
        "Per-sequence fit and prediction",
        "Configured model; per-sequence filtering",
        "Per-sequence descriptive",
    )


@pytest.mark.parametrize(
    "case",
    [
        VMMComparisonCase(
            scope=VMMResultScope.POOLED,
            scope_label="Pooled fit; per-sequence prediction",
            probability_a="0.500",
            probability_b="0.500",
            prediction="Tie",
            entropy="1.000",
        ),
        VMMComparisonCase(
            scope=VMMResultScope.PER_SEQUENCE,
            scope_label="Per-sequence fit and prediction",
            probability_a="0.625",
            probability_b="0.375",
            prediction="A",
            entropy="0.954",
        ),
    ],
)
def test_comparison_dataframe_when_vmm_is_present_maps_predictive_fields(
    case: VMMComparisonCase,
) -> None:
    # Given
    dataset, _ = _batch_inputs()
    results = compare_methods(
        dataset,
        (
                VMMAnalysisRequest(
                    config=VMMConfig(minimum_support=2),
                    result_scope=case.scope,
            ),
        ),
    ).results

    # When
    frame = comparison_dataframe(results, ("A", "B"))

    # Then
    assert frame["Method"].tolist() == [
        "Variable-order Markov",
        "Variable-order Markov",
    ]
    assert frame["Scope"].tolist() == [case.scope_label, case.scope_label]
    assert frame.loc[0, "P(next A)"] == case.probability_a
    assert frame.loc[0, "P(next B)"] == case.probability_b
    assert frame.loc[0, "Prediction"] == case.prediction
    assert frame.loc[0, "Predictive entropy (bits)"] == case.entropy
