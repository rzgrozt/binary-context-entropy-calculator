import math

import pytest
import streamlit as st

from binary_entropy.domain import BinaryLabels
from binary_entropy.methods.vmm import analyze_vmm
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT
from binary_entropy.ui.vmm_results import (
    vmm_depth_column_config,
    vmm_depth_dataframe,
    vmm_final_column_config,
    vmm_record_dataframe,
)
from binary_entropy.vmm_types import VMMConfig


def _recurrent_dataset() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (
            SequenceRecord(
                "recurrent-aa",
                (0, 0, 1, 0, 0, 1, 0, 0),
                actual_target_index=1,
            ),
        ),
    )


def test_vmm_final_dataframe_when_context_is_supported_preserves_raw_values() -> None:
    # Given
    dataset = _recurrent_dataset()
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))

    # When
    frame = vmm_record_dataframe(analysis, dataset)

    # Then
    assert frame.loc[0, "Sequence ID"] == "recurrent-aa"
    assert frame.loc[0, "Effective predictive context depth"] == 2
    assert frame.loc[0, "Actual context used"] == "A, A"
    assert frame.loc[0, "Support count"] == 2
    assert frame.loc[0, "P(next B)"] == 2.5 / 3.0
    assert frame.loc[0, "Prediction"] == "B"
    assert frame.loc[0, "Actual target"] == "B"
    assert frame.loc[0, "Actual-target probability"] == 2.5 / 3.0
    assert frame.loc[0, "Evaluation status"] == ("In-sample evaluation, not held out")


def test_vmm_depth_dataframe_when_rendered_includes_every_depth_in_order() -> None:
    # Given
    dataset = _recurrent_dataset()
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))
    record = analysis.records[0]

    # When
    frame = vmm_depth_dataframe(analysis, dataset, record)

    # Then
    assert list(frame.columns) == [
        "Dataset role",
        "Record ID",
        "Workflow",
        "Result scope",
        "Requested depth",
        "Actual depth",
        "Context",
        "Context occurrence count",
        "Next A count",
        "Next B count",
        "Support rule",
        "Support status",
        "Sparse status",
        "Estimation rule",
        "Smoothing alpha",
        "Suffix-backoff selection",
        "Suffix-backoff reason",
        "P(next A)",
        "P(next B)",
        "Predictive entropy (bits)",
        "Evaluation status",
        "Observed target",
        "Target probability",
        "Target surprisal (bits)",
    ]
    assert frame["Requested depth"].tolist() == list(range(9))
    assert frame["Actual depth"].tolist() == [2] * 9
    assert frame.loc[0, "Context"] == "Order 0 (no suffix)"
    assert frame.loc[2, "Context"] == "A, A"
    assert frame.loc[2, "Next A count"] == 0
    assert frame.loc[2, "Next B count"] == 2
    assert frame.loc[2, "P(next B)"] == 2.5 / 3.0
    assert frame.loc[2, "Support status"] == "accepted"
    assert frame.loc[2, "Sparse status"] == "not_sparse"
    assert frame.loc[2, "Suffix-backoff selection"] == "selected"
    assert frame.loc[2, "Evaluation status"] == ("In-sample evaluation, not held out")
    assert frame.loc[2, "Observed target"] == "B"
    assert frame.loc[2, "Target probability"] == 2.5 / 3.0
    assert frame.loc[0, "Evaluation status"] == ("In-sample evaluation, not held out")
    assert frame.loc[0, "Observed target"] == "B"
    assert frame.loc[0, "Target probability"] == pytest.approx(5 / 18)
    assert frame.loc[0, "Target surprisal (bits)"] == pytest.approx(
        -math.log2(5 / 18)
    )
    assert frame.loc[2, "Target surprisal (bits)"] == pytest.approx(
        -math.log2(5 / 6)
    )
    assert frame.loc[6, "Support status"] == "unavailable"
    assert frame.loc[6, "Evaluation status"] == "In-sample evaluation unavailable"
    assert frame.loc[6, "Observed target"] == "B"
    assert bool(
        frame.loc[
            6,
            ["Target probability", "Target surprisal (bits)"],
        ].isna().all()
    )
    assert frame.loc[8, "Support status"] == "unavailable"
    assert frame.loc[8, "Suffix-backoff selection"] == "rejected_for_backoff"
    assert frame.loc[8, "Evaluation status"] == "In-sample evaluation unavailable"


def test_vmm_depth_dataframe_when_using_kt_exposes_raw_configuration_values() -> None:
    # Given
    dataset = _recurrent_dataset()
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))

    # When
    frame = vmm_depth_dataframe(analysis, dataset, analysis.records[0])

    # Then
    assert set(frame["Dataset role"]) == {"training"}
    assert set(frame["Record ID"]) == {"recurrent-aa"}
    assert set(frame["Workflow"]) == {"variable_order_markov"}
    assert set(frame["Result scope"]) == {"pooled"}
    assert set(frame["Support rule"]) == {"minimum_support=2"}
    assert set(frame["Estimation rule"]) == {"krichevsky_trofimov"}
    assert set(frame["Smoothing alpha"]) == {0.5}


def test_vmm_final_dataframe_when_prediction_is_unavailable_names_the_state() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(
        labels,
        (SequenceRecord("empty", (), actual_target_index=1),),
    )
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))

    # When
    frame = vmm_record_dataframe(analysis, dataset)

    # Then
    assert frame.loc[0, "Effective predictive context depth"] == "Unavailable"
    assert frame.loc[0, "Actual context used"] == "Unavailable"
    assert frame.loc[0, "Prediction"] == "Unavailable"
    assert frame.loc[0, "P(next A)"] is None
    assert frame.loc[0, "Actual target"] == "B"
    assert frame.loc[0, "Evaluation status"] == "In-sample evaluation unavailable"


def test_vmm_dataframe_columns_when_configured_use_three_decimal_display() -> None:
    # Given
    labels = ("A", "B")
    expected = st.column_config.NumberColumn(format=UI_NUMBER_FORMAT)

    # When
    final_config = vmm_final_column_config(labels)
    depth_config = vmm_depth_column_config(labels)

    # Then
    for column in (
        "P(next A)",
        "P(next B)",
        "Predictive Shannon entropy (bits)",
        "Surprisal of A (bits)",
        "Surprisal of B (bits)",
        "Actual-target probability",
        "Actual-target surprisal (bits)",
    ):
        assert final_config[column] == expected
    for column in (
        "P(next A)",
        "P(next B)",
        "Predictive entropy (bits)",
        "Smoothing alpha",
        "Target probability",
        "Target surprisal (bits)",
    ):
        assert depth_config[column] == expected
