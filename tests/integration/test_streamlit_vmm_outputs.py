import math

import pytest

from binary_entropy.ui.state import ActualTargetChoice

from .vmm_app_support import calculate, workspace


def test_vmm_when_calculated_renders_exact_evidence_and_static_chart() -> None:
    # Given
    app = workspace()
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("A,A,B,A,A,B,A,A")
    _ = app.radio[0].set_value(ActualTargetChoice.SECOND)
    _ = app.run()

    # When
    app = calculate(app)

    # Then
    evidence = next(
        item.value for item in app.dataframe if "Dataset role" in item.value.columns
    )
    assert list(evidence.columns) == [
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
    assert evidence["Requested depth"].tolist() == list(range(9))
    assert set(evidence["Workflow"]) == {"variable_order_markov"}
    assert set(evidence["Result scope"]) == {"pooled"}
    assert evidence.loc[0, "Target probability"] == pytest.approx(5 / 18)
    assert evidence.loc[0, "Target surprisal (bits)"] == pytest.approx(
        -math.log2(5 / 18)
    )
    assert evidence.loc[2, "Target probability"] == pytest.approx(5 / 6)
    assert evidence.loc[2, "Target surprisal (bits)"] == pytest.approx(
        -math.log2(5 / 6)
    )
    assert evidence.loc[6, "Evaluation status"] == "In-sample evaluation unavailable"
    assert bool(
        evidence.loc[
            6,
            ["Target probability", "Target surprisal (bits)"],
        ].isna().all()
    )
    assert len(app.get("plotly_chart")) == 1
    chart_text = "\n".join(item.value for item in app.markdown)
    assert "Predictive entropy by requested context depth" in chart_text
    assert any("Exact values remain in the table" in item.value for item in app.caption)


def test_vmm_when_batch_paste_contains_two_identical_sequences_renders_two_charts() -> (
    None
):
    # Given
    app = workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "Batch paste"
    )
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Batch sequences"
    ).set_value("A,A,B,A,A\nA,A,B,A,A")
    _ = app.run()

    # When
    app = calculate(app)

    # Then
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2


def test_vmm_when_calculated_exposes_ready_artifacts_and_reproducibility() -> None:
    # Given
    app = workspace()

    # When
    app = calculate(app)

    # Then
    downloads = {item.label: item for item in app.download_button}
    expected_labels = {
        "Download Context model export (JSON)",
        "Download Context evidence export (CSV)",
        "Download Evaluation export (CSV)",
    }
    assert expected_labels <= downloads.keys()
    assert all(not downloads[label].disabled for label in expected_labels)
    assert any("Experimental raw VMM artifacts" in item.value for item in app.info)
    assert "VMM reproducibility details" in [item.label for item in app.expander]
    reproducibility = "\n".join(item.value for item in app.text)
    assert "Estimator: krichevsky_trofimov" in reproducibility
    assert "Smoothing alpha: 0.500" in reproducibility
    assert "Training dataset role: training" in reproducibility
    assert "Evaluation dataset identifier: not present" in reproducibility
    assert "Record IDs in source order: sequence-001" in reproducibility
    assert "Visible precision: exactly 3 decimal places" in reproducibility
