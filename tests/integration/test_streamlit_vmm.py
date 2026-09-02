from typing import Final

import pytest

from binary_entropy.ui.state import ActualTargetChoice

from .vmm_app_support import calculate, workspace

MLE_LABEL: Final = "Maximum likelihood (alpha = 0.000)"
MLE_UNAVAILABLE_TEXT: Final = (
    "MLE unavailable: unseen context has no occurrences in the training dataset."
)


def test_vmm_workspace_when_opened_shows_default_compact_controls() -> None:
    # Given / When
    app = workspace()

    # Then
    selectors = {item.label: item for item in app.selectbox}
    assert selectors["Markov workflow"].options == [
        "Variable-order Markov",
        "First-order Markov",
    ]
    assert selectors["Markov workflow"].value == "Variable-order Markov"
    assert selectors["VMM smoothing"].options == [
        "Krichevsky-Trofimov (alpha = 0.500)",
        MLE_LABEL,
        "Custom additive smoothing",
    ]
    assert selectors["VMM smoothing"].value == ("Krichevsky-Trofimov (alpha = 0.500)")
    assert selectors["Markov result scope"].value == "Pooled model"
    numbers = {item.label: item for item in app.number_input}
    assert numbers["Minimum context support"].value == 2
    assert "Estimation method" not in selectors
    assert "Prefix prediction mode" not in selectors


def test_first_order_when_selected_reveals_existing_controls_unchanged() -> None:
    # Given
    app = workspace()
    workflow = next(item for item in app.selectbox if item.label == "Markov workflow")

    # When
    _ = workflow.set_value("First-order Markov")
    _ = app.run()

    # Then
    selectors = {item.label: item for item in app.selectbox}
    assert selectors["Estimation method"].options == [
        "Maximum likelihood",
        "Laplace/add-one smoothing",
        "Custom additive smoothing alpha",
    ]
    assert selectors["Prefix prediction mode"].options == [
        "Fixed fitted transition matrix",
        "Re-estimate from each prefix",
    ]
    assert selectors["Markov result scope"].options == [
        "Pooled model",
        "Per-sequence analysis",
    ]
    order = next(item for item in app.text_input if item.label == "Markov order")
    assert order.value == "1"
    assert order.disabled
    assert "VMM smoothing" not in selectors
    assert "Minimum context support" not in [item.label for item in app.number_input]


def test_vmm_additive_when_selected_reveals_positive_custom_alpha() -> None:
    # Given
    app = workspace()
    smoothing = next(item for item in app.selectbox if item.label == "VMM smoothing")
    assert "Custom additive alpha" not in [item.label for item in app.number_input]

    # When
    _ = smoothing.set_value("Custom additive smoothing")
    _ = app.run()

    # Then
    alpha = next(
        item for item in app.number_input if item.label == "Custom additive alpha"
    )
    assert alpha.value == 0.5
    assert alpha.min is not None
    assert alpha.min > 0.0


def test_vmm_mle_when_selected_exposes_fixed_alpha_in_results() -> None:
    # Given
    app = workspace()
    smoothing = next(item for item in app.selectbox if item.label == "VMM smoothing")

    # When
    _ = smoothing.set_value(MLE_LABEL)
    _ = app.run()
    app = calculate(app)

    # Then
    assert "Custom additive alpha" not in [item.label for item in app.number_input]
    result_text = "\n".join(item.value for item in app.markdown)
    assert "Maximum likelihood (MLE) with alpha 0.000" in result_text


def test_vmm_mle_when_context_is_unseen_shows_exact_unavailable_contract() -> None:
    # Given
    app = workspace()
    _ = next(item for item in app.selectbox if item.label == "VMM smoothing").set_value(
        MLE_LABEL
    )
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("")
    _ = app.run()

    # When
    app = calculate(app)

    # Then
    assert MLE_UNAVAILABLE_TEXT in [item.value for item in app.info]


def test_vmm_mle_when_seen_evidence_is_below_support_does_not_claim_unseen() -> None:
    # Given
    app = workspace()
    _ = next(item for item in app.selectbox if item.label == "VMM smoothing").set_value(
        MLE_LABEL
    )
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("A")
    _ = app.run()

    # When
    app = calculate(app)

    # Then
    notices = [item.value for item in app.info]
    assert MLE_UNAVAILABLE_TEXT not in notices
    assert (
        "Prediction unavailable: no context meets the configured minimum support."
        in notices
    )


def test_vmm_when_default_is_calculated_renders_result_section() -> None:
    # Given
    app = workspace()

    # When
    app = calculate(app)

    # Then
    assert "Variable-order Markov" in [item.value for item in app.subheader]
    captions = "\n".join(item.value for item in app.caption)
    assert (
        "The model detects and predicts recurrent finite-context statistical "
        "dependencies in binary sequences."
    ) in captions
    assert any(
        "Effective predictive context depth" in frame.value.columns
        for frame in app.dataframe
    )
    assert any("Requested depth" in frame.value.columns for frame in app.dataframe)
    assert len(app.get("plotly_chart")) == 1


def test_vmm_when_recurrent_context_is_calculated_shows_hand_checked_values() -> None:
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
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Effective predictive context depth"] == "2"
    assert metrics["Actual context used"] == "A, A"
    assert metrics["Support count"] == "2"
    assert metrics["P(next B)"] == "0.833"
    assert metrics["Actual target"] == "B"
    assert metrics["Actual-target surprisal (bits)"] == "0.263"
    depth_frame = next(
        item.value for item in app.dataframe if "Requested depth" in item.value.columns
    )
    assert depth_frame["Requested depth"].tolist() == list(range(9))
    assert depth_frame.loc[2, "Context"] == "A, A"
    assert depth_frame.loc[2, "P(next B)"] == pytest.approx(2.5 / 3.0)


@pytest.mark.parametrize(
    ("scope", "expected_scope"),
    [
        ("Pooled model", "Pooled fit; per-sequence prediction"),
        ("Per-sequence analysis", "Per-sequence fit and prediction"),
    ],
)
def test_vmm_when_batch_scope_is_selected_calculates_each_record(
    scope: str,
    expected_scope: str,
) -> None:
    # Given
    app = workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "Batch paste"
    )
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Batch sequences"
    ).set_value("A,A,B,A,A\nB,A,B,B,A")
    _ = next(
        item for item in app.selectbox if item.label == "Markov result scope"
    ).set_value(scope)
    _ = app.run()

    # When
    app = calculate(app)

    # Then
    summary = next(
        item.value
        for item in app.dataframe
        if "Effective predictive context depth" in item.value.columns
    )
    comparison = next(
        item.value for item in app.dataframe if "Method" in item.value.columns
    )
    assert len(summary) == 2
    assert comparison["Method"].tolist() == [
        "Variable-order Markov",
        "Variable-order Markov",
    ]
    assert comparison["Scope"].tolist() == [expected_scope, expected_scope]


def test_vmm_results_when_control_changes_become_stale() -> None:
    # Given
    app = calculate(workspace())
    assert app.metric

    # When
    support = next(
        item for item in app.number_input if item.label == "Minimum context support"
    )
    _ = support.set_value(3)
    _ = app.run()

    # Then
    assert any("Recalculation required" in item.value for item in app.warning)
    assert not app.metric
    assert not app.dataframe
    assert not any(
        "Context model export" in item.label
        or "Context evidence export" in item.label
        or "Evaluation export" in item.label
        for item in app.download_button
    )


def test_vmm_results_when_smoothing_selection_changes_become_stale() -> None:
    # Given
    app = calculate(workspace())
    assert app.metric

    # When
    smoothing = next(item for item in app.selectbox if item.label == "VMM smoothing")
    _ = smoothing.set_value(MLE_LABEL)
    _ = app.run()

    # Then
    assert any("Recalculation required" in item.value for item in app.warning)
    assert not app.metric
    assert not app.dataframe
