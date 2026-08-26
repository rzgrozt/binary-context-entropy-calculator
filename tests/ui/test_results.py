from dataclasses import replace

import pytest

from binary_entropy.ui.results import final_metrics, prefix_dataframe
from binary_entropy.ui.state import (
    CalculationFailure,
    CalculationSuccess,
    CalculatorForm,
    calculate_form,
    default_form,
)
from tests.unit.helpers import load_hand_fixture


def _calculate(form: CalculatorForm) -> CalculationSuccess:
    outcome = calculate_form(form)
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationSuccess() as success:
            return success
        case CalculationFailure(message=message):
            raise AssertionError(message)


def test_prefix_dataframe_when_using_demo_has_context_and_candidate_surprisals() -> (
    None
):
    # Given
    success = _calculate(default_form())

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert len(frame) == 8
    assert frame["Depth"].tolist() == list(range(8))
    assert frame.loc[0, "Observed context"] == "(empty prefix)"
    assert frame.loc[3, "Observed context"] == "A, B, B"
    assert "Surprisal if next A (bits)" in frame.columns
    assert "Surprisal if next B (bits)" in frame.columns
    assert "next_target_symbol" not in frame.columns


def test_prefix_dataframe_when_using_demo_preserves_unrounded_numeric_values() -> None:
    # Given
    success = _calculate(default_form())

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert frame.loc[0, "P(next A)"] == 0.62
    assert frame.loc[0, "Predictive entropy (bits)"] == 0.9580420222262995
    assert frame.loc[0, "Posterior status"] == "Unavailable before observation"


def test_prefix_dataframe_when_hand_fixture_matches_all_depth_display_profile() -> None:
    # Given
    success = _calculate(default_form())
    fixture = load_hand_fixture()

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert len(frame) == len(fixture.rows)
    for _, expected in zip(success.analysis.rows, fixture.rows, strict=True):
        assert frame.iat[expected.depth, 1] == expected.context
        assert frame.iat[expected.depth, 3] == pytest.approx(
            expected.predictive[0], abs=1e-15
        )
        assert frame.iat[expected.depth, 4] == pytest.approx(
            expected.predictive[1], abs=1e-15
        )
        assert (
            frame.iat[expected.depth, 5]
            == success.model.labels.observables[expected.predicted_index]
        )
        assert frame.iat[expected.depth, 6] == pytest.approx(
            expected.entropy_bits, abs=1e-15
        )
        assert frame.iat[expected.depth, 7] == pytest.approx(
            expected.surprisal_bits[0], abs=1e-15
        )
        assert frame.iat[expected.depth, 8] == pytest.approx(
            expected.surprisal_bits[1], abs=1e-15
        )


def test_final_metrics_when_sequence_is_empty_explains_depth_zero_convention() -> None:
    # Given
    success = _calculate(replace(default_form(), sequence_text=""))

    # When
    metrics = final_metrics(success.analysis, success.model)

    # Then
    assert metrics.depth == 0
    assert metrics.context == "(empty prefix)"
    assert metrics.posterior == "Unavailable before observation"
    assert metrics.next_hidden_0 == "0.600"
    assert metrics.next_hidden_1 == "0.400"


def test_prefix_dataframe_when_labels_are_untrusted_keeps_them_as_text() -> None:
    # Given
    form = default_form()
    untrusted_label = "<script>alert(1)</script>"
    form = replace(
        form,
        model=replace(
            form.model,
            state_labels=("State <one>", "State & two"),
            observable_labels=(untrusted_label, "B & C"),
        ),
        sequence_text=f"{untrusted_label}, B & C",
    )
    success = _calculate(form)

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert f"P(next {untrusted_label})" in frame.columns
    assert frame.loc[1, "Observed symbol"] == untrusted_label
    assert "next_target_symbol" not in frame.columns
    assert "actual_target_probability" not in frame.columns


def test_prefix_dataframe_when_using_demo_preserves_all_exact_rows() -> None:
    # Given
    success = _calculate(default_form())

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert len(frame) == len(success.analysis.rows)
    assert frame.loc[0, "Predictive entropy (bits)"] == pytest.approx(
        0.958042022226, abs=1e-12
    )
    assert frame.loc[7, "Predictive entropy (bits)"] == pytest.approx(
        0.972577939805, abs=1e-12
    )


def test_prefix_dataframe_when_sequence_is_empty_has_depth_zero_row() -> None:
    # Given
    success = _calculate(replace(default_form(), sequence_text=""))

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert len(frame) == 1
    assert frame.loc[0, "Observed context"] == "(empty prefix)"
