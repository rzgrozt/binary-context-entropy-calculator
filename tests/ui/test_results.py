from dataclasses import replace

from binary_entropy.ui.results import final_metrics, prefix_dataframe
from binary_entropy.ui.results_view import prefix_table_html
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


def test_prefix_dataframe_when_using_demo_formats_canonical_precision() -> None:
    # Given
    success = _calculate(default_form())

    # When
    frame = prefix_dataframe(success.analysis, success.model)

    # Then
    assert frame.loc[0, "P(next A)"] == "0.620000000000"
    assert frame.loc[0, "Predictive entropy (bits)"] == "0.958042022226"
    assert frame.loc[0, "Posterior State 1"] == "Unavailable before observation"


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
        assert frame.iat[expected.depth, 3] == f"{expected.predictive[0]:.12f}"
        assert frame.iat[expected.depth, 4] == f"{expected.predictive[1]:.12f}"
        assert (
            frame.iat[expected.depth, 5]
            == success.model.labels.observables[expected.predicted_index]
        )
        assert frame.iat[expected.depth, 6] == f"{expected.entropy_bits:.12f}"
        assert frame.iat[expected.depth, 7] == f"{expected.surprisal_bits[0]:.12f}"
        assert frame.iat[expected.depth, 8] == f"{expected.surprisal_bits[1]:.12f}"


def test_final_metrics_when_sequence_is_empty_explains_depth_zero_convention() -> None:
    # Given
    success = _calculate(replace(default_form(), sequence_text=""))

    # When
    metrics = final_metrics(success.analysis, success.model)

    # Then
    assert metrics.depth == 0
    assert metrics.context == "(empty prefix)"
    assert metrics.posterior == "Unavailable before observation"
    assert metrics.next_hidden_0 == "0.600000000000"
    assert metrics.next_hidden_1 == "0.400000000000"


def test_prefix_table_html_when_labels_are_untrusted_is_semantic_and_escaped() -> None:
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
    table_html = prefix_table_html(success.analysis, success.model)

    # Then
    assert 'class="prefix-table-overflow"' in table_html
    assert 'role="region"' in table_html
    assert 'tabindex="0"' in table_html
    assert 'aria-describedby="prefix-table-scroll-instruction"' in table_html
    assert '<table class="dataframe prefix-results-table"' in table_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in table_html
    assert "<script>" not in table_html
    assert "next_target_symbol" not in table_html
    assert "actual_target_probability" not in table_html


def test_prefix_table_html_when_using_demo_preserves_all_exact_rows() -> None:
    # Given
    success = _calculate(default_form())

    # When
    table_html = prefix_table_html(success.analysis, success.model)

    # Then
    assert table_html.count("<tr") == len(success.analysis.rows) + 1
    assert "0.958042022226" in table_html
    assert "0.972577939805" in table_html


def test_prefix_table_html_when_sequence_is_empty_has_depth_zero_row() -> None:
    # Given
    success = _calculate(replace(default_form(), sequence_text=""))

    # When
    table_html = prefix_table_html(success.analysis, success.model)

    # Then
    assert table_html.count("<tr") == 2
    assert "(empty prefix)" in table_html
