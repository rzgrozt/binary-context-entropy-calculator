from dataclasses import replace

import pytest

from binary_entropy.ui.state import (
    ActualTargetChoice,
    CalculationFailure,
    CalculationOutcome,
    CalculationSuccess,
    PresetImportFailure,
    PresetImportSuccess,
    calculate_form,
    default_form,
    import_preset,
)


def _require_success(outcome: CalculationOutcome) -> CalculationSuccess:
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationSuccess() as success:
            return success
        case CalculationFailure(message=message):
            raise AssertionError(message)


def test_calculate_form_when_using_demo_sequence_returns_every_prefix() -> None:
    # Given
    form = default_form()

    # When
    success = _require_success(calculate_form(form))

    # Then
    assert len(success.analysis.sequence) == 7
    assert len(success.analysis.rows) == 8
    assert success.analysis.rows[-1].depth == 7


def test_calculate_form_when_probability_row_is_invalid_preserves_error() -> None:
    # Given
    form = default_form()
    invalid_model = replace(form.model, initial=(0.7, 0.4))

    # When
    outcome = calculate_form(replace(form, model=invalid_model))

    # Then
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationFailure(message=message):
            assert "initial must sum to 1" in message
        case CalculationSuccess():
            pytest.fail("invalid probabilities were accepted")


def test_calculate_form_when_sequence_token_is_invalid_names_position() -> None:
    # Given
    form = replace(default_form(), sequence_text="A, B, unknown, A")

    # When
    outcome = calculate_form(form)

    # Then
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationFailure(message=message):
            assert "invalid sequence token 'unknown' at position 3" in message
        case CalculationSuccess():
            pytest.fail("invalid sequence was accepted")


def test_calculate_form_when_sequence_is_empty_returns_depth_zero() -> None:
    # Given
    form = replace(default_form(), sequence_text="  \n ")

    # When
    success = _require_success(calculate_form(form))

    # Then
    assert len(success.analysis.rows) == 1
    assert success.analysis.rows[0].posterior is None
    assert success.analysis.observed_entropy_bits is None


def test_calculate_form_when_sequence_is_impossible_returns_failure() -> None:
    # Given
    form = default_form()
    impossible_model = replace(
        form.model,
        emission=((1.0, 0.0), (1.0, 0.0)),
    )
    impossible_form = replace(
        form,
        model=impossible_model,
        sequence_text="B",
    )

    # When
    outcome = calculate_form(impossible_form)

    # Then
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationFailure(message=message):
            assert "observable 1 at position 1 has zero likelihood" in message
        case CalculationSuccess():
            pytest.fail("impossible sequence was accepted")


def test_calculate_form_when_actual_target_is_selected_assesses_final_row() -> None:
    # Given
    form = replace(
        default_form(),
        actual_target=ActualTargetChoice.FIRST,
    )

    # When
    success = _require_success(calculate_form(form))

    # Then
    assert success.target_assessment is not None
    assert success.target_assessment.actual_target_index == 0
    assert (
        success.target_assessment.probability == success.analysis.rows[-1].predictive[0]
    )


def test_form_fingerprint_when_any_export_input_changes_becomes_stale() -> None:
    # Given
    form = default_form()

    # When
    changed = replace(form, sequence_id="sequence-002")

    # Then
    assert changed.fingerprint() != form.fingerprint()


def test_import_preset_when_valid_replaces_only_model_inputs() -> None:
    # Given
    form = replace(default_form(), sequence_text="B, A", sequence_id="candidate-9")
    payload = b"""{
      "schema_version": 1,
      "preset_name": "Imported model",
      "state_labels": ["Quiet", "Active"],
      "observable_labels": ["Left", "Right"],
      "initial": [0.25, 0.75],
      "transition": [[0.8, 0.2], [0.1, 0.9]],
      "emission": [[0.6, 0.4], [0.3, 0.7]]
    }"""

    # When
    outcome = import_preset(payload, form)

    # Then
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case PresetImportSuccess(form=imported):
            assert imported.model.state_labels == ("Quiet", "Active")
            assert imported.preset_name == "Imported model"
            assert imported.sequence_text == "B, A"
            assert imported.sequence_id == "candidate-9"
        case PresetImportFailure(message=message):
            raise AssertionError(message)


def test_import_preset_when_invalid_does_not_return_partial_form() -> None:
    # Given
    form = default_form()

    # When
    outcome = import_preset(b"{not-json}", form)

    # Then
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case PresetImportFailure(message=message):
            assert "could not be decoded" in message
            assert form == default_form()
        case PresetImportSuccess():
            pytest.fail("invalid preset was accepted")
