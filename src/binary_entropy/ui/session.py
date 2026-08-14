"""Typed access to calculation records stored in Streamlit session state."""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from binary_entropy.ui.state import (
    CalculationFailure,
    CalculationOutcome,
    CalculationSuccess,
    CalculatorForm,
    PresetImportFailure,
    PresetImportOutcome,
    PresetImportSuccess,
)

CALCULATION_KEY: Final = "_calculation_record"
SUBMISSION_KEY: Final = "_submission_failure"
PRESET_IMPORT_KEY: Final = "_preset_import_outcome"


@dataclass(frozen=True, slots=True)
class CalculationRecord:
    """Successful calculation paired with its submitted input identity."""

    success: CalculationSuccess
    fingerprint: str


@dataclass(frozen=True, slots=True)
class SubmissionFailure:
    """Calculation error paired with the invalid submitted inputs."""

    message: str
    fingerprint: str


def store_calculation(outcome: CalculationOutcome, form: CalculatorForm) -> None:
    """Replace prior output with one complete success or failure."""
    clear_calculation()
    match outcome:  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationSuccess() as success:
            st.session_state[CALCULATION_KEY] = CalculationRecord(
                success=success,
                fingerprint=form.fingerprint(),
            )
        case CalculationFailure(message=message):
            st.session_state[SUBMISSION_KEY] = SubmissionFailure(
                message=message,
                fingerprint=form.fingerprint(),
            )


def calculation_record() -> CalculationRecord | None:
    """Return a typed stored result when one exists."""
    match st.session_state.get(CALCULATION_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case CalculationRecord() as record:
            return record
        case _:
            return None


def submission_failure(form: CalculatorForm) -> SubmissionFailure | None:
    """Return only the error belonging to the current unchanged inputs."""
    match st.session_state.get(SUBMISSION_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case SubmissionFailure() as failure if (
            failure.fingerprint == form.fingerprint()
        ):
            return failure
        case _:
            return None


def clear_calculation() -> None:
    """Remove submitted output and validation state."""
    for key in (CALCULATION_KEY, SUBMISSION_KEY):
        if key in st.session_state:
            del st.session_state[key]


def store_preset_import(outcome: PresetImportOutcome) -> None:
    """Persist the latest transactional preset outcome."""
    st.session_state[PRESET_IMPORT_KEY] = outcome


def preset_import_outcome() -> PresetImportOutcome | None:
    """Return the latest typed preset import outcome."""
    match st.session_state.get(PRESET_IMPORT_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case PresetImportSuccess() as success:
            return success
        case PresetImportFailure() as failure:
            return failure
        case _:
            return None


def clear_preset_import() -> None:
    """Remove a prior preset status notice."""
    if PRESET_IMPORT_KEY in st.session_state:
        del st.session_state[PRESET_IMPORT_KEY]
