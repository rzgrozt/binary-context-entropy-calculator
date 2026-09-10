"""Typed immutable Stimulus Search session-state lifecycle."""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from bspe import (
    MatchingResult,
    MatchTolerances,
    SearchResult,
    StimulusCandidate,
    ValidationReport,
)
from bspe.stimulus_metadata import PresentationMetadata

PRESENTATION_METADATA_KEY: Final = "_stimulus_presentation_metadata"

SEARCH_SNAPSHOT_KEY: Final = "_stimulus_search_snapshot"
SEARCH_FAILURE_KEY: Final = "_stimulus_search_failure"
COMPLEMENTS_KEY: Final = "_stimulus_complements"
MATCHING_RESULT_KEY: Final = "_stimulus_matching_result"
MATCH_TOLERANCES_KEY: Final = "_stimulus_matching_tolerances"
ASSIGNED_CANDIDATES_KEY: Final = "_stimulus_assigned_candidates"
ASSIGNMENT_SEED_KEY: Final = "_stimulus_assignment_seed"
VALIDATION_REPORT_KEY: Final = "_stimulus_validation_report"
CANDIDATE_SELECTION_KEY: Final = "stimulus-search-candidate-id"


@dataclass(frozen=True, slots=True)
class StimulusSearchFailure:
    """An invalid search configuration shown at the UI boundary."""

    message: str


@dataclass(frozen=True, slots=True)
class _CandidateCollection:
    candidates: tuple[StimulusCandidate, ...]


def store_search_snapshot(snapshot: SearchResult) -> None:
    """Replace the search snapshot and clear only derived state."""
    clear_derived_state()
    if SEARCH_FAILURE_KEY in st.session_state:
        del st.session_state[SEARCH_FAILURE_KEY]
    st.session_state[SEARCH_SNAPSHOT_KEY] = snapshot


def store_search_failure(message: str) -> None:
    """Make a failed submission unavailable without affecting Analyzer state."""
    clear_derived_state()
    if SEARCH_SNAPSHOT_KEY in st.session_state:
        del st.session_state[SEARCH_SNAPSHOT_KEY]
    st.session_state[SEARCH_FAILURE_KEY] = StimulusSearchFailure(message)


def invalidate_stimulus_search() -> None:
    """Clear stale search and derived output after computational edits."""
    clear_derived_state()
    for key in (SEARCH_SNAPSHOT_KEY, SEARCH_FAILURE_KEY):
        if key in st.session_state:
            del st.session_state[key]


def search_snapshot() -> SearchResult | None:
    """Return the current immutable search snapshot when available."""
    match st.session_state.get(SEARCH_SNAPSHOT_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case SearchResult() as snapshot:
            return snapshot
        case _:
            return None


def search_failure() -> StimulusSearchFailure | None:
    """Return the latest specific search-boundary failure."""
    match st.session_state.get(SEARCH_FAILURE_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case StimulusSearchFailure() as failure:
            return failure
        case _:
            return None


def store_complements(candidates: tuple[StimulusCandidate, ...]) -> None:
    """Store independently analyzed complement candidates."""
    st.session_state[COMPLEMENTS_KEY] = _CandidateCollection(candidates)
    clear_matching_state()


def complement_candidates() -> tuple[StimulusCandidate, ...] | None:
    """Return the stored typed complement tuple."""
    return _candidate_tuple(COMPLEMENTS_KEY)


def store_matching(result: MatchingResult, tolerances: MatchTolerances) -> None:
    """Store matching output and its exact tolerance configuration."""
    st.session_state[MATCHING_RESULT_KEY] = result
    st.session_state[MATCH_TOLERANCES_KEY] = tolerances
    clear_assignment_state()


def matching_result() -> MatchingResult | None:
    """Return the current immutable matching result."""
    match st.session_state.get(MATCHING_RESULT_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case MatchingResult() as result:
            return result
        case _:
            return None


def matching_tolerances() -> MatchTolerances | None:
    """Return tolerances belonging to the current matching result."""
    match st.session_state.get(MATCH_TOLERANCES_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case MatchTolerances() as tolerances:
            return tolerances
        case _:
            return None


def store_assignment(candidates: tuple[StimulusCandidate, ...], seed: int) -> None:
    """Store target assignments separately from the search snapshot."""
    st.session_state[ASSIGNED_CANDIDATES_KEY] = _CandidateCollection(candidates)
    st.session_state[ASSIGNMENT_SEED_KEY] = seed
    if VALIDATION_REPORT_KEY in st.session_state:
        del st.session_state[VALIDATION_REPORT_KEY]


def assigned_candidates() -> tuple[StimulusCandidate, ...] | None:
    """Return the independently stored target-assigned tuple."""
    return _candidate_tuple(ASSIGNED_CANDIDATES_KEY)


def assignment_seed() -> int | None:
    """Return the seed used for the current target assignment."""
    match st.session_state.get(ASSIGNMENT_SEED_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case bool():
            return None
        case int() as seed:
            return seed
        case _:
            return None


def store_validation(report: ValidationReport) -> None:
    """Store descriptive quality-control output separately."""
    st.session_state[VALIDATION_REPORT_KEY] = report


def validation_report() -> ValidationReport | None:
    """Return the current descriptive validation report."""
    match st.session_state.get(VALIDATION_REPORT_KEY):  # noqa: RUF100  # noqa: MATCH_OK
        case ValidationReport() as report:
            return report
        case _:
            return None


def clear_derived_state() -> None:
    """Remove only data derived from a prior search snapshot."""
    for key in (
        PRESENTATION_METADATA_KEY,
        COMPLEMENTS_KEY,
        MATCHING_RESULT_KEY,
        MATCH_TOLERANCES_KEY,
        ASSIGNED_CANDIDATES_KEY,
        ASSIGNMENT_SEED_KEY,
        VALIDATION_REPORT_KEY,
    ):
        if key in st.session_state:
            del st.session_state[key]


def clear_matching_state() -> None:
    """Remove matching and every downstream assignment artifact."""
    for key in (MATCHING_RESULT_KEY, MATCH_TOLERANCES_KEY):
        if key in st.session_state:
            del st.session_state[key]
    clear_assignment_state()


def clear_assignment_state() -> None:
    """Remove assignment and validation without changing search or matching."""
    for key in (
        ASSIGNED_CANDIDATES_KEY,
        ASSIGNMENT_SEED_KEY,
        VALIDATION_REPORT_KEY,
    ):
        if key in st.session_state:
            del st.session_state[key]


def _candidate_tuple(key: str) -> tuple[StimulusCandidate, ...] | None:
    match st.session_state.get(key):  # noqa: RUF100  # noqa: MATCH_OK
        case _CandidateCollection(candidates=candidates):
            return candidates
        case _:
            return None


def store_presentation_metadata(metadata: PresentationMetadata) -> None:
    """Keep presentation settings separate from analysis and target assignment."""
    st.session_state[PRESENTATION_METADATA_KEY] = metadata


def presentation_metadata() -> PresentationMetadata | None:
    """Return explicitly applied final-set presentation settings."""
    value = st.session_state.get(PRESENTATION_METADATA_KEY)
    return value if isinstance(value, PresentationMetadata) else None
