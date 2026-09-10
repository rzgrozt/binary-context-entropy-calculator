"""Search result, candidate detail, and export presentation."""

from typing import Final

import streamlit as st

from bspe import (
    SearchResult,
    StimulusCandidate,
    validate_stimuli,
)
from bspe.stimulus_metadata import apply_presentation_metadata
from bspe.ui.help_text import UI_HELP
from bspe.ui.stimulus_session import (
    assigned_candidates,
    matching_result,
    presentation_metadata,
)
from bspe.ui.workspace_components import (
    COMPARISON_ROW_LIMIT,
    render_bounded_records,
)

_CANDIDATE_COLUMNS: Final = (
    "Selected",
    "Stimulus ID",
    "Sequence",
    "A count",
    "B count",
    "A proportion",
    "Switches",
    "Switch rate",
    "Longest A run",
    "Longest B run",
    "Longest run",
    "Prediction",
    "Probability A",
    "Probability B",
    "Predictive entropy",
    "Effective depth",
    "Support",
    "Rank score",
)

_CANDIDATE_COLUMN_HELP: Final = {
    "A count": UI_HELP["a_count"],
    "A proportion": UI_HELP["a_proportion"],
    "Switches": UI_HELP["switches"],
    "Switch rate": UI_HELP["switch_rate"],
    "Longest A run": UI_HELP["longest_a_run"],
    "Longest B run": UI_HELP["longest_b_run"],
    "Longest run": UI_HELP["longest_run"],
    "Prediction": UI_HELP["prediction"],
    "Probability A": UI_HELP["probability_a"],
    "Probability B": UI_HELP["probability_b"],
    "Predictive entropy": UI_HELP["predictive_entropy"],
    "Effective depth": UI_HELP["effective_depth"],
    "Support": UI_HELP["context_support"],
    "Rank score": UI_HELP["rank_score"],
}


def render_search_result(result: SearchResult) -> None:
    """Render one stored immutable search result."""
    _ = st.subheader("Search results")
    columns = st.columns(5)
    values = (
        ("Status", result.status.value.title(), UI_HELP["partial_reason"]),
        ("Requested", result.config.desired_stimuli, UI_HELP["desired_count"]),
        (
            "Evaluated",
            result.evaluated_count,
            "Sequences sampled and analyzed before ranking.",
        ),
        (
            "Accepted",
            result.accepted_count,
            "Candidates that passed every hard constraint.",
        ),
        (
            "Selected",
            result.selected_count,
            "Top-ranked accepted candidates chosen toward the desired count.",
        ),
    )
    for column, (label, value, help_text) in zip(columns, values, strict=True):
        _ = column.metric(label, value, help=help_text)
    _ = st.caption(
        f"Search seed: {result.config.seed}. Acceptance rate: "
        f"{result.accepted_count / result.evaluated_count:.3%}. "
        "The complete bounded sample is evaluated before ranking and selection."
    )
    report = validate_stimuli(result.accepted)
    with st.expander("Achieved ranges across accepted candidates", expanded=False):
        _ = st.caption(
            "Min/mean/max of key metrics over accepted candidates, for context."
        )
        summaries = (
            ("Predictive entropy (bits)", report.predictive_entropy),
            ("P(predicted)", report.predicted_probability),
            ("Effective context depth", report.effective_depth),
            ("A proportion", report.a_proportion),
            ("Switch rate", report.switch_rate),
            ("Longest run", report.longest_run),
        )
        render_bounded_records(
            [
                {
                    "Metric": name,
                    "Minimum": value.minimum,
                    "Mean": value.mean,
                    "Maximum": value.maximum,
                }
                for name, value in summaries
                if value is not None
            ],
            COMPARISON_ROW_LIMIT,
            "metric",
        )
    if result.partial_reason is not None:
        _ = st.warning(f"Partial result: {result.partial_reason.value}.")
    selected_ids = {candidate.stimulus_id for candidate in result.selected}
    render_bounded_records(
        [
            _candidate_row(
                candidate,
                selected=candidate.stimulus_id in selected_ids,
            )
            for candidate in result.accepted
        ],
        COMPARISON_ROW_LIMIT,
        "rank score, then stimulus ID",
        columns=_CANDIDATE_COLUMNS,
        column_help=_CANDIDATE_COLUMN_HELP,
    )
    if result.violations:
        render_bounded_records(
            [
                {
                    "Constraint": violation.constraint,
                    "Failure count": violation.count,
                    "Frequency": violation.frequency,
                }
                for violation in result.violations
            ],
            COMPARISON_ROW_LIMIT,
            "constraint name",
        )


def final_candidates(result: SearchResult) -> tuple[StimulusCandidate, ...]:
    """Return the final statistical stage with separately assigned presentation."""
    candidates = _statistical_candidates(result)
    metadata = presentation_metadata()
    return (
        candidates
        if metadata is None
        else apply_presentation_metadata(candidates, metadata)
    )


def _statistical_candidates(result: SearchResult) -> tuple[StimulusCandidate, ...]:
    assigned = assigned_candidates()
    if assigned is not None:
        return assigned
    matching = matching_result()
    if matching is not None:
        paired = tuple(
            candidate
            for pair in matching.pairs
            for candidate in (pair.first, pair.second)
        )
        return paired + matching.unmatched
    return result.selected


def _candidate_row(
    candidate: StimulusCandidate,
    *,
    selected: bool,
) -> dict[str, str | int | float | bool | None]:
    prediction = (
        "Unavailable"
        if candidate.probability_a is None
        else "Tie"
        if candidate.predicted_target_index is None
        else candidate.symbol_mapping[candidate.predicted_target_index]
    )
    return {
        "Selected": selected,
        "Stimulus ID": str(candidate.stimulus_id),
        "Sequence": _sequence(candidate),
        "A count": candidate.metrics.a_count,
        "B count": candidate.metrics.b_count,
        "A proportion": candidate.metrics.a_proportion,
        "Switches": candidate.metrics.switches,
        "Switch rate": candidate.metrics.switch_rate,
        "Longest A run": candidate.metrics.longest_a_run,
        "Longest B run": candidate.metrics.longest_b_run,
        "Longest run": candidate.metrics.longest_run,
        "Prediction": prediction,
        "Probability A": candidate.probability_a,
        "Probability B": candidate.probability_b,
        "Predictive entropy": candidate.predictive_entropy_bits,
        "Effective depth": candidate.effective_context_depth,
        "Support": candidate.support_count,
        "Rank score": candidate.rank_score,
    }


def _sequence(candidate: StimulusCandidate) -> str:
    return "".join(candidate.symbol_mapping[symbol] for symbol in candidate.sequence)
