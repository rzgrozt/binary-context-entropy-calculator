"""Explicit complement, matching, assignment, and QC stages."""

import streamlit as st

from bspe import (
    MatchingResult,
    MatchTolerances,
    MetricSummary,
    SearchResult,
    StimulusCandidate,
    ValidationReport,
    create_complement_candidates,
    evaluate_candidate,
    match_stimuli,
    validate_stimuli,
)
from bspe.stimulus_targets import TargetAssignmentMode, assign_outcomes
from bspe.ui.help_text import UI_HELP
from bspe.ui.stimulus_metadata_controls import render_presentation_controls
from bspe.ui.stimulus_results import final_candidates
from bspe.ui.stimulus_session import (
    assigned_candidates,
    complement_candidates,
    matching_result,
    store_assignment,
    store_complements,
    store_matching,
    store_validation,
    validation_report,
)
from bspe.ui.workspace_components import (
    COMPARISON_ROW_LIMIT,
    render_bounded_records,
)


def render_derived_workflow(result: SearchResult) -> None:
    """Render each optional operation as an explicit immutable stage."""
    _ = st.subheader("Derived operations")
    _ = st.caption(
        "Complements are independently analyzed controls; they may violate the "
        "original symbol-specific filters. Matching compares composition and runs "
        "after exchanging A/B identities and requires equal sequence lengths."
    )
    if st.button("Create complements"):
        store_complements(create_complement_candidates(result.selected, result.config))
        st.rerun()
    complements = complement_candidates()
    if complements is not None:
        _ = st.metric(
            "Complement candidates",
            len(complements),
            help="Bitwise A/B inversions analyzed as independent control records.",
        )

    tolerances = _render_tolerances()
    if st.button("Match stimuli"):
        source = result.selected + (complements or ())
        store_matching(match_stimuli(source, tolerances), tolerances)
        st.rerun()
    matching = matching_result()
    if matching is not None:
        _render_matching(matching)

    assignment_seed = int(
        st.number_input(
            "Assignment seed",
            min_value=0,
            max_value=(1 << 53) - 1,
            value=0,
            key="stimulus-assignment-seed-input",
            help=UI_HELP["assignment_seed"],
        )
    )
    assignment_mode = st.selectbox(
        "Target assignment",
        options=tuple(TargetAssignmentMode),
        format_func=lambda value: value.value,
        help=UI_HELP["target_assignment"],
    )
    if st.button("Assign targets"):
        store_assignment(
            assign_outcomes(final_candidates(result), assignment_mode, assignment_seed),
            assignment_seed,
        )
        st.rerun()
    assigned = assigned_candidates()
    if assigned is not None:
        _render_assignments(assigned)
    render_presentation_controls()
    if st.button("Run descriptive QC"):
        store_validation(validate_stimuli(final_candidates(result)))
    report = validation_report()
    if report is not None:
        _render_validation(report)
        _render_final_constraints(result)


def _render_tolerances() -> MatchTolerances:
    with st.expander("Matching tolerances", expanded=False):
        _ = st.caption(UI_HELP["matching_tolerances_section"])
        entropy = st.number_input(
            "Entropy tolerance",
            min_value=0.0,
            value=1.0,
            key="stimulus-match-entropy",
            help=UI_HELP["tolerance_entropy"],
        )
        prediction = st.number_input(
            "Prediction strength tolerance",
            min_value=0.0,
            value=1.0,
            key="stimulus-match-prediction-strength",
            help=UI_HELP["tolerance_prediction_strength"],
        )
        a_proportion = st.number_input(
            "A proportion tolerance",
            min_value=0.0,
            value=1.0,
            key="stimulus-match-a-proportion",
            help=UI_HELP["tolerance_a_proportion"],
        )
        switch_rate = st.number_input(
            "Switch rate tolerance",
            min_value=0.0,
            value=1.0,
            key="stimulus-match-switch-rate",
            help=UI_HELP["tolerance_switch_rate"],
        )
        run_structure = st.number_input(
            "Run structure tolerance",
            min_value=0.0,
            value=8.0,
            key="stimulus-match-run-structure",
            help=UI_HELP["tolerance_run_structure"],
        )
        effective_depth = st.number_input(
            "Effective depth tolerance",
            min_value=0.0,
            value=8.0,
            key="stimulus-match-effective-depth",
            help=UI_HELP["tolerance_effective_depth"],
        )
    return MatchTolerances(
        entropy,
        prediction,
        a_proportion,
        switch_rate,
        run_structure,
        effective_depth,
    )


def _render_matching(result: MatchingResult) -> None:
    columns = st.columns(2)
    _ = columns[0].metric(
        "Matched pairs",
        len(result.pairs),
        help="Pairs formed within every tolerance after exchanging A/B identities.",
    )
    _ = columns[1].metric(
        "Unmatched candidates",
        len(result.unmatched),
        help="Candidates with no qualifying opposite-prediction partner.",
    )
    rows = [
        {
            "Pair ID": pair.pair_id,
            "First stimulus ID": str(pair.first.stimulus_id),
            "Second stimulus ID": str(pair.second.stimulus_id),
            "Total distance": pair.distance,
        }
        for pair in result.pairs
    ]
    render_bounded_records(rows, COMPARISON_ROW_LIMIT, "pair ID")
    if result.unmatched:
        unmatched = [
            {
                "Stimulus ID": str(candidate.stimulus_id),
                "Sequence": "".join(
                    candidate.symbol_mapping[symbol] for symbol in candidate.sequence
                ),
            }
            for candidate in result.unmatched
        ]
        render_bounded_records(unmatched, COMPARISON_ROW_LIMIT, "stimulus ID")


def _render_assignments(candidates: tuple[StimulusCandidate, ...]) -> None:
    rows = [
        {
            "Stimulus ID": str(candidate.stimulus_id),
            "Sequence": "".join(
                candidate.symbol_mapping[symbol] for symbol in candidate.sequence
            ),
            "Target": candidate.symbol_mapping[candidate.target_index]
            if candidate.target_index is not None
            else "Unavailable",
            "Congruency": candidate.target_congruency.value
            if candidate.target_congruency is not None
            else "unassigned",
            "Target probability": candidate.target_probability,
            "Target surprisal": candidate.target_surprisal_bits,
        }
        for candidate in candidates
    ]
    render_bounded_records(rows, COMPARISON_ROW_LIMIT, "stimulus ID")


def _render_validation(report: ValidationReport) -> None:
    _ = st.subheader("Descriptive quality control", help=UI_HELP["descriptive_qc_section"])
    _ = st.caption("Descriptive similarity does not establish statistical equivalence.")
    metrics = (
        ("Stimuli", report.stimulus_count),
        ("Predicted A", report.predicted_a_count),
        ("Predicted B", report.predicted_b_count),
        ("Ties", report.tie_count),
        ("Unavailable", report.unavailable_count),
        ("Expected targets", report.expected_target_count),
        ("Unexpected targets", report.unexpected_target_count),
        ("Tie targets", report.tie_target_count),
        ("Unassigned targets", report.unassigned_target_count),
        ("Imbalance flags", ", ".join(report.imbalance_flags) or "None"),
    )
    for start in range(0, len(metrics), 3):
        row = metrics[start : start + 3]
        columns = st.columns(len(row))
        for column, (label, value) in zip(columns, row, strict=True):
            _ = column.metric(label, value)
    summaries = (
        ("A count", report.a_count),
        ("B count", report.b_count),
        ("A proportion", report.a_proportion),
        ("B proportion", report.b_proportion),
        ("Switches", report.switches),
        ("Switch rate", report.switch_rate),
        ("Longest A run", report.longest_a_run),
        ("Longest B run", report.longest_b_run),
        ("Longest run", report.longest_run),
        ("Predicted probability", report.predicted_probability),
        ("Predictive entropy", report.predictive_entropy),
        ("Effective depth", report.effective_depth),
        ("Context support", report.context_support),
        ("Target surprisal", report.target_surprisal),
    )
    summary_rows = [
        _summary_row(label, summary)
        for label, summary in summaries
        if summary is not None
    ]
    render_bounded_records(
        summary_rows,
        COMPARISON_ROW_LIMIT,
        "metric name",
    )


def _summary_row(label: str, summary: MetricSummary) -> dict[str, str | float]:
    return {
        "Metric": label,
        "Minimum": summary.minimum,
        "Mean": summary.mean,
        "Maximum": summary.maximum,
    }


def _render_final_constraints(result: SearchResult) -> None:
    candidates = final_candidates(result)
    failures = [
        {
            "Stimulus ID": str(candidate.stimulus_id),
            "Failed original filters": ", ".join(failed),
        }
        for candidate in candidates
        if (failed := evaluate_candidate(candidate, result.config.constraints))
    ]
    if failures:
        _ = st.warning(
            f"{len(failures)} final stimuli fail original search filters. "
            "Inspect explicit complement controls before using this set."
        )
        render_bounded_records(failures, COMPARISON_ROW_LIMIT, "final-set order")
    elif candidates:
        _ = st.success("Every final stimulus satisfies the original search filters.")
