"""Top-level Stimulus Search workspace."""

from collections.abc import Sequence

import streamlit as st

from bspe import (
    SearchResult,
    StimulusCandidate,
    stimulus_candidate_csv,
    stimulus_experiment_csv,
    stimulus_generator_config_json,
    stimulus_scientific_csv,
)
from bspe.domain import BinaryLabels
from bspe.records import SequenceDataset, SequenceRecord
from bspe.ui.stimulus_controls import render_search_controls
from bspe.ui.stimulus_derived import render_derived_workflow
from bspe.ui.stimulus_results import final_candidates, render_search_result
from bspe.ui.stimulus_session import (
    CANDIDATE_SELECTION_KEY,
    assigned_candidates,
    assignment_seed,
    matching_tolerances,
    search_failure,
    search_snapshot,
    validation_report,
)
from bspe.ui.vmm_view import render_vmm_result
from bspe.ui.workspace_components import (
    EVIDENCE_ROW_LIMIT,
    render_bounded_records,
)
from bspe.vmm_types import VMMResultScope
from bspe.workbench import VMMAnalysisRequest, analyze_dataset


def render_stimulus_search_sidebar() -> None:
    """Render persistent search configuration, status, and exports."""
    _ = st.title("Binary entropy workbench")
    _ = st.caption("Configure a bounded, reproducible stimulus search.")
    render_search_controls()
    failure = search_failure()
    if failure is not None:
        _ = st.error(failure.message)
    result = search_snapshot()
    with st.expander("Search status", expanded=False):
        if result is None:
            _ = st.info("Not searched.")
        else:
            _ = st.success(f"Current search: {result.status.value.title()}.")
    with st.expander("Complete exports"):
        if result is None:
            _ = st.caption("Exports require a current search result.")
        else:
            render_search_exports(result)


def render_stimulus_search() -> None:
    """Render the latest immutable search and derived stages."""
    _ = st.title("Stimulus Search")
    _ = st.markdown(
        "Generate and inspect bounded, reproducible binary stimulus candidates."
    )
    result = search_snapshot()
    if result is None:
        _ = st.info("Configure the bounded search and submit it explicitly.")
        return
    render_search_result(result)
    if result.accepted:
        with st.expander("Inspect a candidate", expanded=False):
            known_ids = {candidate.stimulus_id for candidate in result.accepted}
            derived = tuple(
                candidate
                for candidate in final_candidates(result)
                if candidate.stimulus_id not in known_ids
            )
            _render_candidate_detail(result.accepted + derived, result)
    with st.expander("Matching, targets and final validation", expanded=False):
        render_derived_workflow(result)


def render_search_exports(result: SearchResult) -> None:
    """Render raw-precision downloads for the current search stages."""
    final = final_candidates(result)
    _ = st.download_button(
        "Candidate CSV",
        stimulus_candidate_csv(result),
        "stimulus-candidates.csv",
        "text/csv",
        on_click="ignore",
    )
    _ = st.download_button(
        "Scientific CSV",
        stimulus_scientific_csv(result),
        "stimulus-scientific.csv",
        "text/csv",
        on_click="ignore",
    )
    _ = st.download_button(
        "Reproducibility configuration JSON",
        stimulus_generator_config_json(
            result,
            match_tolerances=matching_tolerances(),
            assignment_seed=assignment_seed(),
            validation_report=validation_report(),
            final_candidates=final,
        ),
        "stimulus-search.json",
        "application/json",
        on_click="ignore",
    )
    _ = st.download_button(
        "Experiment-ready CSV",
        stimulus_experiment_csv(final),
        "stimulus-experiment.csv",
        "text/csv",
        on_click="ignore",
    )


def _render_candidate_detail(
    candidates: Sequence[StimulusCandidate], result: SearchResult
) -> None:
    identifiers = tuple(str(candidate.stimulus_id) for candidate in candidates)
    stored = st.session_state.get(CANDIDATE_SELECTION_KEY)
    if stored not in identifiers:
        st.session_state[CANDIDATE_SELECTION_KEY] = identifiers[0]
    selected_id = st.selectbox(
        "Candidate detail",
        options=identifiers,
        key=CANDIDATE_SELECTION_KEY,
    )
    candidate = next(
        candidate for candidate in candidates if candidate.stimulus_id == selected_id
    )
    candidate = next(
        (
            derived
            for derived in assigned_candidates() or ()
            if derived.stimulus_id == candidate.stimulus_id
        ),
        candidate,
    )
    _ = st.subheader("Selected candidate detail")
    summary_columns = st.columns(3)
    _ = summary_columns[0].metric("Stimulus ID", selected_id)
    _ = summary_columns[1].metric("Prediction", _prediction(candidate))
    _ = summary_columns[2].metric("Rank score", candidate.rank_score)
    _ = st.markdown(f"**Sequence:** `{_sequence(candidate)}`")
    context = (
        "Unavailable"
        if candidate.context_used is None
        else "".join(
            candidate.symbol_mapping[symbol] for symbol in candidate.context_used
        )
    )
    mapping = candidate.symbol_mapping
    mapping_text = (
        f"Display mapping: 0 = {mapping[0]}, 1 = {mapping[1]}; context = {context}."
    )
    _ = st.caption(mapping_text)
    _render_candidate_frames(candidate)
    if st.checkbox("Show analyzer diagnostics, plot and sequence exports"):
        dataset = SequenceDataset(
            BinaryLabels(states=("internal-0", "internal-1"), observables=("A", "B")),
            (SequenceRecord(str(candidate.stimulus_id), candidate.sequence),),
        )
        analysis = analyze_dataset(
            dataset,
            VMMAnalysisRequest(result.config.vmm_config, VMMResultScope.PER_SEQUENCE),
        )
        render_vmm_result(analysis, dataset)
    evidence = [
        {
            "Depth": row.depth,
            "Matched suffix": "".join(
                candidate.symbol_mapping[symbol] for symbol in row.matched_suffix
            ),
            "Support": row.support,
            "Count A": row.count_a,
            "Count B": row.count_b,
            "Probability A": row.probability_a,
            "Probability B": row.probability_b,
            "Predictive entropy": row.predictive_entropy_bits,
            "Status": row.status.value,
        }
        for row in candidate.depth_rows
    ]
    render_bounded_records(evidence, EVIDENCE_ROW_LIMIT, "requested depth")


def _render_candidate_frames(candidate: StimulusCandidate) -> None:
    _ = st.dataframe(
        (
            {
                "Length": candidate.metrics.length,
                "A count": candidate.metrics.a_count,
                "B count": candidate.metrics.b_count,
                "A proportion": candidate.metrics.a_proportion,
                "B proportion": candidate.metrics.b_proportion,
                "Switches": candidate.metrics.switches,
                "Switch rate": candidate.metrics.switch_rate,
                "Alternation tendency": candidate.metrics.alternation_tendency,
                "Longest A run": candidate.metrics.longest_a_run,
                "Longest B run": candidate.metrics.longest_b_run,
                "Longest run": candidate.metrics.longest_run,
            },
        ),
        hide_index=True,
        width="stretch",
        height="content",
    )
    _ = st.dataframe(
        (
            {
                "Probability A": candidate.probability_a,
                "Probability B": candidate.probability_b,
                "Predictive entropy": candidate.predictive_entropy_bits,
                "Effective depth": candidate.effective_context_depth,
                "Support": candidate.support_count,
                "Surprisal A": candidate.surprisal_a_bits,
                "Surprisal B": candidate.surprisal_b_bits,
                "Group ID": candidate.group_id,
                "Pair ID": candidate.pair_id,
                "Condition": candidate.condition,
                "Target": _target(candidate),
                "Target probability": candidate.target_probability,
                "Target surprisal": candidate.target_surprisal_bits,
                "Congruency": (
                    None
                    if candidate.target_congruency is None
                    else candidate.target_congruency.value
                ),
            },
        ),
        hide_index=True,
        width="stretch",
        height="content",
    )


def _sequence(candidate: StimulusCandidate) -> str:
    return "".join(candidate.symbol_mapping[symbol] for symbol in candidate.sequence)


def _prediction(candidate: StimulusCandidate) -> str:
    if candidate.probability_a is None:
        return "Unavailable"
    if candidate.predicted_target_index is None:
        return "Tie"
    return candidate.symbol_mapping[candidate.predicted_target_index]


def _target(candidate: StimulusCandidate) -> str | None:
    if candidate.target_index is None:
        return None
    return candidate.symbol_mapping[candidate.target_index]
