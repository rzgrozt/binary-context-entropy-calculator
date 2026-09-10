"""Render fitted Markov model evidence at the selected result scope."""

import streamlit as st

from bspe.domain import float_values
from bspe.markov_information import (
    empirical_conditional_entropy,
    entropy_rate,
    stationary_distribution,
)
from bspe.markov_types import (
    MarkovBatchAnalysis,
    MarkovEstimation,
    MarkovModel,
    MarkovResultScope,
    UnavailableStationaryDistribution,
    UniqueStationaryDistribution,
)
from bspe.ui.markov_results import markov_transition_dataframe
from bspe.ui.text import joined_text
from bspe.ui.tokens import UI_NUMBER_FORMAT, format_ui_decimal
from bspe.ui.workspace_selection import SequenceScope


def render_markov_model_evidence(
    analysis: MarkovBatchAnalysis,
    scope: SequenceScope,
) -> None:
    """Render transition evidence prominently and diagnostics secondarily."""
    models = tuple(record.model for record in analysis.records)

    if analysis.model.estimation_method is MarkovEstimation.MAXIMUM_LIKELIHOOD and any(
        row is None for model in models for row in model.transition_matrix
    ):
        _ = st.warning(
            joined_text(
                (
                    "MLE unavailable for at least one Markov transition row: no ",
                    "outgoing transition was observed for that current state. Add ",
                    "evidence for the state or select additive smoothing.",
                )
            )
        )

    _ = st.subheader("Learned Transition Pattern")
    _ = st.caption(
        joined_text(
            (
                "These probabilities describe what was observed to follow each ",
                "current ",
                "state. The row matching the current final state generates the ",
                "next-target prediction.",
            )
        )
    )
    _ = st.markdown("`T[i,j] = P(next state = j | current state = i)`")

    match analysis.result_scope:
        case MarkovResultScope.POOLED:
            observed_starts = tuple(
                _observed_start(record.sequence, analysis.model)
                for record in analysis.records
            )
            _render_fitted_model(
                analysis.model,
                observed_starts,
                "Advanced Markov Statistics",
            )

        case MarkovResultScope.PER_SEQUENCE:
            match scope:
                case SequenceScope.ONE:
                    record = analysis.records[0]
                    _ = st.markdown(f"**Fitted model: {record.sequence_id}**")
                    _render_fitted_model(
                        record.model,
                        (_observed_start(record.sequence, record.model),),
                        f"Advanced Markov Statistics — {record.sequence_id}",
                    )
                case SequenceScope.MULTIPLE | SequenceScope.ALL:
                    _ = st.info(
                        joined_text(
                            (
                                "Select One sequence scope to inspect independently ",
                                "fitted Markov model transition evidence and advanced ",
                                "statistics.",
                            )
                        )
                    )


def _render_fitted_model(
    model: MarkovModel,
    observed_starts: tuple[str, ...],
    expander_label: str,
) -> None:
    _render_transition_table(model)
    with st.expander(expander_label, expanded=False):
        _render_advanced_statistics(model, observed_starts)


def _render_transition_table(model: MarkovModel) -> None:
    """Render the fitted transition pattern without secondary diagnostics."""
    labels = model.observable_labels

    _ = st.dataframe(
        markov_transition_dataframe(model),
        hide_index=True,
        width="stretch",
        height="content",
        column_config={
            f"P(next {labels[0]})": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"P(next {labels[1]})": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            "Row sum": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
        },
    )


def _render_advanced_statistics(
    model: MarkovModel,
    observed_starts: tuple[str, ...],
) -> None:
    """Render whole-model diagnostics separately from current prediction."""
    labels = model.observable_labels

    starting = (
        None
        if model.starting_distribution is None
        else float_values(model.starting_distribution)
    )

    _ = st.markdown("**Sequence-start distribution**")

    if starting is None:
        _ = st.markdown("Unavailable")
    else:
        _ = st.markdown(
            joined_text(
                (
                    f"{labels[0]}: {format_ui_decimal(starting[0])}  \n",
                    f"{labels[1]}: {format_ui_decimal(starting[1])}",
                )
            )
        )

    if len(observed_starts) == 1:
        _ = st.caption(
            joined_text(
                (
                    f"Observed sequence start: {observed_starts[0]}. ",
                    "This is not a next-target prediction.",
                )
            )
        )
    else:
        _ = st.caption(
            joined_text(
                (
                    "This describes how often the submitted sequences begin with each ",
                    "symbol. It is not a next-target prediction.",
                )
            )
        )

    empirical = empirical_conditional_entropy(model.transition_counts)

    _ = st.metric(
        "Whole-model empirical conditional entropy (bits)",
        "Unavailable"
        if empirical is None
        else format_ui_decimal(empirical),
        help=(
            "Average uncertainty of the next symbol after knowing the current "
            "symbol, weighted by how often each current state occurred in the "
            "observed transition data. This is not the current trial's "
            "predictive entropy."
        ),
    )

    stationary = stationary_distribution(model.transition_matrix)

    match stationary:
        case UniqueStationaryDistribution(distribution=distribution):
            stationary_a, stationary_b = float_values(distribution)
            stationary_text = joined_text(
                (
                    f"{labels[0]} {format_ui_decimal(stationary_a)}; ",
                    f"{labels[1]} {format_ui_decimal(stationary_b)}",
                )
            )

        case UnavailableStationaryDistribution(reason=reason):
            stationary_text = (
                f"Unavailable: {reason.value.replace('_', ' ')}"
            )

    _ = st.metric(
        "Long-run stationary distribution",
        stationary_text,
        help=(
            "The long-run proportion of time the fitted Markov chain would be "
            "expected to spend in each state if the process continued "
            "indefinitely. This is not the current next-target prediction."
        ),
    )

    rate = entropy_rate(model.transition_matrix, stationary)

    _ = st.metric(
        "Long-run entropy rate (bits/symbol)",
        "Unavailable" if rate is None else format_ui_decimal(rate),
        help=(
            "Expected uncertainty contributed by each new symbol when the "
            "fitted Markov chain operates in its stationary regime. This is "
            "a whole-model summary, not the current predictive entropy."
        ),
    )

    estimation_method = model.estimation_method.value.replace("_", " ")
    _ = st.caption(
        joined_text(
            (
                f"Estimation method: {estimation_method}. ",
                f"Smoothing alpha: {format_ui_decimal(model.smoothing_alpha)}. ",
                f"Transitions used: {model.source_transition_count}.",
            )
        )
    )

def _observed_start(sequence: tuple[int, ...], model: MarkovModel) -> str:
    return "Empty" if not sequence else model.observable_labels[sequence[0]]
