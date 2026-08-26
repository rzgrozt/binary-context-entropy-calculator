"""Streamlit rendering for one submitted Markov analysis."""
import math

import pandas as pd

import streamlit as st

from binary_entropy.domain import float_values
from binary_entropy.information import surprisal
from binary_entropy.markov_batch_serialization import markov_batch_summary_csv
from binary_entropy.markov_serialization import markov_model_json, markov_sequence_csv
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovEstimation,
    MarkovPredictionMode,
    MarkovRecordAnalysis,
)
from binary_entropy.ui.chart import markov_entropy_figure, markov_probability_figure
from binary_entropy.ui.markov_model_view import render_markov_model_evidence
from binary_entropy.ui.markov_results import (
    markov_prediction_label,
    markov_prefix_dataframe,
    markov_record_dataframe,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT, format_ui_decimal


def render_markov_result(analysis: MarkovBatchAnalysis) -> None:
    """Render model evidence, per-record predictions, prefixes, and raw exports."""
    labels = analysis.model.observable_labels
    _ = st.subheader("Markov Chain")
    _ = st.caption(
        joined_text(
            (
                f"First-order {analysis.result_scope.value.replace('_', ' ')}; ",
                f"{analysis.model.estimation_method.value.replace('_', ' ')}; ",
                f"alpha {format_ui_decimal(analysis.model.smoothing_alpha)}.",
            )
        )
    )
    _ = st.markdown(
        joined_text(
            (
                f"{len(analysis.records)} independent sequences · ",
                f"{analysis.model.source_transition_count} transitions",
            )
        )
    )
    _render_final_metrics(analysis.records[0], labels)
    _render_prediction_explanation(analysis.records[0], labels)
    _render_candidate_surprisal(analysis.records[0], labels)

    _ = st.markdown("Per-sequence final predictions in deterministic input order.")
    _ = st.dataframe(
        markov_record_dataframe(analysis),
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
            "Predictive entropy (bits)": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            "Target probability": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            "Target surprisal (bits)": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
        },
    )

    render_markov_model_evidence(analysis)
    _render_prefixes(analysis)
    _render_downloads(analysis)


def _render_final_metrics(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> None:
    final = record.rows[-1]
    _ = st.header("Next-Target Prediction")
    if final.predictive is None:
        _ = st.info(
            joined_text(
                (
                    "Final Markov prediction unavailable because the current ",
                    "transition row is not estimated.",
                )
            )
        )
        return
    current_state = (
        "Unavailable"
        if final.context is None
        else labels[final.context[-1]]
    )

    _ = st.markdown(f"**Current state:** {current_state}")
    _ = st.caption(
        "First-order Markov prediction is based on the final observed symbol."
    )
    probability_a, probability_b = float_values(final.predictive)
    
    prediction_columns = st.columns(3)

    _ = prediction_columns[0].metric(
        f"P(next {labels[0]})",
        format_ui_decimal(probability_a),
    )

    _ = prediction_columns[1].metric(
        f"P(next {labels[1]})",
        format_ui_decimal(probability_b),
    )

    _ = prediction_columns[2].metric(
        "Prediction",
        markov_prediction_label(
            probability_a,
            probability_b,
            labels,
        ),
    )

    _ = st.metric(
        "Predictive entropy (bits)",
        "Unavailable"
        if final.entropy_bits is None
        else format_ui_decimal(final.entropy_bits),
    )


def _render_prediction_explanation(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> None:
    final = record.rows[-1]

    _ = st.subheader("Why this prediction?")

    if final.context is None:
        _ = st.info(
            "A prediction requires at least one observed symbol to define "
            "the current state."
        )
        return

    current_index = final.context[-1]
    current_label = labels[current_index]

    count_a, count_b = record.model.transition_counts[current_index]
    total = count_a + count_b

    _ = st.markdown(f"**Current state:** {current_label}")
    _ = st.markdown(
        joined_text(
            (
                f"Observed transitions from {current_label}:  \n",
                f"**{current_label} → {labels[0]}:** {count_a}  \n",
                f"**{current_label} → {labels[1]}:** {count_b}",
            )
        )
    )

    if final.predictive is None:
        _ = st.info(
            joined_text(
                (
                    "Prediction unavailable. The current state has not been ",
                    "observed with a following symbol, so its transition ",
                    "probabilities cannot be estimated with maximum likelihood.",
                )
            )
        )
        return

    probability_a, probability_b = float_values(final.predictive)

    match record.model.estimation_method:
        case MarkovEstimation.MAXIMUM_LIKELIHOOD:
            _ = st.markdown(
                joined_text(
                    (
                        "**Therefore:**  \n",
                        f"`P({labels[0]} | {current_label}) = "
                        f"{count_a} / {total} = "
                        f"{format_ui_decimal(probability_a)}`  \n",
                        f"`P({labels[1]} | {current_label}) = "
                        f"{count_b} / {total} = "
                        f"{format_ui_decimal(probability_b)}`",
                    )
                )
            )

        case MarkovEstimation.ADDITIVE_SMOOTHING:
            alpha = record.model.smoothing_alpha
            denominator = total + 2.0 * alpha

            _ = st.markdown(
                joined_text(
                    (
                        f"Additive smoothing is active with "
                        f"`alpha = {format_ui_decimal(alpha)}`.  \n",
                        "**Therefore:**  \n",
                        f"`P({labels[0]} | {current_label}) = "
                        f"({count_a} + {format_ui_decimal(alpha)}) / "
                        f"{format_ui_decimal(denominator)} = "
                        f"{format_ui_decimal(probability_a)}`  \n",
                        f"`P({labels[1]} | {current_label}) = "
                        f"({count_b} + {format_ui_decimal(alpha)}) / "
                        f"{format_ui_decimal(denominator)} = "
                        f"{format_ui_decimal(probability_b)}`",
                    )
                )
            )

    prediction = markov_prediction_label(
        probability_a,
        probability_b,
        labels,
    )

    if prediction == "Tie":
        _ = st.markdown(
            f"**The model therefore has no preference between "
            f"{labels[0]} and {labels[1]}.**"
        )
    else:
        _ = st.markdown(
            f"**The fitted transition evidence favors {prediction} "
            "as the more likely next target.**"
        )


def _render_candidate_surprisal(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> None:
    final = record.rows[-1]

    if final.predictive is None:
        return

    probability_a, probability_b = float_values(final.predictive)

    surprisal_a = surprisal(probability_a)
    surprisal_b = surprisal(probability_b)

    prediction = markov_prediction_label(
        probability_a,
        probability_b,
        labels,
    )

    if prediction == "Tie":
        interpretation_a = "Equally expected"
        interpretation_b = "Equally expected"
    elif probability_a > probability_b:
        interpretation_a = "More expected"
        interpretation_b = "More surprising"
    else:
        interpretation_a = "More surprising"
        interpretation_b = "More expected"

    _ = st.subheader("If the next target appears...")

    _ = st.caption(
        "Predictive entropy measures uncertainty before the target appears. "
        "Surprisal measures how unexpected a specific target would be if it appeared."
    )

    table = pd.DataFrame(
        {
            "Target": labels,
            "Probability": (
                format_ui_decimal(probability_a),
                format_ui_decimal(probability_b),
            ),
            "Surprisal (bits)": (
                _format_surprisal(surprisal_a),
                _format_surprisal(surprisal_b),
            ),
            "Interpretation": (
                interpretation_a,
                interpretation_b,
            ),
        }
    )

    _ = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height="content",
    )
    
    
def _format_surprisal(value: float) -> str:
    if math.isinf(value):
        return "∞"
    return format_ui_decimal(value)
    
    
def _render_prefixes(analysis: MarkovBatchAnalysis) -> None:
    labels = analysis.model.observable_labels

    _ = st.subheader("Prediction Across Sequence Depth")

    match analysis.prediction_mode:
        case MarkovPredictionMode.FIXED_MODEL:
            _ = st.markdown("**Mode: Fixed fitted transition matrix**")
            _ = st.caption(
                "The transition matrix is estimated from the complete submitted "
                "sequence or selected analysis scope. At each prefix, prediction "
                "depends on the final symbol of that prefix."
            )

        case MarkovPredictionMode.CUMULATIVE_PREFIX:
            _ = st.markdown("**Mode: Cumulative transition learning**")
            _ = st.caption(
                "At each sequence depth, transition probabilities are re-estimated "
                "using only the evidence observed up to that point."
            )

    for record in analysis.records:
        _ = st.markdown(f"**Sequence: {record.sequence_id}**")
        _ = st.dataframe(
            markov_prefix_dataframe(record, labels),
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
                "Predictive entropy (bits)": st.column_config.NumberColumn(
                    format=UI_NUMBER_FORMAT
                ),
            },
        )
        if any(row.predictive is not None for row in record.rows):
            _ = st.markdown("Next-symbol probability by prefix depth.")
            _ = st.write(markov_probability_figure(record, labels))
            _ = st.markdown(
                "Predictive entropy by prefix depth, from 0.000 to 1.000 bits."
            )
            _ = st.write(markov_entropy_figure(record))


def _render_downloads(analysis: MarkovBatchAnalysis) -> None:
    columns = st.columns(3)
    _ = columns[0].download_button(
        "Download Markov model JSON",
        data=markov_model_json(analysis),
        file_name="markov-model.json",
        mime="application/json",
        on_click="ignore",
    )
    _ = columns[1].download_button(
        "Download Markov prefix CSV",
        data=markov_sequence_csv(analysis),
        file_name="markov-prefix.csv",
        mime="text/csv; charset=utf-8",
        on_click="ignore",
    )
    _ = columns[2].download_button(
        "Download Markov batch-summary CSV",
        data=markov_batch_summary_csv(analysis),
        file_name="markov-batch-summary.csv",
        mime="text/csv; charset=utf-8",
        on_click="ignore",
    )
