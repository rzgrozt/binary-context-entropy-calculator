"""Scientific summary rendering for one current calculation."""

from dataclasses import dataclass

import streamlit as st

from binary_entropy.domain import TargetClassification
from binary_entropy.presentation import format_decimal
from binary_entropy.ui.results import final_metrics, format_information
from binary_entropy.ui.state import CalculationSuccess
from binary_entropy.ui.text import joined_text


@dataclass(frozen=True, slots=True)
class MetricDisplay:
    """One labeled value and its scientific qualification."""

    label: str
    value: str
    help_text: str


def render_summary(success: CalculationSuccess) -> None:
    """Render final predictive, hidden-state, and descriptive metrics."""
    analysis = success.analysis
    model = success.model
    values = final_metrics(analysis, model)
    observable_0, observable_1 = model.labels.observables
    state_0, state_1 = model.labels.states
    metrics = (
        MetricDisplay(
            "Context depth", str(values.depth), "Number of observations consumed."
        ),
        MetricDisplay(
            "Observed context", values.context, "The complete entered prefix."
        ),
        MetricDisplay(
            f"P(next {observable_0})",
            values.probability_0,
            "HMM predictive probability after the observed context.",
        ),
        MetricDisplay(
            f"P(next {observable_1})",
            values.probability_1,
            "HMM predictive probability after the observed context.",
        ),
        MetricDisplay(
            "Predicted target",
            values.predicted_target,
            "A tie resolves to variable 1.",
        ),
        MetricDisplay(
            "HMM predictive entropy (bits)",
            values.entropy_bits,
            "Binary uncertainty in the final predictive distribution.",
        ),
        MetricDisplay(
            f"Surprisal if next {observable_0} (bits)",
            values.surprisal_0,
            "Self-information under the final prediction.",
        ),
        MetricDisplay(
            f"Surprisal if next {observable_1} (bits)",
            values.surprisal_1,
            "Self-information under the final prediction.",
        ),
        MetricDisplay(f"Posterior {state_0}", values.posterior_0, values.posterior),
        MetricDisplay(f"Posterior {state_1}", values.posterior_1, values.posterior),
        MetricDisplay(
            f"Next-hidden {state_0}",
            values.next_hidden_0,
            "Hidden-state distribution used for the next observation.",
        ),
        MetricDisplay(
            f"Next-hidden {state_1}",
            values.next_hidden_1,
            "Hidden-state distribution used for the next observation.",
        ),
    )
    with st.container(key="summary-metrics"):
        columns = st.columns(len(metrics))
        for column, metric in zip(columns, metrics, strict=True):
            _ = column.metric(
                label=metric.label,
                value=metric.value,
                help=metric.help_text,
            )
    if values.depth == 0:
        _ = st.info(
            joined_text(
                (
                    "At depth 0, the hidden posterior is unavailable before any ",
                    "observation. Under this convention, next-hidden equals pi.",
                )
            )
        )
    _render_observed_entropy(success)
    _render_actual_target(success)


def _render_observed_entropy(success: CalculationSuccess) -> None:
    _ = st.subheader("Observed-symbol Shannon entropy")
    _ = st.markdown("This is descriptive and not HMM next-target predictive entropy.")
    observed_entropy = success.analysis.observed_entropy_bits
    if observed_entropy is None:
        _ = st.info(
            "Observed-symbol Shannon entropy is unavailable for an empty sequence."
        )
        return
    _ = st.metric(
        "Observed-symbol Shannon entropy (bits)",
        format_decimal(observed_entropy),
        help="Entropy of the empirical variable frequencies in the entered sequence.",
    )


def _render_actual_target(success: CalculationSuccess) -> None:
    assessment = success.target_assessment
    if assessment is None:
        return
    target_label = success.model.labels.observables[assessment.actual_target_index]
    match assessment.classification:  # noqa: RUF100  # noqa: MATCH_OK
        case TargetClassification.MODAL:
            wording = "The selected target is modal (highest probability)."
        case TargetClassification.LOWER_PROBABILITY:
            wording = "The selected target has lower probability than the modal target."
        case TargetClassification.TIED:
            wording = "The two target probabilities are tied."
    _ = st.subheader("Actual next target assessment")
    with st.container(key="actual-target-metrics"):
        target_columns = st.columns(3)
        _ = target_columns[0].metric("Actual next target", target_label)
        _ = target_columns[1].metric(
            "Actual-target probability", format_decimal(assessment.probability)
        )
        _ = target_columns[2].metric(
            "Realized surprisal (bits)",
            format_information(assessment.surprisal_bits),
        )
    _ = st.markdown(wording)
