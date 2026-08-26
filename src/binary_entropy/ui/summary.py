"""HMM summaries, prefix evidence, and compatibility exports."""

from dataclasses import dataclass

import streamlit as st

from binary_entropy.domain import (
    BinaryHMM,
    SequenceAnalysis,
    TargetAssessment,
    TargetClassification,
)
from binary_entropy.methods.hmm import HMMBatchAnalysis, HMMRecordAnalysis
from binary_entropy.serialization import (
    CandidateMetadata,
    candidate_summary_csv,
    prefix_csv,
)
from binary_entropy.ui.chart import entropy_figure
from binary_entropy.ui.results import (
    final_metrics,
    format_information,
    prefix_dataframe,
)
from binary_entropy.ui.state import (
    CalculationSuccess,
    CalculatorForm,
    PresetExportFailure,
    PresetExportSuccess,
    export_preset,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT, format_ui_decimal
from binary_entropy.ui.workbench_state import WorkbenchForm


@dataclass(frozen=True, slots=True)
class MetricDisplay:
    """One labeled value and its scientific qualification."""

    label: str
    value: str
    help_text: str


@dataclass(frozen=True, slots=True)
class HMMViewContext:
    """Submitted HMM model, metadata, and download-label scope."""

    form: WorkbenchForm
    model: BinaryHMM
    single_record: bool


def render_hmm_result(analysis: HMMBatchAnalysis, form: WorkbenchForm) -> None:
    """Render every independent HMM result under the submitted fixed model."""
    context = HMMViewContext(
        form=form,
        model=form.hmm_model.to_model(),
        single_record=len(analysis.records) == 1,
    )
    _ = st.subheader("Hidden Markov Model")
    _ = st.caption(
        joined_text(
            (
                "Each sequence is filtered independently under the submitted fixed ",
                "HMM; no pooled fit is performed.",
            )
        )
    )
    for record in analysis.records:
        if context.single_record:
            _render_record(record, context)
        else:
            with st.expander(f"HMM sequence: {record.sequence_id}", expanded=True):
                _render_record(record, context)
    _render_preset_download(context)


def render_summary(success: CalculationSuccess) -> None:
    """Preserve the legacy single-analysis summary adapter."""
    _render_metrics(
        success.analysis,
        success.model,
        success.target_assessment,
    )


def _render_record(record: HMMRecordAnalysis, context: HMMViewContext) -> None:
    _ = st.markdown(f"**Sequence ID:** `{record.sequence_id}`")
    _render_metrics(record.analysis, context.model, record.target_assessment)
    _ = st.markdown("Every-prefix HMM values in deterministic depth order.")
    labels = context.model.labels.observables
    states = context.model.labels.states
    _ = st.dataframe(
        prefix_dataframe(record.analysis, context.model),
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
            f"Surprisal if next {labels[0]} (bits)": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"Surprisal if next {labels[1]} (bits)": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"Posterior {states[0]}": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"Posterior {states[1]}": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"Next-hidden {states[0]}": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"Next-hidden {states[1]}": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
        },
    )
    _ = st.markdown(
        "HMM predictive entropy by context depth, from 0.000 to 1.000 bits."
    )
    _ = st.write(entropy_figure(record.analysis))
    _render_record_downloads(record, context)


def _render_metrics(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
    target_assessment: TargetAssessment | None,
) -> None:
    values = final_metrics(analysis, model)
    observable_0, observable_1 = model.labels.observables
    state_0, state_1 = model.labels.states
    metrics = (
        MetricDisplay("Context depth", str(values.depth), "Observations consumed."),
        MetricDisplay("Observed context", values.context, "Complete entered prefix."),
        MetricDisplay(
            f"P(next {observable_0})", values.probability_0, "HMM prediction."
        ),
        MetricDisplay(
            f"P(next {observable_1})", values.probability_1, "HMM prediction."
        ),
        MetricDisplay(
            "Predicted target", values.predicted_target, "Ties choose observable A."
        ),
        MetricDisplay(
            "HMM predictive entropy (bits)",
            values.entropy_bits,
            "Next-symbol uncertainty.",
        ),
        MetricDisplay(
            f"Candidate surprisal {observable_0} (bits)",
            values.surprisal_0,
            "Self-information.",
        ),
        MetricDisplay(
            f"Candidate surprisal {observable_1} (bits)",
            values.surprisal_1,
            "Self-information.",
        ),
        MetricDisplay(f"Posterior {state_0}", values.posterior_0, values.posterior),
        MetricDisplay(f"Posterior {state_1}", values.posterior_1, values.posterior),
        MetricDisplay(
            f"Next-hidden {state_0}", values.next_hidden_0, "Next hidden distribution."
        ),
        MetricDisplay(
            f"Next-hidden {state_1}", values.next_hidden_1, "Next hidden distribution."
        ),
    )
    columns = st.columns(3)
    for index, metric in enumerate(metrics):
        _ = columns[index % 3].metric(
            label=metric.label,
            value=metric.value,
            help=metric.help_text,
        )
    if values.depth == 0:
        _ = st.info(
            "At depth 0, the hidden posterior is unavailable; next-hidden equals pi."
        )
    if target_assessment is not None:
        _render_target(target_assessment, model)


def _render_target(assessment: TargetAssessment, model: BinaryHMM) -> None:
    target_label = model.labels.observables[assessment.actual_target_index]
    match assessment.classification:
        case TargetClassification.MODAL:
            wording = "The selected target is modal (highest probability)."
        case TargetClassification.LOWER_PROBABILITY:
            wording = "The selected target has lower probability than the modal target."
        case TargetClassification.TIED:
            wording = "The two target probabilities are tied."
    columns = st.columns(3)
    _ = columns[0].metric("Actual next target", target_label)
    _ = columns[1].metric(
        "Actual-target probability",
        format_ui_decimal(assessment.probability),
    )
    _ = columns[2].metric(
        "Realized surprisal (bits)",
        format_information(assessment.surprisal_bits),
    )
    _ = st.markdown(wording)


def _render_record_downloads(
    record: HMMRecordAnalysis,
    context: HMMViewContext,
) -> None:
    suffix = "" if context.single_record else f" — {record.sequence_id}"
    metadata = CandidateMetadata(
        sequence_id=record.sequence_id,
        preset_name=context.form.preset_name,
        actual_target_index=(
            None
            if record.target_assessment is None
            else record.target_assessment.actual_target_index
        ),
    )
    columns = st.columns(2)
    _ = columns[0].download_button(
        f"Download HMM prefix CSV{suffix}",
        data=prefix_csv(record.analysis, context.model),
        file_name=f"hmm-{record.sequence_id}-prefix.csv",
        mime="text/csv; charset=utf-8",
        on_click="ignore",
    )
    _ = columns[1].download_button(
        f"Download HMM candidate-summary CSV{suffix}",
        data=candidate_summary_csv(record.analysis, context.model, metadata),
        file_name=f"hmm-{record.sequence_id}-candidate-summary.csv",
        mime="text/csv; charset=utf-8",
        on_click="ignore",
    )


def _render_preset_download(context: HMMViewContext) -> None:
    legacy = CalculatorForm(
        model=context.form.hmm_model,
        sequence_text=context.form.intake.text,
        actual_target=context.form.intake.actual_target,
        sequence_id=context.form.intake.sequence_id,
        preset_name=context.form.preset_name,
    )
    match export_preset(legacy):
        case PresetExportSuccess(payload=payload):
            _ = st.download_button(
                "Download HMM preset JSON",
                data=payload,
                file_name="binary-hmm-preset.json",
                mime="application/json",
                on_click="ignore",
            )
        case PresetExportFailure(message=message):
            _ = st.error(f"HMM preset unavailable: {message}")
