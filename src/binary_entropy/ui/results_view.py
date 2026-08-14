"""Result lifecycle, prefix evidence, chart, and export rendering."""

import streamlit as st

from binary_entropy.constants import (
    DISPLAY_DECIMALS,
    PRESET_SCHEMA_VERSION,
    PROBABILITY_TOLERANCE,
)
from binary_entropy.domain import BinaryHMM, SequenceAnalysis
from binary_entropy.serialization import (
    CandidateMetadata,
    candidate_summary_csv,
    prefix_csv,
)
from binary_entropy.ui.chart import entropy_figure
from binary_entropy.ui.results import prefix_dataframe
from binary_entropy.ui.session import CalculationRecord, SubmissionFailure
from binary_entropy.ui.state import (
    CalculatorForm,
    PresetExportFailure,
    PresetExportSuccess,
    actual_target_index,
    export_preset,
)
from binary_entropy.ui.summary import render_summary
from binary_entropy.ui.text import joined_text


def prefix_table_html(analysis: SequenceAnalysis, model: BinaryHMM) -> str:
    """Build the keyboard-reachable exact-results table with escaped cells."""
    table = prefix_dataframe(analysis, model).to_html(
        index=False,
        escape=True,
        border=0,
        classes="prefix-results-table",
    )
    return joined_text(
        (
            '<div class="prefix-table-overflow" role="region" tabindex="0" ',
            'aria-label="Every-prefix exact results" ',
            'aria-describedby="prefix-table-scroll-instruction">',
            '<p id="prefix-table-scroll-instruction" ',
            'class="prefix-table-scroll-instruction">',
            "Scroll horizontally to inspect every exact-value column.</p>",
            table,
            "</div>",
        )
    )


def render_results(
    form: CalculatorForm,
    record: CalculationRecord | None,
    failure: SubmissionFailure | None,
) -> None:
    """Render exactly one uncalculated, invalid, stale, or current result state."""
    _ = st.header("Results")
    if record is not None and record.fingerprint != form.fingerprint():
        _ = st.warning(
            joined_text(
                (
                    "Inputs changed after the last successful calculation. ",
                    "Recalculation is required; prior outputs and downloads ",
                    "are hidden.",
                )
            )
        )
        return
    if failure is not None:
        _ = st.error(failure.message)
        _render_preset_download(form)
        return
    if record is None:
        _ = st.info(
            joined_text(
                (
                    "Results are not calculated. Review the model and sequence, then ",
                    "select Calculate entropy.",
                )
            )
        )
        _render_preset_download(form)
        return
    _ = st.success("Calculation complete.")
    render_summary(record.success)
    _render_prefix_evidence(record)
    _render_downloads(form, record)
    _render_reproducibility(form, record)


def _render_prefix_evidence(record: CalculationRecord) -> None:
    success = record.success
    _ = st.subheader("Every-prefix results")
    _ = st.caption(
        joined_text(
            (
                "Depth is the number of observations already consumed and ",
                "begins at 0. ",
                "The table is the authoritative exact-value fallback for the chart.",
                " On narrow screens, scroll horizontally to inspect every column.",
            )
        )
    )
    _ = st.html(prefix_table_html(success.analysis, success.model))
    _ = st.subheader("Predictive entropy by context depth")
    _ = st.markdown(
        joined_text(
            (
                "The line and markers show HMM next-target uncertainty for every ",
                "prefix. Hover a marker for the canonical 12-decimal value.",
            )
        )
    )
    with st.container(key="entropy-chart"):
        _ = st.write(entropy_figure(success.analysis))


def _render_downloads(form: CalculatorForm, record: CalculationRecord) -> None:
    _ = st.subheader("Exports")
    _render_preset_download(form)
    success = record.success
    target_index = actual_target_index(form.actual_target)
    metadata = CandidateMetadata(
        sequence_id=form.sequence_id,
        preset_name=form.preset_name,
        actual_target_index=target_index,
    )
    with st.container(key="download-actions"):
        download_columns = st.columns(2)
        _ = download_columns[0].download_button(
            "Download prefix CSV",
            data=prefix_csv(success.analysis, success.model),
            file_name="binary-entropy-prefix.csv",
            mime="text/csv; charset=utf-8",
            on_click="ignore",
            type="secondary",
        )
        _ = download_columns[1].download_button(
            "Download candidate-summary CSV",
            data=candidate_summary_csv(success.analysis, success.model, metadata),
            file_name="binary-entropy-candidate-summary.csv",
            mime="text/csv; charset=utf-8",
            on_click="ignore",
            type="secondary",
        )


def _render_preset_download(form: CalculatorForm) -> None:
    match export_preset(form):  # noqa: RUF100  # noqa: MATCH_OK
        case PresetExportSuccess(payload=payload):
            _ = st.download_button(
                "Download model preset JSON",
                data=payload,
                file_name="binary-hmm-preset.json",
                mime="application/json",
                on_click="ignore",
                type="secondary",
            )
            _ = st.caption("Model parameters only; the sequence is not included.")
        case PresetExportFailure(message=message):
            _ = st.caption(f"Model preset JSON unavailable: {message}")


def _render_reproducibility(
    form: CalculatorForm,
    record: CalculationRecord,
) -> None:
    with st.expander("Reproducibility details"):
        _ = st.markdown(
            joined_text(
                (
                    "- Display and export precision: ",
                    f"{DISPLAY_DECIMALS} decimal places\n",
                    f"- Probability sum tolerance: {PROBABILITY_TOLERANCE:.12f}\n",
                    f"- Preset schema version: {PRESET_SCHEMA_VERSION}\n",
                    "- Parsed sequence length: ",
                    f"{len(record.success.analysis.sequence)}\n",
                    f"- Preset name: `{form.preset_name}`\n",
                    f"- Sequence ID: `{form.sequence_id}`",
                )
            )
        )
        match export_preset(form):  # noqa: RUF100  # noqa: MATCH_OK
            case PresetExportSuccess(payload=payload):
                _ = st.code(payload.decode("utf-8"), language="json")
            case PresetExportFailure(message=message):
                _ = st.error(message)
