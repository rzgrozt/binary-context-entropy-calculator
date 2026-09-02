"""Experimental VMM downloads and reproducibility details."""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.tokens import format_ui_decimal
from binary_entropy.ui.vmm_evidence import (
    vmm_backoff_reason,
    vmm_backoff_selection,
    vmm_estimation_rule,
)
from binary_entropy.vmm_serialization import (
    vmm_context_evidence_csv,
    vmm_context_model_json,
    vmm_evaluation_csv,
)
from binary_entropy.vmm_types import VMMAnalysis, VMMRecordAnalysis, VMMResultScope

VMM_EXPERIMENTAL_NOTICE: Final = (
    "Experimental raw VMM artifacts: retain each file's schema, source details, "
    "and experimental-status notice when reused."
)


@dataclass(frozen=True, slots=True)
class VMMDownloadArtifact:
    """One ready raw result artifact with deterministic response metadata."""

    name: str
    label: str
    data: str | bytes
    file_name: str
    mime: str


def vmm_download_artifacts(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> tuple[VMMDownloadArtifact, ...]:
    """Build all three raw artifacts from the public VMM serializers."""
    scope = analysis.result_scope.value.replace("_", "-")
    return (
        VMMDownloadArtifact(
            name="Context model export",
            label="Download Context model export (JSON)",
            data=vmm_context_model_json(analysis, dataset),
            file_name=f"vmm-{scope}-context-model.json",
            mime="application/json",
        ),
        VMMDownloadArtifact(
            name="Context evidence export",
            label="Download Context evidence export (CSV)",
            data=vmm_context_evidence_csv(analysis, dataset),
            file_name=f"vmm-{scope}-context-evidence.csv",
            mime="text/csv; charset=utf-8",
        ),
        VMMDownloadArtifact(
            name="Evaluation export",
            label="Download Evaluation export (CSV)",
            data=vmm_evaluation_csv(analysis, dataset),
            file_name=f"vmm-{scope}-evaluation.csv",
            mime="text/csv; charset=utf-8",
        ),
    )


def vmm_reproducibility_lines(
    analysis: VMMAnalysis,
    dataset: SequenceDataset,
) -> tuple[str, ...]:
    """Describe the retained settings, stimuli, precision, and ordering."""
    pairs = tuple(zip(analysis.records, dataset.records, strict=True))
    minimum_support = analysis.config.minimum_support
    identifiers = ", ".join(str(source.sequence_id) for _, source in pairs)
    lengths = ", ".join(
        f"{source.sequence_id}={len(source.sequence)}" for _, source in pairs
    )
    requested = ", ".join(
        f"{record.sequence_id}={record.depth_rows[0].depth}..{record.depth_rows[-1].depth}"
        for record, _ in pairs
    )
    actual = ", ".join(
        f"{record.sequence_id}={_available_depth(record)}" for record, _ in pairs
    )
    selections = ", ".join(
        f"{record.sequence_id}={vmm_backoff_selection(record)}" for record, _ in pairs
    )
    reasons = "; ".join(
        f"{record.sequence_id}={_backoff_reason(record, analysis)}"
        for record, _ in pairs
    )
    availability = ", ".join(
        f"{record.sequence_id}={_availability(record)}" for record, _ in pairs
    )
    targets = ", ".join(
        f"{record.sequence_id}={_evaluation_status(record, source)}"
        for record, source in pairs
    )
    return (
        "Method: Variable-order Markov",
        "Assumption: finite suffix contexts; longer contexts are not assumed better",
        f"Estimator: {vmm_estimation_rule(analysis.config.smoothing)}",
        f"Smoothing alpha: {format_ui_decimal(analysis.config.smoothing.alpha)}",
        f"Support rule: minimum_support={minimum_support}",
        f"Sparse rule: support below {minimum_support} is sparse",
        "Backoff rule: deepest supported suffix, then shorter suffixes",
        f"Result scope: {analysis.result_scope.value}",
        f"Pooled rule: {_pooled_rule(analysis.result_scope)}",
        "Training dataset role: training",
        "Evaluation dataset role: in-sample when a training target is supplied",
        "Evaluation dataset identifier: not present",
        f"Parsed record count: {len(pairs)}",
        f"Accepted record IDs in source order: {identifiers}",
        "Rejected record IDs: none in the calculated dataset",
        f"Record IDs in source order: {identifiers}",
        f"Sequence lengths in source order: {lengths}",
        f"Requested depths: {requested}",
        f"Actual selected depths: {actual}",
        f"Backoff selections: {selections}",
        f"Backoff reasons: {reasons}",
        f"Prediction availability: {availability}",
        f"Target evaluation: {targets}",
        "Units: predictive entropy and target surprisal in bits",
        "Visible precision: exactly 3 decimal places",
        "Raw export precision: unrounded float64 or at least 12 decimal places",
        "Ordering: submitted record order, then ascending requested depth",
    )


def render_vmm_artifacts(analysis: VMMAnalysis, dataset: SequenceDataset) -> None:
    """Render ready downloads and their adjacent reproducibility record."""
    _ = st.subheader("Raw VMM artifacts")
    _ = st.info(VMM_EXPERIMENTAL_NOTICE)
    scope = analysis.result_scope.value
    _ = st.caption(f"All three exports are ready for the current {scope} result.")
    columns = st.columns(3)
    for column, artifact in zip(
        columns,
        vmm_download_artifacts(analysis, dataset),
        strict=True,
    ):
        _ = column.download_button(
            artifact.label,
            data=artifact.data,
            file_name=artifact.file_name,
            mime=artifact.mime,
            help=f"Experimental {artifact.name} for the current VMM result.",
            on_click="ignore",
        )
    with st.expander("VMM reproducibility details"):
        _ = st.text("\n".join(vmm_reproducibility_lines(analysis, dataset)))


def _available_depth(record: VMMRecordAnalysis) -> str:
    depth = record.effective_context_depth
    return "unavailable" if depth is None else str(depth)


def _pooled_rule(result_scope: VMMResultScope) -> str:
    match result_scope:
        case VMMResultScope.POOLED:
            return "sum within-record context counts without crossing record boundaries"
        case VMMResultScope.PER_SEQUENCE:
            return "not applied; each record is fitted independently"


def _availability(record: VMMRecordAnalysis) -> str:
    return "available" if record.predictive_entropy_bits is not None else "unavailable"


def _backoff_reason(record: VMMRecordAnalysis, analysis: VMMAnalysis) -> str:
    reason = vmm_backoff_reason(record, analysis.config.smoothing)
    return "none (requested depth selected)" if reason is None else reason


def _evaluation_status(
    record: VMMRecordAnalysis,
    source: SequenceRecord,
) -> str:
    if source.actual_target_index is None:
        return "Not supplied"
    if record.target_assessment is None:
        return "In-sample evaluation unavailable"
    return "In-sample evaluation, not held out"
