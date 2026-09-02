"""Canonical visible status values for retained VMM evidence."""

from binary_entropy.domain import TargetAssessment, TargetClassification
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMDepthAnalysis,
    VMMDepthStatus,
    VMMRecordAnalysis,
    VMMSmoothing,
)


def vmm_evidence_label(status: VMMDepthStatus) -> str:
    """Name whether one depth is accepted or why it is rejected."""
    match status:
        case VMMDepthStatus.ACCEPTED:
            return "Accepted"
        case VMMDepthStatus.LOW_SUPPORT:
            return "Rejected - low support"
        case VMMDepthStatus.UNAVAILABLE:
            return "Rejected - unavailable"


def vmm_target_classification_label(assessment: TargetAssessment) -> str:
    """Return the target-assessment classification as visible text."""
    match assessment.classification:
        case TargetClassification.MODAL:
            return "Modal"
        case TargetClassification.LOWER_PROBABILITY:
            return "Lower probability"
        case TargetClassification.TIED:
            return "Tied"


def vmm_support_status(status: VMMDepthStatus) -> tuple[str, str]:
    """Return separate support and sparse status values."""
    match status:
        case VMMDepthStatus.ACCEPTED:
            return "accepted", "not_sparse"
        case VMMDepthStatus.LOW_SUPPORT:
            return "low_support", "sparse"
        case VMMDepthStatus.UNAVAILABLE:
            return "unavailable", "unavailable"


def vmm_estimation_rule(smoothing: VMMSmoothing) -> str:
    """Name the estimator without inferring it from alpha."""
    match smoothing:
        case MLESmoothing():
            return "maximum_likelihood"
        case KTSmoothing():
            return "krichevsky_trofimov"
        case AdditiveSmoothing():
            return "additive_smoothing"


def vmm_context_reason(
    status: VMMDepthStatus,
    smoothing: VMMSmoothing,
) -> str | None:
    """Explain why one requested context has no accepted evidence."""
    match status:
        case VMMDepthStatus.ACCEPTED:
            return None
        case VMMDepthStatus.LOW_SUPPORT:
            return "Context support is below the configured minimum."
        case VMMDepthStatus.UNAVAILABLE:
            match smoothing:
                case MLESmoothing():
                    return (
                        "MLE unavailable: unseen context has no occurrences in the "
                        "training dataset."
                    )
                case KTSmoothing() | AdditiveSmoothing():
                    return "Unseen context has no occurrences in the training dataset."


def vmm_depth_selection(
    record: VMMRecordAnalysis,
    row: VMMDepthAnalysis,
) -> str:
    """Name one row's role in explicit suffix selection."""
    if record.effective_context_depth is None:
        return "no_context_selected"
    if row.depth == record.effective_context_depth:
        return "selected"
    if row.depth > record.effective_context_depth:
        return "rejected_for_backoff"
    return "not_selected_shorter_context"


def vmm_backoff_selection(record: VMMRecordAnalysis) -> str:
    """Name the final requested-depth or suffix-backoff outcome."""
    if record.effective_context_depth is None:
        return "unavailable"
    if record.effective_context_depth == len(record.sequence):
        return "requested_depth_selected"
    return "backed_off_to_shorter_suffix"


def vmm_backoff_reason(
    record: VMMRecordAnalysis,
    smoothing: VMMSmoothing,
) -> str | None:
    """Explain why the final selection used a shorter suffix."""
    if record.effective_context_depth == len(record.sequence):
        return None
    reason = vmm_context_reason(record.depth_rows[-1].status, smoothing)
    return reason or "No context meets the configured minimum support."
