"""Submitted-result lifecycle and selected-method rendering."""

from dataclasses import dataclass

import streamlit as st

from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.ui.comparison import render_comparison
from binary_entropy.ui.markov_view import render_markov_result
from binary_entropy.ui.session import (
    WorkbenchCalculationRecord,
    WorkbenchSubmissionFailure,
)
from binary_entropy.ui.shannon_results import render_shannon_result
from binary_entropy.ui.summary import render_hmm_result
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import format_ui_decimal
from binary_entropy.ui.vmm_view import render_vmm_result
from binary_entropy.ui.workbench_state import (
    MarkovWorkflow,
    MethodCalculationFailure,
    MethodChoice,
    WorkbenchCalculationSuccess,
    WorkbenchForm,
)
from binary_entropy.vmm_types import VMMAnalysis
from binary_entropy.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class CurrentSubmission:
    """Current method results, failures, and stale selected methods."""

    results: tuple[WorkbenchResult, ...]
    failures: tuple[MethodCalculationFailure, ...]
    stale_methods: tuple[MethodChoice, ...]


def render_results(
    form: WorkbenchForm,
    record: WorkbenchCalculationRecord | None,
    failure: WorkbenchSubmissionFailure | None,
) -> None:
    """Render only results whose shared and method-specific inputs are current."""
    _ = st.header("Results")
    if failure is not None:
        _ = st.error(failure.message)
        return
    if record is None:
        _ = st.info(
            joined_text(
                (
                    "Results are not calculated. Review the selected controls and ",
                    "choose Calculate selected methods.",
                )
            )
        )
        return
    current = _current_submission(form, record.success)
    if current.stale_methods:
        names = ", ".join(method.value for method in current.stale_methods)
        _ = st.warning(
            f"Recalculation required: {names}. Prior dependent outputs are hidden."
        )
    if not current.results and not current.failures:
        return
    if not current.stale_methods and not current.failures:
        _ = st.success("Calculation complete.")
    if current.results:
        _ = st.caption("Wide comparison and result tables scroll horizontally.")
        render_comparison(current.results, form.intake.observable_labels)
    for method in form.methods:
        result = _result_for_method(current.results, method)
        method_failure = _failure_for_method(current.failures, method)
        if result is not None:
            _render_method_result(result, form, record.success)
        elif method_failure is not None:
            _ = st.subheader(method.value)
            _ = st.error(method_failure.message)
    _render_reproducibility(form, record.success)


def _current_submission(
    form: WorkbenchForm,
    success: WorkbenchCalculationSuccess,
) -> CurrentSubmission:
    fingerprints = dict(success.fingerprints)
    stale = tuple(
        method
        for method in form.methods
        if fingerprints.get(method) != form.method_fingerprint(method)
    )
    current_methods = tuple(method for method in form.methods if method not in stale)
    results = tuple(
        result
        for result in success.results
        if _method_for_result(result) in current_methods
    )
    failures = tuple(
        failure for failure in success.failures if failure.method in current_methods
    )
    return CurrentSubmission(results, failures, stale)


def _render_method_result(
    result: WorkbenchResult,
    form: WorkbenchForm,
    success: WorkbenchCalculationSuccess,
) -> None:
    match result:
        case VMMAnalysis() as vmm:
            render_vmm_result(vmm, success.dataset)
        case MarkovBatchAnalysis() as markov:
            render_markov_result(markov)
        case HMMBatchAnalysis() as hmm:
            render_hmm_result(hmm, form)
        case ShannonBatchAnalysis() as shannon:
            has_targets = any(
                record.actual_target_index is not None
                for record in success.dataset.records
            )
            render_shannon_result(shannon, has_targets=has_targets)


def _method_for_result(result: WorkbenchResult) -> MethodChoice:
    match result:
        case VMMAnalysis():
            return MethodChoice.MARKOV
        case MarkovBatchAnalysis():
            return MethodChoice.MARKOV
        case HMMBatchAnalysis():
            return MethodChoice.HMM
        case ShannonBatchAnalysis():
            return MethodChoice.SHANNON


def _result_for_method(
    results: tuple[WorkbenchResult, ...],
    method: MethodChoice,
) -> WorkbenchResult | None:
    return next(
        (result for result in results if _method_for_result(result) is method), None
    )


def _failure_for_method(
    failures: tuple[MethodCalculationFailure, ...],
    method: MethodChoice,
) -> MethodCalculationFailure | None:
    return next((failure for failure in failures if failure.method is method), None)


def _render_reproducibility(
    form: WorkbenchForm,
    success: WorkbenchCalculationSuccess,
) -> None:
    selected_methods = ", ".join(method.value for method in form.methods)
    markov_details: tuple[str, ...] = ()
    if MethodChoice.MARKOV in form.methods:
        match form.markov.workflow:
            case MarkovWorkflow.VMM:
                smoothing = form.markov.vmm_smoothing()
                markov_details = (
                    f"- Markov workflow: {form.markov.workflow.value}",
                    f"- VMM smoothing: {form.markov.vmm_smoothing_choice.value}",
                    f"- VMM alpha: {format_ui_decimal(smoothing.alpha)}",
                    f"- VMM minimum context support: {form.markov.minimum_support}",
                    "- VMM backoff: deepest supported suffix, then shorter suffixes",
                )
            case MarkovWorkflow.FIRST_ORDER:
                markov_details = (
                    f"- Markov workflow: {form.markov.workflow.value}",
                    "- Markov order: 1",
                )
    with st.expander("Reproducibility details"):
        _ = st.markdown(
            "\n".join(
                (
                    f"- Selected methods: {selected_methods}",
                    f"- Parsed records: {len(success.dataset.records)}",
                    "- Visible scientific precision: 3 decimal places",
                    joined_text(
                        (
                            "- Raw export precision: unrounded float64 or at least ",
                            "12 decimal places",
                        )
                    ),
                    *markov_details,
                    "- Record ordering: deterministic submitted order",
                    joined_text(
                        (
                            "- Evaluation targets score existing predictions and ",
                            "never affect fitting",
                        )
                    ),
                )
            )
        )
