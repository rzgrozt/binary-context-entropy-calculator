"""Current-result filtering for edited workbench forms."""

from dataclasses import dataclass, replace
from typing import assert_never

from bspe.markov_types import MarkovBatchAnalysis
from bspe.methods.hmm import HMMBatchAnalysis
from bspe.methods.shannon import ShannonBatchAnalysis
from bspe.ui.session import WorkbenchCalculationRecord
from bspe.ui.workbench_state import (
    MethodCalculationFailure,
    MethodChoice,
    WorkbenchCalculationSuccess,
    WorkbenchForm,
)
from bspe.vmm_types import VMMAnalysis
from bspe.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class CurrentSubmission:
    """Current method results, failures, and stale selected methods."""

    results: tuple[WorkbenchResult, ...]
    failures: tuple[MethodCalculationFailure, ...]
    stale_methods: tuple[MethodChoice, ...]


def current_submission(
    form: WorkbenchForm,
    success: WorkbenchCalculationSuccess,
) -> CurrentSubmission:
    """Filter a stored calculation against method-specific fingerprints."""
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
        if method_for_result(result) in current_methods
    )
    failures = tuple(
        failure for failure in success.failures if failure.method in current_methods
    )
    return CurrentSubmission(results, failures, stale)


def current_calculation_record(
    form: WorkbenchForm,
    record: WorkbenchCalculationRecord | None,
) -> WorkbenchCalculationRecord | None:
    """Return a snapshot containing only computationally current results."""
    if record is None:
        return None
    current = current_submission(form, record.success)
    return WorkbenchCalculationRecord(
        replace(
            record.success,
            results=current.results,
            failures=current.failures,
        )
    )


def method_for_result(result: WorkbenchResult) -> MethodChoice:
    """Map an immutable scientific result to its top-level UI method."""
    match result:
        case VMMAnalysis() | MarkovBatchAnalysis():
            return MethodChoice.MARKOV
        case HMMBatchAnalysis():
            return MethodChoice.HMM
        case ShannonBatchAnalysis():
            return MethodChoice.SHANNON
    assert_never(result)


def result_for_method(
    results: tuple[WorkbenchResult, ...],
    method: MethodChoice,
) -> WorkbenchResult | None:
    """Return the current result matching one selected top-level method."""
    return next(
        (result for result in results if method_for_result(result) is method),
        None,
    )


def failure_for_method(
    failures: tuple[MethodCalculationFailure, ...],
    method: MethodChoice,
) -> MethodCalculationFailure | None:
    """Return the current failure matching one selected top-level method."""
    return next((failure for failure in failures if failure.method is method), None)
