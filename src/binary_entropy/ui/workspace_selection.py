"""Immutable sequence-scope projection of calculated workbench results."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import assert_never

from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.records import SequenceDataset
from binary_entropy.ui.session import WorkbenchCalculationRecord
from binary_entropy.vmm_types import VMMAnalysis
from binary_entropy.workbench import WorkbenchResult


class SequenceScope(StrEnum):
    """Visible record scopes in the result workspace."""

    ONE = "One"
    MULTIPLE = "Multiple"
    ALL = "All"


@dataclass(frozen=True, slots=True)
class WorkspaceSelection:
    """Presentation-only sequence scope and selected record identifiers."""

    scope: SequenceScope
    selected_ids: tuple[str, ...]


def reconcile_selection(
    selection: WorkspaceSelection,
    dataset: SequenceDataset,
) -> WorkspaceSelection:
    """Retain valid IDs in source order and provide the One-scope fallback."""
    source_ids = tuple(str(record.sequence_id) for record in dataset.records)
    selected = frozenset(selection.selected_ids)
    survivors = tuple(identifier for identifier in source_ids if identifier in selected)
    match selection.scope:
        case SequenceScope.ALL:
            return WorkspaceSelection(SequenceScope.ALL, source_ids)
        case SequenceScope.MULTIPLE:
            return WorkspaceSelection(SequenceScope.MULTIPLE, survivors)
        case SequenceScope.ONE:
            identifier = survivors[:1] or source_ids[:1]
            return WorkspaceSelection(SequenceScope.ONE, identifier)
    assert_never(selection.scope)


def project_record(
    record: WorkbenchCalculationRecord,
    selection: WorkspaceSelection,
) -> WorkbenchCalculationRecord:
    """Slice stored records without recalculating or changing fitted values."""
    reconciled = reconcile_selection(selection, record.success.dataset)
    selected = frozenset(reconciled.selected_ids)
    success = record.success
    if reconciled.scope is SequenceScope.ALL:
        return record
    dataset = SequenceDataset(
        success.dataset.labels,
        tuple(
            item
            for item in success.dataset.records
            if str(item.sequence_id) in selected
        ),
    )
    results = tuple(_project_result(result, selected) for result in success.results)
    return WorkbenchCalculationRecord(
        replace(success, dataset=dataset, results=results)
    )


def _project_result(
    result: WorkbenchResult,
    selected: frozenset[str],
) -> WorkbenchResult:
    match result:
        case VMMAnalysis(records=records):
            return replace(
                result,
                records=tuple(
                    item for item in records if str(item.sequence_id) in selected
                ),
            )
        case MarkovBatchAnalysis(records=records):
            return replace(
                result,
                records=tuple(
                    item for item in records if str(item.sequence_id) in selected
                ),
            )
        case HMMBatchAnalysis(records=records):
            return replace(
                result,
                records=tuple(
                    item for item in records if str(item.sequence_id) in selected
                ),
            )
        case ShannonBatchAnalysis(records=records):
            return replace(
                result,
                records=tuple(
                    item for item in records if str(item.sequence_id) in selected
                ),
            )
    assert_never(result)
