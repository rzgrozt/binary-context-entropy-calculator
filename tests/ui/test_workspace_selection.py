import importlib
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import pytest

from bspe.records import SequenceDataset
from bspe.ui import workbench_state
from bspe.ui.session import WorkbenchCalculationRecord
from bspe.ui.workbench_state import (
    MarkovWorkflow,
    WorkbenchCalculationOutcome,
    WorkbenchForm,
)
from tests.ui.workspace_support import calculated_workspace


class _ScopeValue(Protocol):
    value: str


class _ScopeNamespace(Protocol):
    ALL: _ScopeValue
    MULTIPLE: _ScopeValue
    ONE: _ScopeValue


class _Selection(Protocol):
    selected_ids: tuple[str, ...]


@runtime_checkable
class _SelectionApi(Protocol):
    SequenceScope: _ScopeNamespace
    WorkspaceSelection: Callable[[_ScopeValue, tuple[str, ...]], _Selection]
    project_record: Callable[
        [WorkbenchCalculationRecord, _Selection], WorkbenchCalculationRecord
    ]
    reconcile_selection: Callable[[_Selection, SequenceDataset], _Selection]


def _selection_api() -> _SelectionApi:
    module = importlib.import_module("bspe.ui.workspace_selection")
    assert isinstance(module, _SelectionApi)
    return module


def _record_ids(record: WorkbenchCalculationRecord) -> tuple[str, ...]:
    return tuple(str(item.sequence_id) for item in record.success.dataset.records)


def test_projection_when_scope_is_all_preserves_dataset_source_order() -> None:
    # Given
    _, record = calculated_workspace()
    api = _selection_api()
    selection = api.WorkspaceSelection(api.SequenceScope.ALL, ())

    # When
    projected = api.project_record(record, selection)

    # Then
    assert isinstance(projected, WorkbenchCalculationRecord)
    assert _record_ids(projected) == (
        "sequence-001",
        "sequence-002",
        "sequence-003",
    )


def test_projection_when_ids_are_clicked_out_of_order_uses_dataset_order() -> None:
    # Given
    _, record = calculated_workspace()
    api = _selection_api()
    selection = api.WorkspaceSelection(
        api.SequenceScope.MULTIPLE,
        ("sequence-003", "sequence-001"),
    )

    # When
    projected = api.project_record(record, selection)

    # Then
    assert _record_ids(projected) == ("sequence-001", "sequence-003")


@pytest.mark.parametrize(
    ("scope_name", "prior_ids", "expected_ids"),
    [
        (
            "MULTIPLE",
            ("sequence-003", "removed", "sequence-001"),
            ("sequence-001", "sequence-003"),
        ),
        ("ONE", ("removed",), ("sequence-001",)),
    ],
)
def test_selection_when_recalculated_reconciles_survivors_or_safe_fallback(
    scope_name: str,
    prior_ids: tuple[str, ...],
    expected_ids: tuple[str, ...],
) -> None:
    # Given
    _, record = calculated_workspace()
    api = _selection_api()
    scopes = {
        "MULTIPLE": api.SequenceScope.MULTIPLE,
        "ONE": api.SequenceScope.ONE,
    }
    selection = api.WorkspaceSelection(scopes[scope_name], prior_ids)

    # When
    reconciled = api.reconcile_selection(selection, record.success.dataset)

    # Then
    assert reconciled.selected_ids == expected_ids


@pytest.mark.parametrize("workflow", [MarkovWorkflow.VMM, MarkovWorkflow.FIRST_ORDER])
def test_projection_when_subset_is_selected_slices_every_result_without_recalculation(
    workflow: MarkovWorkflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    _, record = calculated_workspace(workflow)

    def unexpected_calculation(_form: WorkbenchForm) -> WorkbenchCalculationOutcome:
        message = "selector projection invoked calculation"
        raise AssertionError(message)

    monkeypatch.setattr(workbench_state, "calculate_workbench", unexpected_calculation)
    api = _selection_api()
    selection = api.WorkspaceSelection(
        api.SequenceScope.MULTIPLE,
        ("sequence-003", "sequence-001"),
    )

    # When
    projected = api.project_record(record, selection)

    # Then
    assert _record_ids(projected) == ("sequence-001", "sequence-003")
    assert tuple(
        tuple(str(item.sequence_id) for item in result.records)
        for result in projected.success.results
    ) == (("sequence-001", "sequence-003"),) * 3
