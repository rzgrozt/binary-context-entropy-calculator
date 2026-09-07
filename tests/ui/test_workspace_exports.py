import csv
import importlib
import io
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from binary_entropy.markov_batch_serialization import markov_batch_summary_csv
from binary_entropy.markov_serialization import markov_model_json, markov_sequence_csv
from binary_entropy.markov_types import MarkovBatchAnalysis
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.serialization import (
    CandidateMetadata,
    candidate_summary_csv,
    prefix_csv,
)
from binary_entropy.ui.session import WorkbenchCalculationRecord
from binary_entropy.ui.workbench_state import MarkovWorkflow, WorkbenchForm
from binary_entropy.vmm_serialization import (
    vmm_context_evidence_csv,
    vmm_context_model_json,
    vmm_evaluation_csv,
)
from binary_entropy.vmm_types import VMMAnalysis
from tests.ui.workspace_support import calculated_workspace


class _Artifact(Protocol):
    name: str
    data: str | bytes
    record_id: str | None


@runtime_checkable
class _ExportsApi(Protocol):
    complete_artifacts: Callable[
        [WorkbenchCalculationRecord, WorkbenchForm], tuple[_Artifact, ...]
    ]
    selected_artifacts: Callable[
        [WorkbenchCalculationRecord, WorkbenchForm, tuple[str, ...]],
        tuple[_Artifact, ...],
    ]


def _exports_api() -> _ExportsApi:
    module = importlib.import_module("binary_entropy.ui.workspace_exports")
    assert isinstance(module, _ExportsApi)
    return module


def _artifact_map(artifacts: Sequence[_Artifact]) -> dict[str, str | bytes]:
    return {artifact.name: artifact.data for artifact in artifacts}


def _csv_ids(payload: str, column: str) -> tuple[str, ...]:
    return tuple(row[column] for row in csv.DictReader(io.StringIO(payload)))


def test_complete_markov_exports_when_built_match_existing_serializers() -> None:
    # Given
    form, record = calculated_workspace(MarkovWorkflow.FIRST_ORDER)
    result = next(
        item for item in record.success.results if isinstance(item, MarkovBatchAnalysis)
    )
    api = _exports_api()

    # When
    artifacts = _artifact_map(api.complete_artifacts(record, form))

    # Then
    assert artifacts["Markov model JSON"] == markov_model_json(result)
    assert artifacts["Markov prefix CSV"] == markov_sequence_csv(result)
    assert artifacts["Markov batch-summary CSV"] == markov_batch_summary_csv(result)


def test_complete_vmm_exports_when_built_match_existing_serializers() -> None:
    # Given
    form, record = calculated_workspace()
    result = next(
        item for item in record.success.results if isinstance(item, VMMAnalysis)
    )
    api = _exports_api()

    # When
    artifacts = _artifact_map(api.complete_artifacts(record, form))

    # Then
    assert artifacts["Context model export"] == vmm_context_model_json(
        result, record.success.dataset
    )
    assert artifacts["Context evidence export"] == vmm_context_evidence_csv(
        result, record.success.dataset
    )
    assert artifacts["Evaluation export"] == vmm_evaluation_csv(
        result, record.success.dataset
    )


def test_complete_hmm_exports_when_built_match_existing_serializers() -> None:
    # Given
    form, record = calculated_workspace()
    hmm = next(
        item for item in record.success.results if isinstance(item, HMMBatchAnalysis)
    )
    api = _exports_api()

    # When
    artifacts = _artifact_map(api.complete_artifacts(record, form))

    # Then
    for result in hmm.records:
        identifier = str(result.sequence_id)
        metadata = CandidateMetadata(
            sequence_id=identifier,
            preset_name=form.preset_name,
            actual_target_index=None,
        )
        assert artifacts[f"HMM prefix CSV — {identifier}"] == prefix_csv(
            result.analysis, form.hmm_model.to_model()
        )
        assert artifacts[f"HMM candidate-summary CSV — {identifier}"] == (
            candidate_summary_csv(
                result.analysis,
                form.hmm_model.to_model(),
                metadata,
            )
        )


def test_selected_markov_and_vmm_exports_contain_only_source_order_ids() -> None:
    # Given
    api = _exports_api()
    first_form, first_record = calculated_workspace(MarkovWorkflow.FIRST_ORDER)
    vmm_form, vmm_record = calculated_workspace()
    clicked_ids = ("sequence-003", "sequence-001")

    # When
    markov = _artifact_map(
        api.selected_artifacts(first_record, first_form, clicked_ids)
    )
    vmm = _artifact_map(api.selected_artifacts(vmm_record, vmm_form, clicked_ids))

    # Then
    expected = ("sequence-001", "sequence-003")
    assert isinstance(markov["Markov prefix CSV"], str)
    assert isinstance(markov["Markov batch-summary CSV"], str)
    assert isinstance(vmm["Context evidence export"], str)
    assert isinstance(vmm["Evaluation export"], str)
    assert isinstance(vmm["Context model export"], bytes)
    assert _csv_ids(markov["Markov prefix CSV"], "sequence_id") == (
        "sequence-001",
        "sequence-001",
        "sequence-001",
        "sequence-001",
        "sequence-001",
        "sequence-003",
        "sequence-003",
        "sequence-003",
        "sequence-003",
        "sequence-003",
    )
    assert _csv_ids(markov["Markov batch-summary CSV"], "sequence_id") == expected
    evidence_ids = _csv_ids(vmm["Context evidence export"], "record_identifier")
    assert evidence_ids[0] == expected[0]
    assert set(_csv_ids(vmm["Context evidence export"], "record_identifier")) == set(
        expected
    )
    assert _csv_ids(vmm["Evaluation export"], "record_identifier") == expected
    model = vmm["Context model export"].decode("utf-8")
    assert model.count('"record_identifier"') == 2
    assert model.index("sequence-001") < model.index("sequence-003")
    assert "sequence-002" not in model


def test_selected_hmm_exports_contain_only_selected_ids_in_source_order() -> None:
    # Given
    form, record = calculated_workspace()
    hmm = next(
        item for item in record.success.results if isinstance(item, HMMBatchAnalysis)
    )
    api = _exports_api()
    clicked_ids = ("sequence-003", "sequence-001")

    # When
    artifacts = api.selected_artifacts(record, form, clicked_ids)
    hmm_artifacts = tuple(item for item in artifacts if item.name.startswith("HMM "))

    # Then
    assert tuple(item.record_id for item in hmm_artifacts) == (
        "sequence-001",
        "sequence-001",
        "sequence-003",
        "sequence-003",
    )
    by_name = _artifact_map(hmm_artifacts)
    selected_records = (hmm.records[0], hmm.records[2])
    for selected in selected_records:
        identifier = str(selected.sequence_id)
        metadata = CandidateMetadata(
            sequence_id=identifier,
            preset_name=form.preset_name,
            actual_target_index=None,
        )
        assert by_name[f"HMM prefix CSV — {identifier}"] == prefix_csv(
            selected.analysis, form.hmm_model.to_model()
        )
        assert by_name[f"HMM candidate-summary CSV — {identifier}"] == (
            candidate_summary_csv(
                selected.analysis,
                form.hmm_model.to_model(),
                metadata,
            )
        )


def test_workspace_exports_when_shannon_is_present_remain_export_free() -> None:
    # Given
    form, record = calculated_workspace()
    api = _exports_api()

    # When
    complete = api.complete_artifacts(record, form)
    selected = api.selected_artifacts(record, form, ("sequence-001",))

    # Then
    assert not any("shannon" in item.name.lower() for item in (*complete, *selected))
