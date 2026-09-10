"""Raw workbench artifacts for complete and selected record scopes."""

from dataclasses import dataclass
from typing import assert_never

from bspe.domain import BinaryHMM
from bspe.markov_batch_serialization import markov_batch_summary_csv
from bspe.markov_serialization import markov_model_json, markov_sequence_csv
from bspe.markov_types import MarkovBatchAnalysis
from bspe.methods.hmm import HMMBatchAnalysis, HMMRecordAnalysis
from bspe.methods.shannon import ShannonBatchAnalysis
from bspe.serialization import (
    CandidateMetadata,
    candidate_summary_csv,
    prefix_csv,
)
from bspe.ui.session import WorkbenchCalculationRecord
from bspe.ui.state import (
    CalculatorForm,
    PresetExportFailure,
    PresetExportSuccess,
    export_preset,
)
from bspe.ui.vmm_artifacts import vmm_download_artifacts
from bspe.ui.workbench_state import WorkbenchForm
from bspe.ui.workspace_selection import (
    SequenceScope,
    WorkspaceSelection,
    project_record,
)
from bspe.vmm_types import VMMAnalysis
from bspe.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class WorkspaceArtifact:
    """One deterministic download built by an existing serializer."""

    name: str
    data: str | bytes
    record_id: str | None
    file_name: str
    mime: str


def complete_artifacts(
    record: WorkbenchCalculationRecord,
    form: WorkbenchForm,
) -> tuple[WorkspaceArtifact, ...]:
    """Build every supported artifact for the complete calculated dataset."""
    artifacts = _artifacts(record, form)
    if _has_hmm_result(record.success.results):
        return (*artifacts, *_hmm_preset_artifact(form))
    return artifacts


def selected_artifacts(
    record: WorkbenchCalculationRecord,
    form: WorkbenchForm,
    selected_ids: tuple[str, ...],
) -> tuple[WorkspaceArtifact, ...]:
    """Build existing artifacts for selected records in source order."""
    if not selected_ids:
        return ()
    projected = project_record(
        record,
        WorkspaceSelection(SequenceScope.MULTIPLE, selected_ids),
    )
    return _artifacts(projected, form)


def _artifacts(
    record: WorkbenchCalculationRecord,
    form: WorkbenchForm,
) -> tuple[WorkspaceArtifact, ...]:
    artifacts: list[WorkspaceArtifact] = []
    for result in record.success.results:
        artifacts.extend(_result_artifacts(result, record, form))
    return tuple(artifacts)


def _result_artifacts(
    result: WorkbenchResult,
    record: WorkbenchCalculationRecord,
    form: WorkbenchForm,
) -> tuple[WorkspaceArtifact, ...]:
    match result:
        case MarkovBatchAnalysis() as markov:
            return (
                WorkspaceArtifact(
                    "Markov model JSON",
                    markov_model_json(markov),
                    None,
                    "markov-model.json",
                    "application/json",
                ),
                WorkspaceArtifact(
                    "Markov prefix CSV",
                    markov_sequence_csv(markov),
                    None,
                    "markov-prefix.csv",
                    "text/csv; charset=utf-8",
                ),
                WorkspaceArtifact(
                    "Markov batch-summary CSV",
                    markov_batch_summary_csv(markov),
                    None,
                    "markov-batch-summary.csv",
                    "text/csv; charset=utf-8",
                ),
            )
        case VMMAnalysis() as vmm:
            return tuple(
                WorkspaceArtifact(
                    item.name,
                    item.data,
                    None,
                    item.file_name,
                    item.mime,
                )
                for item in vmm_download_artifacts(vmm, record.success.dataset)
            )
        case HMMBatchAnalysis(records=records):
            model = form.hmm_model.to_model()
            return tuple(
                artifact
                for item in records
                for artifact in _hmm_artifacts(item, model, form.preset_name)
            )
        case ShannonBatchAnalysis():
            return ()
    assert_never(result)


def _hmm_artifacts(
    item: HMMRecordAnalysis,
    model: BinaryHMM,
    preset_name: str,
) -> tuple[WorkspaceArtifact, ...]:
    identifier = str(item.sequence_id)
    target_index = (
        None
        if item.target_assessment is None
        else item.target_assessment.actual_target_index
    )
    metadata = CandidateMetadata(identifier, preset_name, target_index)
    return (
        WorkspaceArtifact(
            f"HMM prefix CSV — {identifier}",
            prefix_csv(item.analysis, model),
            identifier,
            f"hmm-{identifier}-prefix.csv",
            "text/csv; charset=utf-8",
        ),
        WorkspaceArtifact(
            f"HMM candidate-summary CSV — {identifier}",
            candidate_summary_csv(item.analysis, model, metadata),
            identifier,
            f"hmm-{identifier}-candidate-summary.csv",
            "text/csv; charset=utf-8",
        ),
    )


def _hmm_preset_artifact(form: WorkbenchForm) -> tuple[WorkspaceArtifact, ...]:
    legacy = CalculatorForm(
        model=form.hmm_model,
        sequence_text=form.intake.text,
        actual_target=form.intake.actual_target,
        sequence_id=form.intake.sequence_id,
        preset_name=form.preset_name,
    )
    match export_preset(legacy):
        case PresetExportSuccess(payload=payload):
            return (
                WorkspaceArtifact(
                    "HMM preset JSON",
                    payload,
                    None,
                    "binary-hmm-preset.json",
                    "application/json",
                ),
            )
        case PresetExportFailure():
            return ()


def _has_hmm_result(results: tuple[WorkbenchResult, ...]) -> bool:
    for result in results:
        match result:
            case HMMBatchAnalysis():
                return True
            case MarkovBatchAnalysis() | ShannonBatchAnalysis() | VMMAnalysis():
                continue
        assert_never(result)
    return False
