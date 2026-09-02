"""Immutable state and calculation boundary for the Streamlit workbench."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

from binary_entropy.batch_parsing import (
    CsvBatchColumns,
    parse_csv_batch,
    parse_manual_batch,
    parse_txt_batch,
)
from binary_entropy.domain import BinaryLabels
from binary_entropy.errors import (
    BatchParseError,
    BatchParseErrorCode,
    BinaryEntropyError,
)
from binary_entropy.markov_types import MarkovPredictionMode, MarkovResultScope
from binary_entropy.parsing import parse_sequence
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.ui.markov_state import ESTIMATION_OPTIONS as _ESTIMATION_OPTIONS
from binary_entropy.ui.markov_state import MarkovControls as _MarkovControls
from binary_entropy.ui.markov_state import (
    MarkovEstimationChoice as _MarkovEstimationChoice,
)
from binary_entropy.ui.markov_state import MarkovWorkflow as _MarkovWorkflow
from binary_entropy.ui.markov_state import VMMSmoothingChoice as _VMMSmoothingChoice
from binary_entropy.ui.state import (
    ActualTargetChoice,
    ModelForm,
    actual_target_index,
    default_form,
)
from binary_entropy.vmm_types import VMMConfig
from binary_entropy.workbench import (
    HMMAnalysisRequest,
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
    VMMAnalysisRequest,
    WorkbenchRequest,
    WorkbenchResult,
    analyze_dataset,
)

ESTIMATION_OPTIONS: Final = _ESTIMATION_OPTIONS
MarkovControls = _MarkovControls
MarkovEstimationChoice = _MarkovEstimationChoice
MarkovWorkflow = _MarkovWorkflow
VMMSmoothingChoice = _VMMSmoothingChoice


class MethodChoice(StrEnum):
    """Methods available in the workbench selection control."""

    MARKOV = "Markov Chain"
    HMM = "Hidden Markov Model"
    SHANNON = "Observed Shannon Entropy"


class InputMode(StrEnum):
    """Supported shared sequence intake boundaries."""

    SINGLE = "Single sequence"
    BATCH = "Batch paste"
    TXT = "TXT upload"
    CSV = "CSV upload"


METHOD_OPTIONS: Final = tuple(MethodChoice)
INPUT_MODE_OPTIONS: Final = tuple(InputMode)


@dataclass(frozen=True, slots=True)
class IntakeForm:
    """Shared labels, record boundary, payload, and evaluation target."""

    observable_labels: tuple[str, str]
    mode: InputMode
    text: str
    upload_payload: bytes | None
    csv_columns: CsvBatchColumns | None
    actual_target: ActualTargetChoice
    sequence_id: str


@dataclass(frozen=True, slots=True)
class WorkbenchForm:
    """One complete editable workbench configuration."""

    methods: tuple[MethodChoice, ...]
    intake: IntakeForm
    markov: MarkovControls
    hmm_model: ModelForm
    preset_name: str

    def method_fingerprint(self, method: MethodChoice) -> str:
        """Identify only the shared and method-specific inputs affecting a result."""
        shared = self.intake
        match method:
            case MethodChoice.MARKOV:
                specific: MarkovControls | ModelForm | tuple[ModelForm, str] | None = (
                    self.markov
                )
            case MethodChoice.HMM:
                specific = (self.hmm_model, self.preset_name)
            case MethodChoice.SHANNON:
                specific = None
        return repr((shared, specific))

    def fingerprint(self) -> str:
        """Identify the complete submitted form for global parse failures."""
        return repr(self)


@dataclass(frozen=True, slots=True)
class MethodCalculationFailure:
    """One selected method that could not produce a scientific result."""

    method: MethodChoice
    message: str


@dataclass(frozen=True, slots=True)
class WorkbenchCalculationSuccess:
    """Atomic parsed dataset plus available selected-method results."""

    dataset: SequenceDataset
    results: tuple[WorkbenchResult, ...]
    failures: tuple[MethodCalculationFailure, ...]
    fingerprints: tuple[tuple[MethodChoice, str], ...]


@dataclass(frozen=True, slots=True)
class WorkbenchCalculationFailure:
    """Shared input failure that prevents every selected method."""

    message: str
    fingerprint: str


type WorkbenchCalculationOutcome = (
    WorkbenchCalculationSuccess | WorkbenchCalculationFailure
)


def default_workbench_form() -> WorkbenchForm:
    """Return the Markov-first hand-verified starter configuration."""
    legacy = default_form()
    return WorkbenchForm(
        methods=(MethodChoice.MARKOV,),
        intake=IntakeForm(
            observable_labels=legacy.model.observable_labels,
            mode=InputMode.SINGLE,
            text=legacy.sequence_text,
            upload_payload=None,
            csv_columns=None,
            actual_target=legacy.actual_target,
            sequence_id=legacy.sequence_id,
        ),
        markov=MarkovControls(
            estimation=MarkovEstimationChoice.MAXIMUM_LIKELIHOOD,
            custom_alpha=0.5,
            prediction_mode=MarkovPredictionMode.FIXED_MODEL,
            result_scope=MarkovResultScope.POOLED,
        ),
        hmm_model=legacy.model,
        preset_name=legacy.preset_name,
    )


def parse_workbench_dataset(form: WorkbenchForm) -> SequenceDataset:
    """Parse one complete shared intake without crossing record boundaries."""
    labels = BinaryLabels(
        states=("State 1", "State 2"),
        observables=form.intake.observable_labels,
    )
    intake = form.intake
    match intake.mode:
        case InputMode.SINGLE:
            dataset = SequenceDataset(
                labels,
                (
                    SequenceRecord(
                        intake.sequence_id, parse_sequence(intake.text, labels)
                    ),
                ),
            )
        case InputMode.BATCH:
            dataset = parse_manual_batch(intake.text, labels)
        case InputMode.TXT:
            payload = _required_upload(intake.upload_payload, intake.mode)
            dataset = parse_txt_batch(payload, labels)
        case InputMode.CSV:
            payload = _required_upload(intake.upload_payload, intake.mode)
            columns = intake.csv_columns
            if columns is None:
                raise BatchParseError(
                    code=BatchParseErrorCode.INVALID_COLUMNS,
                    detail="select the CSV ID and sequence columns",
                )
            dataset = parse_csv_batch(payload, labels, columns)
    if intake.mode is InputMode.CSV:
        return dataset
    target = actual_target_index(intake.actual_target)
    if target is None:
        return dataset
    records = tuple(
        SequenceRecord(record.sequence_id, record.sequence, target)
        for record in dataset.records
    )
    return SequenceDataset(dataset.labels, records)


def calculate_workbench(form: WorkbenchForm) -> WorkbenchCalculationOutcome:
    """Parse shared input once and independently run every selected method."""
    try:
        dataset = parse_workbench_dataset(form)
    except BinaryEntropyError as error:
        return WorkbenchCalculationFailure(str(error), form.fingerprint())
    results: list[WorkbenchResult] = []
    failures: list[MethodCalculationFailure] = []
    for method in form.methods:
        try:
            result = _calculate_method(form, dataset, method)
        except BinaryEntropyError as error:
            failures.append(MethodCalculationFailure(method, str(error)))
        else:
            results.append(result)
    fingerprints = tuple(
        (method, form.method_fingerprint(method)) for method in form.methods
    )
    return WorkbenchCalculationSuccess(
        dataset=dataset,
        results=tuple(results),
        failures=tuple(failures),
        fingerprints=fingerprints,
    )


def _calculate_method(
    form: WorkbenchForm,
    dataset: SequenceDataset,
    method: MethodChoice,
) -> WorkbenchResult:
    request: WorkbenchRequest
    match method:
        case MethodChoice.MARKOV:
            match form.markov.workflow:
                case MarkovWorkflow.VMM:
                    request = VMMAnalysisRequest(
                        config=VMMConfig(
                            smoothing=form.markov.vmm_smoothing(),
                            minimum_support=form.markov.minimum_support,
                        ),
                        result_scope=form.markov.vmm_result_scope(),
                    )
                case MarkovWorkflow.FIRST_ORDER:
                    request = MarkovAnalysisRequest(
                        smoothing_alpha=form.markov.smoothing_alpha(),
                        prediction_mode=form.markov.prediction_mode,
                        result_scope=form.markov.result_scope,
                    )
        case MethodChoice.HMM:
            model_form = replace(
                form.hmm_model,
                observable_labels=form.intake.observable_labels,
            )
            request = HMMAnalysisRequest(model_form.to_model())
        case MethodChoice.SHANNON:
            request = ShannonAnalysisRequest()
    return analyze_dataset(dataset, request)


def _required_upload(payload: bytes | None, mode: InputMode) -> bytes:
    if payload is None:
        raise BatchParseError(
            code=BatchParseErrorCode.MALFORMED_ROW,
            detail=f"select a {mode.value} file before calculation",
        )
    return payload
