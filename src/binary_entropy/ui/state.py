"""Typed form state and boundary operations for the Streamlit application."""

from dataclasses import dataclass, replace
from enum import StrEnum

from pydantic import ValidationError

from binary_entropy.analysis import analyze_sequence, assess_target
from binary_entropy.domain import (
    BinaryHMM,
    BinaryLabels,
    LabelPair,
    ObservableIndex,
    SequenceAnalysis,
    TargetAssessment,
    float_values,
)
from binary_entropy.errors import BinaryEntropyError
from binary_entropy.parsing import parse_sequence
from binary_entropy.serialization import (
    model_from_preset,
    parse_preset_json,
    preset_from_model,
    preset_json,
)

type ProbabilityPair = tuple[float, float]
type ProbabilityMatrix = tuple[ProbabilityPair, ProbabilityPair]


class ActualTargetChoice(StrEnum):
    """Optional final target selected by the user."""

    NONE = "none"
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True, slots=True)
class ModelForm:
    """Editable representation of one two-state binary HMM."""

    state_labels: LabelPair
    observable_labels: LabelPair
    initial: ProbabilityPair
    transition: ProbabilityMatrix
    emission: ProbabilityMatrix

    def to_model(self) -> BinaryHMM:
        """Parse form values into a validated immutable model."""
        return BinaryHMM(
            labels=BinaryLabels(
                states=self.state_labels,
                observables=self.observable_labels,
            ),
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
        )

    @classmethod
    def from_model(cls, model: BinaryHMM) -> "ModelForm":
        """Copy a validated model into editable scalar values."""
        initial_0, initial_1 = float_values(model.initial)
        transition_00, transition_01, transition_10, transition_11 = float_values(
            model.transition
        )
        emission_00, emission_01, emission_10, emission_11 = float_values(
            model.emission
        )
        return cls(
            state_labels=model.labels.states,
            observable_labels=model.labels.observables,
            initial=(initial_0, initial_1),
            transition=(
                (transition_00, transition_01),
                (transition_10, transition_11),
            ),
            emission=((emission_00, emission_01), (emission_10, emission_11)),
        )


@dataclass(frozen=True, slots=True)
class CalculatorForm:
    """All user-editable values that determine results or exports."""

    model: ModelForm
    sequence_text: str
    actual_target: ActualTargetChoice
    sequence_id: str
    preset_name: str

    def fingerprint(self) -> str:
        """Return a deterministic identity for stale-result detection."""
        return repr(
            (
                self.model,
                self.sequence_text,
                self.actual_target,
                self.sequence_id,
                self.preset_name,
            )
        )


@dataclass(frozen=True, slots=True)
class CalculationSuccess:
    """Validated model, complete analysis, and optional target assessment."""

    model: BinaryHMM
    analysis: SequenceAnalysis
    target_assessment: TargetAssessment | None


@dataclass(frozen=True, slots=True)
class CalculationFailure:
    """User-correctable calculation failure."""

    message: str


type CalculationOutcome = CalculationSuccess | CalculationFailure


@dataclass(frozen=True, slots=True)
class PresetImportSuccess:
    """Transactionally imported model form."""

    form: CalculatorForm


@dataclass(frozen=True, slots=True)
class PresetImportFailure:
    """Preset error that leaves the current form unchanged."""

    message: str


type PresetImportOutcome = PresetImportSuccess | PresetImportFailure


@dataclass(frozen=True, slots=True)
class PresetExportSuccess:
    """Validated JSON preset bytes."""

    payload: bytes


@dataclass(frozen=True, slots=True)
class PresetExportFailure:
    """Reason the current model cannot be exported."""

    message: str


type PresetExportOutcome = PresetExportSuccess | PresetExportFailure


def default_form() -> CalculatorForm:
    """Return the labeled hand-verified demonstration model and sequence."""
    return CalculatorForm(
        model=ModelForm(
            state_labels=("State 1", "State 2"),
            observable_labels=("A", "B"),
            initial=(0.6, 0.4),
            transition=((0.7, 0.3), (0.2, 0.8)),
            emission=((0.9, 0.1), (0.2, 0.8)),
        ),
        sequence_text="A, B, B, A, A, A, B",
        actual_target=ActualTargetChoice.NONE,
        sequence_id="sequence-001",
        preset_name="hand-calculated model",
    )


def calculate_form(form: CalculatorForm) -> CalculationOutcome:
    """Validate and calculate one form only when explicitly requested."""
    try:
        model = form.model.to_model()
        sequence = parse_sequence(form.sequence_text, model.labels)
        analysis = analyze_sequence(model, sequence)
    except BinaryEntropyError as error:
        return CalculationFailure(message=str(error))
    target_index = actual_target_index(form.actual_target)
    target_assessment = (
        assess_target(analysis.rows[-1].predictive, target_index)
        if target_index is not None
        else None
    )
    return CalculationSuccess(
        model=model,
        analysis=analysis,
        target_assessment=target_assessment,
    )


def import_preset(
    payload: str | bytes,
    current: CalculatorForm,
) -> PresetImportOutcome:
    """Validate a preset before replacing any current model input."""
    try:
        preset = parse_preset_json(payload)
        imported_model = ModelForm.from_model(model_from_preset(preset))
    except BinaryEntropyError as error:
        return PresetImportFailure(message=str(error))
    return PresetImportSuccess(
        form=replace(
            current,
            model=imported_model,
            preset_name=preset.preset_name,
        )
    )


def export_preset(form: CalculatorForm) -> PresetExportOutcome:
    """Validate and serialize the current model without its sequence."""
    try:
        model = form.model.to_model()
        payload = preset_json(preset_from_model(model, form.preset_name))
    except (BinaryEntropyError, ValidationError) as error:
        return PresetExportFailure(message=str(error))
    return PresetExportSuccess(payload=payload)


def actual_target_index(choice: ActualTargetChoice) -> ObservableIndex | None:
    """Map the UI target choice to the core observable index."""
    match choice:  # noqa: RUF100  # noqa: MATCH_OK
        case ActualTargetChoice.NONE:
            return None
        case ActualTargetChoice.FIRST:
            return 0
        case ActualTargetChoice.SECOND:
            return 1
