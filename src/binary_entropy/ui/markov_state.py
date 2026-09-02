"""Immutable controls shared by Markov-family workbench workflows."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from binary_entropy.markov_types import MarkovPredictionMode, MarkovResultScope
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMResultScope,
    VMMSmoothing,
)


class MarkovWorkflow(StrEnum):
    """Predictive workflows available under the Markov method family."""

    VMM = "Variable-order Markov"
    FIRST_ORDER = "First-order Markov"


class MarkovEstimationChoice(StrEnum):
    """Visible first-order Markov estimation choices."""

    MAXIMUM_LIKELIHOOD = "Maximum likelihood"
    LAPLACE = "Laplace/add-one smoothing"
    CUSTOM = "Custom additive smoothing alpha"


class VMMSmoothingChoice(StrEnum):
    """Available smoothing families for variable-order prediction."""

    KT = "Krichevsky-Trofimov (alpha = 0.500)"
    MLE = "Maximum likelihood (alpha = 0.000)"
    ADDITIVE = "Custom additive smoothing"


ESTIMATION_OPTIONS: Final = tuple(MarkovEstimationChoice)


@dataclass(frozen=True, slots=True)
class MarkovControls:
    """Immutable first-order and variable-order Markov form state."""

    estimation: MarkovEstimationChoice
    custom_alpha: float
    prediction_mode: MarkovPredictionMode
    result_scope: MarkovResultScope
    workflow: MarkovWorkflow = MarkovWorkflow.VMM
    vmm_smoothing_choice: VMMSmoothingChoice = VMMSmoothingChoice.KT
    vmm_custom_alpha: float = 0.5
    minimum_support: int = 2

    def smoothing_alpha(self) -> float:
        """Map the first-order estimator to its additive alpha."""
        match self.estimation:
            case MarkovEstimationChoice.MAXIMUM_LIKELIHOOD:
                return 0.0
            case MarkovEstimationChoice.LAPLACE:
                return 1.0
            case MarkovEstimationChoice.CUSTOM:
                return self.custom_alpha

    def vmm_smoothing(self) -> VMMSmoothing:
        """Construct the selected typed VMM smoothing value."""
        match self.vmm_smoothing_choice:
            case VMMSmoothingChoice.KT:
                return KTSmoothing()
            case VMMSmoothingChoice.MLE:
                return MLESmoothing()
            case VMMSmoothingChoice.ADDITIVE:
                return AdditiveSmoothing(self.vmm_custom_alpha)

    def vmm_result_scope(self) -> VMMResultScope:
        """Map the shared result scope to the VMM scope type."""
        match self.result_scope:
            case MarkovResultScope.POOLED:
                return VMMResultScope.POOLED
            case MarkovResultScope.PER_SEQUENCE:
                return VMMResultScope.PER_SEQUENCE
