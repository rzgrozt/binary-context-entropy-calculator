"""Typed multi-method routing for the non-UI scientific workbench."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from binary_entropy.domain import BinaryHMM
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovPredictionMode,
    MarkovResultScope,
)
from binary_entropy.methods.hmm import HMMBatchAnalysis, analyze_hmm
from binary_entropy.methods.markov import analyze_markov, analyze_markov_per_sequence
from binary_entropy.methods.shannon import ShannonBatchAnalysis, analyze_shannon
from binary_entropy.records import SequenceDataset


class AnalysisMethod(StrEnum):
    """Scientific methods available through the workbench router."""

    HMM = "hmm"
    MARKOV = "markov"
    OBSERVED_SHANNON = "observed_shannon"


@dataclass(frozen=True, slots=True)
class HMMAnalysisRequest:
    """Request independent legacy analysis under one fixed HMM."""

    model: BinaryHMM
    method: AnalysisMethod = field(default=AnalysisMethod.HMM, init=False)


@dataclass(frozen=True, slots=True)
class MarkovAnalysisRequest:
    """Request one explicit first-order result scope and prefix mode."""

    smoothing_alpha: float = 0.0
    prediction_mode: MarkovPredictionMode = MarkovPredictionMode.FIXED_MODEL
    result_scope: MarkovResultScope = MarkovResultScope.POOLED
    method: AnalysisMethod = field(default=AnalysisMethod.MARKOV, init=False)


@dataclass(frozen=True, slots=True)
class ShannonAnalysisRequest:
    """Request descriptive observed-symbol entropy only."""

    method: AnalysisMethod = field(
        default=AnalysisMethod.OBSERVED_SHANNON,
        init=False,
    )


type WorkbenchRequest = (
    HMMAnalysisRequest | MarkovAnalysisRequest | ShannonAnalysisRequest
)
type WorkbenchResult = HMMBatchAnalysis | MarkovBatchAnalysis | ShannonBatchAnalysis


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """Ordered labeled results for direct method comparison."""

    results: tuple[WorkbenchResult, ...]


def analyze_dataset(
    dataset: SequenceDataset,
    request: WorkbenchRequest,
) -> WorkbenchResult:
    """Route one typed request without admitting invalid parameter combinations."""
    match request:
        case HMMAnalysisRequest(model=model):
            return analyze_hmm(dataset, model)
        case MarkovAnalysisRequest(
            smoothing_alpha=smoothing_alpha,
            prediction_mode=prediction_mode,
            result_scope=result_scope,
        ):
            match result_scope:
                case MarkovResultScope.POOLED:
                    return analyze_markov(dataset, smoothing_alpha, prediction_mode)
                case MarkovResultScope.PER_SEQUENCE:
                    return analyze_markov_per_sequence(
                        dataset,
                        smoothing_alpha,
                        prediction_mode,
                    )
        case ShannonAnalysisRequest():
            return analyze_shannon(dataset)


def compare_methods(
    dataset: SequenceDataset,
    requests: Sequence[WorkbenchRequest],
) -> MethodComparison:
    """Run an ordered collection of typed method requests on one dataset."""
    return MethodComparison(
        results=tuple(analyze_dataset(dataset, request) for request in requests)
    )
