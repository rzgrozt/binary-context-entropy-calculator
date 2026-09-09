"""Reusable core for binary hidden-Markov predictive entropy analysis."""

from binary_entropy.analysis import analyze_sequence
from binary_entropy.batch_parsing import (
    CsvBatchColumns,
    parse_csv_batch,
    parse_manual_batch,
    parse_txt_batch,
)
from binary_entropy.domain import BinaryHMM, BinaryLabels, SequenceAnalysis
from binary_entropy.markov_batch_serialization import markov_batch_summary_csv
from binary_entropy.markov_serialization import (
    markov_model_json,
    markov_sequence_csv,
)
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovEstimation,
    MarkovModel,
    MarkovPredictionMode,
    MarkovResultScope,
)
from binary_entropy.methods.hmm import HMMBatchAnalysis, analyze_hmm
from binary_entropy.methods.markov import (
    analyze_markov,
    analyze_markov_per_sequence,
    fit_markov,
    predict_markov,
)
from binary_entropy.methods.shannon import (
    ShannonBatchAnalysis,
    ShannonPrefixResult,
    analyze_shannon,
)
from binary_entropy.methods.vmm import analyze_vmm, analyze_vmm_per_sequence, fit_vmm
from binary_entropy.parsing import parse_sequence
from binary_entropy.records import (
    BinarySequence,
    SequenceDataset,
    SequenceId,
    SequenceRecord,
)
from binary_entropy.stimulus_generation import generate_candidate_records
from binary_entropy.stimulus_matching import (
    create_complement_candidates,
    match_stimuli,
)
from binary_entropy.stimulus_metrics import sequence_metrics
from binary_entropy.stimulus_search import evaluate_candidate, search_stimuli
from binary_entropy.stimulus_search_csv import (
    stimulus_candidate_csv,
    stimulus_experiment_csv,
    stimulus_scientific_csv,
)
from binary_entropy.stimulus_search_json import stimulus_generator_config_json
from binary_entropy.stimulus_search_results import (
    ConstraintViolation,
    MatchingResult,
    MatchTolerances,
    MetricSummary,
    SearchPartialReason,
    SearchResult,
    SearchStatus,
    SequenceMetrics,
    StimulusCandidate,
    StimulusPair,
    TargetCongruency,
    ValidationReport,
)
from binary_entropy.stimulus_search_types import (
    InclusiveRange,
    InvalidStimulusSearchConfigurationError,
    PredictedSymbol,
    PreferenceMetric,
    SoftPreference,
    StimulusConstraints,
    StimulusId,
    StimulusSearchConfig,
)
from binary_entropy.stimulus_targets import assign_targets, validate_stimuli
from binary_entropy.vmm_serialization import (
    vmm_context_evidence_csv,
    vmm_context_model_json,
    vmm_evaluation_csv,
)
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    InvalidVMMConfigurationError,
    KTSmoothing,
    MLESmoothing,
    VMMAnalysis,
    VMMConfig,
    VMMContextCount,
    VMMDepthAnalysis,
    VMMDepthStatus,
    VMMModel,
    VMMRecordAnalysis,
    VMMResultScope,
    VMMSmoothing,
)
from binary_entropy.workbench import (
    AnalysisMethod,
    HMMAnalysisRequest,
    MarkovAnalysisRequest,
    MethodComparison,
    ShannonAnalysisRequest,
    VMMAnalysisRequest,
    analyze_dataset,
    compare_methods,
)

__all__ = [
    "AdditiveSmoothing",
    "AnalysisMethod",
    "BinaryHMM",
    "BinaryLabels",
    "BinarySequence",
    "ConstraintViolation",
    "CsvBatchColumns",
    "HMMAnalysisRequest",
    "HMMBatchAnalysis",
    "InclusiveRange",
    "InvalidStimulusSearchConfigurationError",
    "InvalidVMMConfigurationError",
    "KTSmoothing",
    "MLESmoothing",
    "MarkovAnalysisRequest",
    "MarkovBatchAnalysis",
    "MarkovEstimation",
    "MarkovModel",
    "MarkovPredictionMode",
    "MarkovResultScope",
    "MatchTolerances",
    "MatchingResult",
    "MethodComparison",
    "MetricSummary",
    "PredictedSymbol",
    "PreferenceMetric",
    "SearchPartialReason",
    "SearchResult",
    "SearchStatus",
    "SequenceAnalysis",
    "SequenceDataset",
    "SequenceId",
    "SequenceMetrics",
    "SequenceRecord",
    "ShannonAnalysisRequest",
    "ShannonBatchAnalysis",
    "ShannonPrefixResult",
    "SoftPreference",
    "StimulusCandidate",
    "StimulusConstraints",
    "StimulusId",
    "StimulusPair",
    "StimulusSearchConfig",
    "TargetCongruency",
    "VMMAnalysis",
    "VMMAnalysisRequest",
    "VMMConfig",
    "VMMContextCount",
    "VMMDepthAnalysis",
    "VMMDepthStatus",
    "VMMModel",
    "VMMRecordAnalysis",
    "VMMResultScope",
    "VMMSmoothing",
    "ValidationReport",
    "analyze_dataset",
    "analyze_hmm",
    "analyze_markov",
    "analyze_markov_per_sequence",
    "analyze_sequence",
    "analyze_shannon",
    "analyze_vmm",
    "analyze_vmm_per_sequence",
    "assign_targets",
    "assign_outcomes",
    "TargetAssignmentMode",
    "PresentationMetadata",
    "apply_presentation_metadata",
    "compare_methods",
    "create_complement_candidates",
    "evaluate_candidate",
    "fit_markov",
    "fit_vmm",
    "generate_candidate_records",
    "markov_batch_summary_csv",
    "markov_model_json",
    "markov_sequence_csv",
    "match_stimuli",
    "parse_csv_batch",
    "parse_manual_batch",
    "parse_sequence",
    "parse_txt_batch",
    "predict_markov",
    "search_stimuli",
    "sequence_metrics",
    "stimulus_candidate_csv",
    "stimulus_experiment_csv",
    "stimulus_generator_config_json",
    "stimulus_scientific_csv",
    "validate_stimuli",
    "vmm_context_evidence_csv",
    "vmm_context_model_json",
    "vmm_evaluation_csv",
]

from binary_entropy.stimulus_metadata import (
    PresentationMetadata,
    apply_presentation_metadata,
)
from binary_entropy.stimulus_targets import TargetAssignmentMode, assign_outcomes
