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
    "CsvBatchColumns",
    "HMMAnalysisRequest",
    "HMMBatchAnalysis",
    "InvalidVMMConfigurationError",
    "KTSmoothing",
    "MLESmoothing",
    "MarkovAnalysisRequest",
    "MarkovBatchAnalysis",
    "MarkovEstimation",
    "MarkovModel",
    "MarkovPredictionMode",
    "MarkovResultScope",
    "MethodComparison",
    "SequenceAnalysis",
    "SequenceDataset",
    "SequenceId",
    "SequenceRecord",
    "ShannonAnalysisRequest",
    "ShannonBatchAnalysis",
    "ShannonPrefixResult",
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
    "analyze_dataset",
    "analyze_hmm",
    "analyze_markov",
    "analyze_markov_per_sequence",
    "analyze_sequence",
    "analyze_shannon",
    "analyze_vmm",
    "analyze_vmm_per_sequence",
    "compare_methods",
    "fit_markov",
    "fit_vmm",
    "markov_batch_summary_csv",
    "markov_model_json",
    "markov_sequence_csv",
    "parse_csv_batch",
    "parse_manual_batch",
    "parse_sequence",
    "parse_txt_batch",
    "predict_markov",
    "vmm_context_evidence_csv",
    "vmm_context_model_json",
    "vmm_evaluation_csv",
]
