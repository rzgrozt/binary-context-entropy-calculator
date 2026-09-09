"""Shared public-VMM adapter for independently analyzed stimuli."""

from collections.abc import Sequence

from binary_entropy.domain import BinaryLabels
from binary_entropy.records import SequenceDataset, SequenceRecord
from binary_entropy.stimulus_metrics import sequence_metrics
from binary_entropy.stimulus_search_results import StimulusCandidate
from binary_entropy.stimulus_search_types import StimulusId, StimulusSearchConfig
from binary_entropy.vmm_types import VMMResultScope
from binary_entropy.workbench import VMMAnalysisRequest, analyze_dataset


def analyze_stimulus_records(
    records: Sequence[SequenceRecord],
    config: StimulusSearchConfig,
) -> tuple[StimulusCandidate, ...]:
    """Analyze each supplied record under an independent VMM model."""
    dataset = SequenceDataset(
        BinaryLabels(
            states=("internal-state-0", "internal-state-1"),
            observables=("A", "B"),
        ),
        records,
    )
    analysis = analyze_dataset(
        dataset,
        VMMAnalysisRequest(
            config=config.vmm_config,
            result_scope=VMMResultScope.PER_SEQUENCE,
        ),
    )
    return tuple(
        StimulusCandidate(
            stimulus_id=StimulusId(str(row.sequence_id)),
            sequence=row.sequence,
            metrics=sequence_metrics(row.sequence),
            probability_a=row.probability_a,
            probability_b=row.probability_b,
            predicted_target_index=row.predicted_target_index,
            predictive_entropy_bits=row.predictive_entropy_bits,
            effective_context_depth=row.effective_context_depth,
            context_used=row.context_used,
            support_count=row.support_count,
            surprisal_a_bits=row.surprisal_a_bits,
            surprisal_b_bits=row.surprisal_b_bits,
            depth_rows=row.depth_rows,
            symbol_mapping=config.symbol_mapping,
        )
        for row in analysis.records
    )
