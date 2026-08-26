"""One-row-per-record Markov experimental exports."""

from typing import Final

from binary_entropy.domain import TargetClassification, float_values
from binary_entropy.information import binary_entropy, surprisal
from binary_entropy.markov_csv import CsvCell, markov_csv_text
from binary_entropy.markov_types import MarkovBatchAnalysis, MarkovRecordAnalysis

MARKOV_BATCH_COLUMNS: Final = (
    "sequence_id",
    "sequence",
    "length",
    "count_A",
    "count_B",
    "AA_count",
    "AB_count",
    "BA_count",
    "BB_count",
    "P_A_given_A",
    "P_B_given_A",
    "P_A_given_B",
    "P_B_given_B",
    "last_symbol",
    "P_next_A",
    "P_next_B",
    "predicted_target",
    "predictive_entropy",
    "surprisal_A",
    "surprisal_B",
    "observed_shannon_entropy",
    "actual_target",
    "actual_target_probability",
    "actual_target_surprisal",
    "modal_or_nonmodal",
    "method",
    "result_scope",
    "prediction_mode",
    "markov_order",
    "estimation_method",
    "smoothing_alpha",
    "source_sequence_count",
    "source_transition_count",
)


def markov_batch_summary_csv(analysis: MarkovBatchAnalysis) -> str:
    """Serialize one deterministic experimental summary row per sequence."""
    rows = tuple(_summary_row(analysis, record) for record in analysis.records)
    return markov_csv_text(MARKOV_BATCH_COLUMNS, rows)


def _summary_row(
    analysis: MarkovBatchAnalysis,
    record: MarkovRecordAnalysis,
) -> tuple[CsvCell, ...]:
    model = record.model
    count_a = record.sequence.count(0)
    count_b = record.sequence.count(1)
    aa_count, ab_count = model.transition_counts[0]
    ba_count, bb_count = model.transition_counts[1]
    row_a, row_b = model.transition_matrix
    if row_a is None:
        probability_a_given_a = None
        probability_b_given_a = None
    else:
        probability_a_given_a, probability_b_given_a = float_values(row_a)
    if row_b is None:
        probability_a_given_b = None
        probability_b_given_b = None
    else:
        probability_a_given_b, probability_b_given_b = float_values(row_b)

    final = record.rows[-1]
    if final.predictive is None:
        probability_next_a = None
        probability_next_b = None
        surprisal_a = None
        surprisal_b = None
    else:
        probability_next_a, probability_next_b = float_values(final.predictive)
        surprisal_a = surprisal(probability_next_a)
        surprisal_b = surprisal(probability_next_b)

    target = record.target_assessment
    if target is None:
        modal_or_nonmodal = None
    else:
        match target.classification:
            case TargetClassification.MODAL:
                modal_or_nonmodal = "modal"
            case TargetClassification.LOWER_PROBABILITY:
                modal_or_nonmodal = "nonmodal"
            case TargetClassification.TIED:
                modal_or_nonmodal = "tied"

    sequence_length = record.sequence_length
    observed_entropy = (
        None if sequence_length == 0 else binary_entropy(count_a / sequence_length)
    )
    return (
        record.sequence_id,
        ",".join(model.observable_labels[index] for index in record.sequence),
        sequence_length,
        count_a,
        count_b,
        aa_count,
        ab_count,
        ba_count,
        bb_count,
        probability_a_given_a,
        probability_b_given_a,
        probability_a_given_b,
        probability_b_given_b,
        None if not record.sequence else model.observable_labels[record.sequence[-1]],
        probability_next_a,
        probability_next_b,
        (
            None
            if final.predicted_index is None
            else model.observable_labels[final.predicted_index]
        ),
        final.entropy_bits,
        surprisal_a,
        surprisal_b,
        observed_entropy,
        (
            None
            if record.actual_target_index is None
            else model.observable_labels[record.actual_target_index]
        ),
        None if target is None else target.probability,
        None if target is None else target.surprisal_bits,
        modal_or_nonmodal,
        analysis.method,
        analysis.result_scope.value,
        analysis.prediction_mode.value,
        model.markov_order,
        model.estimation_method.value,
        model.smoothing_alpha,
        model.source_sequence_count,
        model.source_transition_count,
    )
