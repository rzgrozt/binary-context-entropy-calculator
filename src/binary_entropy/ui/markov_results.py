"""Unrounded native dataframe adapters for Markov results."""

import pandas as pd

from binary_entropy.domain import float_values
from binary_entropy.markov_types import (
    MarkovBatchAnalysis,
    MarkovModel,
    MarkovRecordAnalysis,
)

type MarkovCell = str | int | float | None
type MarkovRow = tuple[MarkovCell, ...]

TIE_TOLERANCE = 1e-12


def markov_prediction_label(
    probability_a: float | None,
    probability_b: float | None,
    labels: tuple[str, str],
) -> str:
    """Return a scientifically meaningful user-facing prediction label."""
    if probability_a is None or probability_b is None:
        return "Insufficient evidence"

    if abs(probability_a - probability_b) < TIE_TOLERANCE:
        return "Tie"

    return labels[0] if probability_a > probability_b else labels[1]

def markov_transition_dataframe(model: MarkovModel) -> pd.DataFrame:
    """Expose counts and raw fitted transition probabilities by current state."""
    label_a, label_b = model.observable_labels
    rows: list[MarkovRow] = []
    for index, label in enumerate((label_a, label_b)):
        matrix_row = model.transition_matrix[index]
        probability_a, probability_b = (
            (None, None) if matrix_row is None else float_values(matrix_row)
        )
        row_sum = (
            None
            if probability_a is None or probability_b is None
            else probability_a + probability_b
        )
        rows.append(
            (
                label,
                *model.transition_counts[index],
                probability_a,
                probability_b,
                row_sum,
            )
        )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Current state",
            f"Count next {label_a}",
            f"Count next {label_b}",
            f"P(next {label_a})",
            f"P(next {label_b})",
            "Row sum",
        ),
    )


def markov_record_dataframe(analysis: MarkovBatchAnalysis) -> pd.DataFrame:
    """Build one unrounded final-prediction row per independent sequence."""
    labels = analysis.model.observable_labels
    rows = tuple(_record_row(record, labels) for record in analysis.records)
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Sequence ID",
            "Length",
            "Current context",
            "Observed target",
            f"P(next {labels[0]})",
            f"P(next {labels[1]})",
            "Prediction",
            "Predictive entropy (bits)",
            "Target probability",
            "Target surprisal (bits)",
        ),
    )


def markov_prefix_dataframe(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> pd.DataFrame:
    """Build unrounded prefix rows for one Markov sequence."""
    rows: list[MarkovRow] = []
    for prefix in record.rows:
        probabilities = (
            (None, None)
            if prefix.predictive is None
            else float_values(prefix.predictive)
        )
        rows.append(
            (
                prefix.depth,
                None if prefix.context is None else labels[prefix.context[0]],
                prefix.fitted_transition_count,
                probabilities[0],
                probabilities[1],
                prefix.entropy_bits,
            )
        )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Prefix depth",
            "Current context",
            "Fitted transitions",
            f"P(next {labels[0]})",
            f"P(next {labels[1]})",
            "Predictive entropy (bits)",
        ),
    )


def _record_row(
    record: MarkovRecordAnalysis,
    labels: tuple[str, str],
) -> MarkovRow:
    final = record.rows[-1]
    probabilities = (
        (None, None) if final.predictive is None else float_values(final.predictive)
    )
    target = record.target_assessment
    return (
        record.sequence_id,
        record.sequence_length,
        None if final.context is None else labels[final.context[0]],
        None
        if record.actual_target_index is None
        else labels[record.actual_target_index],
        probabilities[0],
        probabilities[1],
        markov_prediction_label(
            probabilities[0],
            probabilities[1],
            labels,
        ),
        final.entropy_bits,
        None if target is None else target.probability,
        None if target is None else target.surprisal_bits,
    )
