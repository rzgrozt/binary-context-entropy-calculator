"""Cross-method comparison with explicit non-predictive Shannon cells."""

import pandas as pd
import streamlit as st

from binary_entropy.domain import float_values
from binary_entropy.markov_types import MarkovBatchAnalysis, MarkovResultScope
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import format_ui_decimal
from binary_entropy.ui.vmm_results import vmm_prediction_label
from binary_entropy.vmm_types import VMMAnalysis, VMMResultScope
from binary_entropy.workbench import WorkbenchResult

type ComparisonRow = tuple[str, ...]


def comparison_dataframe(
    results: tuple[WorkbenchResult, ...],
    labels: tuple[str, str],
) -> pd.DataFrame:
    """Compare predictive methods while marking Shannon fields not applicable."""
    record_count = len(results[0].records) if results else 0
    rows = tuple(
        _comparison_row(result, labels, record_index)
        for record_index in range(record_count)
        for result in results
    )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Sequence ID",
            "Method",
            "Scope",
            f"P(next {labels[0]})",
            f"P(next {labels[1]})",
            "Prediction",
            "Target probability",
            "Target surprisal (bits)",
            "Predictive entropy (bits)",
            "Observed entropy (bits)",
        ),
    )


def render_comparison(
    results: tuple[WorkbenchResult, ...],
    labels: tuple[str, str],
) -> None:
    """Render the cross-method table with explicit units and applicability."""
    _ = st.subheader("Selected-method comparison")
    _ = st.caption(
        joined_text(
            (
                "Predictive uncertainty and observed-symbol entropy answer ",
                "different questions and are not interchangeable.",
            )
        )
    )
    _ = st.dataframe(
        comparison_dataframe(results, labels),
        hide_index=True,
        width="stretch",
        height="content",
    )


def _comparison_row(
    result: WorkbenchResult,
    labels: tuple[str, str],
    record_index: int,
) -> ComparisonRow:
    match result:
        case VMMAnalysis(records=records, result_scope=result_scope):
            record = records[record_index]
            target = record.target_assessment
            match result_scope:
                case VMMResultScope.POOLED:
                    scope = "Pooled fit; per-sequence prediction"
                case VMMResultScope.PER_SEQUENCE:
                    scope = "Per-sequence fit and prediction"
            return (
                record.sequence_id,
                "Variable-order Markov",
                scope,
                _display(record.probability_a),
                _display(record.probability_b),
                vmm_prediction_label(record, labels),
                _display(None if target is None else target.probability),
                _display(None if target is None else target.surprisal_bits),
                _display(record.predictive_entropy_bits),
                "N/A",
            )
        case MarkovBatchAnalysis(records=records, result_scope=result_scope):
            record = records[record_index]
            final = record.rows[-1]
            target = record.target_assessment
            probabilities = (
                (None, None)
                if final.predictive is None
                else float_values(final.predictive)
            )
            match result_scope:
                case MarkovResultScope.POOLED:
                    scope = "Pooled fit; per-sequence prediction"
                case MarkovResultScope.PER_SEQUENCE:
                    scope = "Per-sequence fit and prediction"
            return (
                record.sequence_id,
                "Markov Chain",
                scope,
                _display(probabilities[0]),
                _display(probabilities[1]),
                (
                    "N/A"
                    if final.predicted_index is None
                    else labels[final.predicted_index]
                ),
                _display(None if target is None else target.probability),
                _display(None if target is None else target.surprisal_bits),
                _display(final.entropy_bits),
                "N/A",
            )
        case HMMBatchAnalysis(records=records):
            record = records[record_index]
            final = record.analysis.rows[-1]
            target = record.target_assessment
            probability_a, probability_b = float_values(final.predictive)
            return (
                record.sequence_id,
                "Hidden Markov Model",
                "Configured model; per-sequence filtering",
                _display(probability_a),
                _display(probability_b),
                labels[final.predicted_index],
                _display(None if target is None else target.probability),
                _display(None if target is None else target.surprisal_bits),
                _display(final.entropy_bits),
                "N/A",
            )
        case ShannonBatchAnalysis(records=records):
            record = records[record_index]
            return (
                record.sequence_id,
                "Observed Shannon Entropy",
                "Per-sequence descriptive",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                _display(record.summary.entropy_bits),
            )


def _display(value: float | None) -> str:
    return "N/A" if value is None else format_ui_decimal(value)
