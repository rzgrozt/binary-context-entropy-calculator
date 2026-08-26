"""Native dataframe adapters for descriptive Shannon results."""

import pandas as pd
import streamlit as st

from binary_entropy.domain import float_values
from binary_entropy.methods.shannon import ShannonBatchAnalysis, ShannonRecordAnalysis
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT, format_ui_decimal


def shannon_record_dataframe(analysis: ShannonBatchAnalysis) -> pd.DataFrame:
    """Build one descriptive entropy row per independent sequence."""
    labels = analysis.observable_labels
    rows = tuple(
        (
            record.sequence_id,
            record.summary.observation_count,
            *record.summary.symbol_counts,
            (
                None
                if record.summary.symbol_probabilities is None
                else record.summary.symbol_probabilities[0]
            ),
            (
                None
                if record.summary.symbol_probabilities is None
                else record.summary.symbol_probabilities[1]
            ),
            record.summary.entropy_bits,
        )
        for record in analysis.records
    )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "Sequence ID",
            "Observations",
            f"Count {labels[0]}",
            f"Count {labels[1]}",
            f"P(observed {labels[0]})",
            f"P(observed {labels[1]})",
            "Observed entropy (bits)",
        ),
    )


def shannon_prefix_dataframe(record: ShannonRecordAnalysis) -> pd.DataFrame:
    """Build the optional unrounded observed-composition prefix table."""
    return pd.DataFrame.from_records(
        (
            (
                row.depth,
                row.count_a,
                row.count_b,
                row.probability_a,
                row.probability_b,
                row.entropy_bits,
            )
            for row in record.prefixes
        ),
        columns=(
            "Prefix depth",
            "Count A",
            "Count B",
            "P(A)",
            "P(B)",
            "Observed entropy (bits)",
        ),
    )


def render_shannon_result(
    analysis: ShannonBatchAnalysis,
    *,
    has_targets: bool,
) -> None:
    """Render pooled and per-sequence descriptive entropy without predictions."""
    labels = analysis.observable_labels
    pooled = analysis.pooled
    _ = st.subheader("Observed-symbol Shannon entropy")
    _ = st.markdown(
        "This describes symbols already present and is not a next-target prediction."
    )
    columns = st.columns(3)
    _ = columns[0].metric("Observed symbols", str(pooled.observation_count))
    probabilities = (
        None
        if pooled.symbol_probabilities is None
        else float_values(pooled.symbol_probabilities)
    )
    _ = columns[1].metric(
        f"Observed P({labels[0]})",
        "Unavailable" if probabilities is None else format_ui_decimal(probabilities[0]),
    )
    _ = columns[2].metric(
        "Observed Shannon entropy (bits)",
        "Unavailable"
        if pooled.entropy_bits is None
        else format_ui_decimal(pooled.entropy_bits),
    )
    if has_targets:
        _ = st.info(
            joined_text(
                (
                    "Target assessment: Not applicable. Observed Shannon entropy ",
                    "has no predictive distribution.",
                )
            )
        )
    frame = shannon_record_dataframe(analysis)
    _ = st.markdown("Per-sequence observed-symbol summaries.")
    _ = st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height="content",
        column_config={
            f"P(observed {labels[0]})": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            f"P(observed {labels[1]})": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
            "Observed entropy (bits)": st.column_config.NumberColumn(
                format=UI_NUMBER_FORMAT
            ),
        },
    )
    for record in analysis.records:
        if not record.prefixes:
            continue
        with st.expander(f"Observed prefixes: {record.sequence_id}"):
            _ = st.dataframe(
                shannon_prefix_dataframe(record),
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    "P(A)": st.column_config.NumberColumn(format=UI_NUMBER_FORMAT),
                    "P(B)": st.column_config.NumberColumn(format=UI_NUMBER_FORMAT),
                    "Observed entropy (bits)": st.column_config.NumberColumn(
                        format=UI_NUMBER_FORMAT
                    ),
                },
            )
