"""Canonical numeric and tabular presentation."""

import math
from dataclasses import dataclass

import pandas as pd  # noqa: PANDAS_OK

from binary_entropy.constants import DISPLAY_DECIMALS
from binary_entropy.domain import BinaryHMM, SequenceAnalysis, float_values

type CellValue = str | int | float | None
type RowValues = tuple[CellValue, ...]


@dataclass(frozen=True, slots=True)
class AnalysisTable:
    """Stable columns and typed row values for one prefix analysis."""

    columns: tuple[str, ...]
    rows: tuple[RowValues, ...]


def format_decimal(value: float) -> str:
    """Format a scientific value with the canonical decimal precision."""
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{DISPLAY_DECIMALS}f}"


def analysis_dataframe(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
) -> pd.DataFrame:
    """Build a stable depth-sorted prefix result dataframe."""
    table = analysis_table(analysis, model)
    return pd.DataFrame.from_records(table.rows, columns=table.columns)


def analysis_table(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
) -> AnalysisTable:
    """Build stable typed columns and rows without pandas internals."""
    state_0, state_1 = model.labels.states
    observable_0, observable_1 = model.labels.observables
    columns = (
        "depth",
        "observed_symbol",
        "next_target_symbol",
        f"predictive_probability_{observable_0}",
        f"predictive_probability_{observable_1}",
        "predictive_entropy_bits",
        "predicted_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "target_classification",
        f"posterior_{state_0}",
        f"posterior_{state_1}",
        f"next_hidden_{state_0}",
        f"next_hidden_{state_1}",
    )
    rows = tuple(
        _row_values(analysis, model, depth) for depth in range(len(analysis.rows))
    )
    return AnalysisTable(columns=columns, rows=rows)


def _row_values(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
    depth: int,
) -> RowValues:
    row = analysis.rows[depth]
    observed_symbol = (
        model.labels.observables[row.observed_index]
        if row.observed_index is not None
        else None
    )
    target_symbol = (
        model.labels.observables[row.actual_target_index]
        if row.actual_target_index is not None
        else None
    )
    posterior_values = (
        float_values(row.posterior) if row.posterior is not None else None
    )
    predictive_0, predictive_1 = float_values(row.predictive)
    next_hidden_0, next_hidden_1 = float_values(row.next_hidden)
    return (
        row.depth,
        observed_symbol,
        target_symbol,
        predictive_0,
        predictive_1,
        row.entropy_bits,
        model.labels.observables[row.predicted_index],
        row.actual_target_probability,
        row.actual_target_surprisal_bits,
        row.target_classification.value
        if row.target_classification is not None
        else None,
        posterior_values[0] if posterior_values is not None else None,
        posterior_values[1] if posterior_values is not None else None,
        next_hidden_0,
        next_hidden_1,
    )
