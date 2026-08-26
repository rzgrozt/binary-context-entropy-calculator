"""Unrounded HMM table values and three-decimal metric text."""

import math
from dataclasses import dataclass

import pandas as pd

from binary_entropy.domain import BinaryHMM, SequenceAnalysis, float_values
from binary_entropy.information import surprisal
from binary_entropy.ui.tokens import format_ui_decimal

type VisibleCell = str | int | float | None
type VisibleRow = tuple[VisibleCell, ...]


@dataclass(frozen=True, slots=True)
class FinalMetrics:
    """Complete final-prefix metrics prepared for visible display."""

    depth: int
    context: str
    probability_0: str
    probability_1: str
    predicted_target: str
    entropy_bits: str
    surprisal_0: str
    surprisal_1: str
    posterior: str
    posterior_0: str
    posterior_1: str
    next_hidden_0: str
    next_hidden_1: str


def final_metrics(analysis: SequenceAnalysis, model: BinaryHMM) -> FinalMetrics:
    """Prepare the complete final HMM prediction at visible precision."""
    final = analysis.rows[-1]
    probability_0, probability_1 = float_values(final.predictive)
    next_hidden_0, next_hidden_1 = float_values(final.next_hidden)
    posterior = float_values(final.posterior) if final.posterior is not None else None
    posterior_text = (
        "Available after the final observation"
        if posterior is not None
        else "Unavailable before observation"
    )
    return FinalMetrics(
        depth=final.depth,
        context=_context(analysis, model, final.depth),
        probability_0=format_ui_decimal(probability_0),
        probability_1=format_ui_decimal(probability_1),
        predicted_target=model.labels.observables[final.predicted_index],
        entropy_bits=format_ui_decimal(final.entropy_bits),
        surprisal_0=format_information(surprisal(probability_0)),
        surprisal_1=format_information(surprisal(probability_1)),
        posterior=posterior_text,
        posterior_0=(
            format_ui_decimal(posterior[0]) if posterior is not None else posterior_text
        ),
        posterior_1=(
            format_ui_decimal(posterior[1]) if posterior is not None else posterior_text
        ),
        next_hidden_0=format_ui_decimal(next_hidden_0),
        next_hidden_1=format_ui_decimal(next_hidden_1),
    )


def prefix_dataframe(analysis: SequenceAnalysis, model: BinaryHMM) -> pd.DataFrame:
    """Build an unrounded HMM prefix dataframe for native Streamlit display."""
    state_0, state_1 = model.labels.states
    observable_0, observable_1 = model.labels.observables
    columns = (
        "Depth",
        "Observed context",
        "Observed symbol",
        f"P(next {observable_0})",
        f"P(next {observable_1})",
        "Predicted target",
        "Predictive entropy (bits)",
        f"Surprisal if next {observable_0} (bits)",
        f"Surprisal if next {observable_1} (bits)",
        f"Posterior {state_0}",
        f"Posterior {state_1}",
        f"Next-hidden {state_0}",
        f"Next-hidden {state_1}",
        "Posterior status",
    )
    rows = tuple(
        _visible_row(analysis, model, depth) for depth in range(len(analysis.rows))
    )
    return pd.DataFrame.from_records(rows, columns=columns)


def format_information(value: float) -> str:
    """Format finite and impossible-event information for visible display."""
    return "infinity" if math.isinf(value) else format_ui_decimal(value)


def _visible_row(
    analysis: SequenceAnalysis, model: BinaryHMM, depth: int
) -> VisibleRow:
    row = analysis.rows[depth]
    probability_0, probability_1 = float_values(row.predictive)
    next_hidden_0, next_hidden_1 = float_values(row.next_hidden)
    posterior = float_values(row.posterior) if row.posterior is not None else None
    observed = (
        None
        if row.observed_index is None
        else model.labels.observables[row.observed_index]
    )
    return (
        row.depth,
        _context(analysis, model, depth),
        observed,
        probability_0,
        probability_1,
        model.labels.observables[row.predicted_index],
        row.entropy_bits,
        surprisal(probability_0),
        surprisal(probability_1),
        None if posterior is None else posterior[0],
        None if posterior is None else posterior[1],
        next_hidden_0,
        next_hidden_1,
        "Unavailable before observation" if posterior is None else "Available",
    )


def _context(analysis: SequenceAnalysis, model: BinaryHMM, depth: int) -> str:
    labels = tuple(
        model.labels.observables[index] for index in analysis.sequence[:depth]
    )
    return ", ".join(labels) if labels else "(empty prefix)"
