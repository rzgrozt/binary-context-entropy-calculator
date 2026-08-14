"""Canonical visible result values derived from core analysis objects."""

import math
from dataclasses import dataclass

import pandas as pd  # noqa: RUF100  # noqa: PANDAS_OK

from binary_entropy.domain import BinaryHMM, SequenceAnalysis, float_values
from binary_entropy.information import surprisal
from binary_entropy.presentation import format_decimal

type VisibleCell = str | int
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
    """Prepare the complete final prediction at canonical precision."""
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
        probability_0=format_decimal(probability_0),
        probability_1=format_decimal(probability_1),
        predicted_target=model.labels.observables[final.predicted_index],
        entropy_bits=format_decimal(final.entropy_bits),
        surprisal_0=_information_text(surprisal(probability_0)),
        surprisal_1=_information_text(surprisal(probability_1)),
        posterior=posterior_text,
        posterior_0=(
            format_decimal(posterior[0]) if posterior is not None else posterior_text
        ),
        posterior_1=(
            format_decimal(posterior[1]) if posterior is not None else posterior_text
        ),
        next_hidden_0=format_decimal(next_hidden_0),
        next_hidden_1=format_decimal(next_hidden_1),
    )


def prefix_dataframe(analysis: SequenceAnalysis, model: BinaryHMM) -> pd.DataFrame:
    """Build a depth-sorted visible table without internal look-ahead targets."""
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
    )
    rows: tuple[VisibleRow, ...] = tuple(
        _visible_row(analysis, model, depth) for depth in range(len(analysis.rows))
    )
    return pd.DataFrame.from_records(rows, columns=columns)


def _visible_row(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
    depth: int,
) -> VisibleRow:
    row = analysis.rows[depth]
    probability_0, probability_1 = float_values(row.predictive)
    next_hidden_0, next_hidden_1 = float_values(row.next_hidden)
    posterior = float_values(row.posterior) if row.posterior is not None else None
    observed_symbol = (
        model.labels.observables[row.observed_index]
        if row.observed_index is not None
        else "Before first observation"
    )
    posterior_unavailable = "Unavailable before observation"
    return (
        row.depth,
        _context(analysis, model, depth),
        observed_symbol,
        format_decimal(probability_0),
        format_decimal(probability_1),
        model.labels.observables[row.predicted_index],
        format_decimal(row.entropy_bits),
        _information_text(surprisal(probability_0)),
        _information_text(surprisal(probability_1)),
        (
            format_decimal(posterior[0])
            if posterior is not None
            else posterior_unavailable
        ),
        (
            format_decimal(posterior[1])
            if posterior is not None
            else posterior_unavailable
        ),
        format_decimal(next_hidden_0),
        format_decimal(next_hidden_1),
    )


def _context(
    analysis: SequenceAnalysis,
    model: BinaryHMM,
    depth: int,
) -> str:
    labels = tuple(
        model.labels.observables[index] for index in analysis.sequence[:depth]
    )
    return ", ".join(labels) if labels else "(empty prefix)"


def _information_text(value: float) -> str:
    return "Infinite (zero probability)" if math.isinf(value) else format_decimal(value)


def format_information(value: float) -> str:
    """Format finite or impossible-event information for visible display."""
    return _information_text(value)
