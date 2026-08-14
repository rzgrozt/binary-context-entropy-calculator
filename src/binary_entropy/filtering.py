"""Reusable hidden-Markov filtering operations."""

import math
from dataclasses import dataclass

import numpy as np

from binary_entropy.domain import (
    BinaryHMM,
    FloatArray,
    ObservableIndex,
    float_values,
    readonly_vector,
)
from binary_entropy.errors import NumericalInvariantError, ZeroLikelihoodError


@dataclass(frozen=True, slots=True)
class ObservedSymbol:
    """One indexed observation at a one-based sequence position."""

    index: ObservableIndex
    position: int


@dataclass(frozen=True, slots=True)
class FilterStep:
    """Hidden and observed distributions after one observed symbol."""

    posterior: FloatArray
    next_hidden: FloatArray
    predictive: FloatArray


def initial_prediction(model: BinaryHMM) -> FloatArray:
    """Predict the first observable with no transition."""
    calculated: FloatArray = model.initial @ model.emission
    return _normalized(calculated, "initial predictive distribution")


def filter_observation(
    model: BinaryHMM,
    prior: FloatArray,
    observation: ObservedSymbol,
) -> FilterStep:
    """Filter one observation and predict the following observable."""
    likelihood: FloatArray = prior * model.emission[:, observation.index]
    likelihood_total = math.fsum(float_values(likelihood))
    if likelihood_total == 0.0:
        raise ZeroLikelihoodError(
            observable_index=observation.index,
            position=observation.position,
        )
    posterior = _normalized(likelihood, "posterior distribution")
    next_hidden_calculated: FloatArray = posterior @ model.transition
    next_hidden = _normalized(next_hidden_calculated, "next hidden distribution")
    predictive_calculated: FloatArray = next_hidden @ model.emission
    predictive = _normalized(predictive_calculated, "predictive distribution")
    return FilterStep(
        posterior=posterior,
        next_hidden=next_hidden,
        predictive=predictive,
    )


def _normalized(values: FloatArray, quantity: str) -> FloatArray:
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)):
        raise NumericalInvariantError(quantity=quantity, value=float("nan"))
    total = math.fsum(float_values(values))
    if not math.isfinite(total) or total <= 0.0:
        raise NumericalInvariantError(quantity=quantity, value=total)
    normalized: FloatArray = values / total
    return readonly_vector(normalized)
