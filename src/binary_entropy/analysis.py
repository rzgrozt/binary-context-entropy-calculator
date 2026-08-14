"""Sequence-level predictive entropy analysis."""

from dataclasses import dataclass

from binary_entropy.constants import PROBABILITY_TOLERANCE
from binary_entropy.domain import (
    BinaryHMM,
    FloatArray,
    ObservableIndex,
    PrefixResult,
    SequenceAnalysis,
    TargetAssessment,
    TargetClassification,
    float_values,
)
from binary_entropy.filtering import (
    ObservedSymbol,
    filter_observation,
    initial_prediction,
)
from binary_entropy.information import binary_entropy, surprisal


@dataclass(frozen=True, slots=True)
class _PrefixInputs:
    depth: int
    observed_index: ObservableIndex | None
    posterior: FloatArray | None
    next_hidden: FloatArray
    predictive: FloatArray
    actual_target_index: ObservableIndex | None


def analyze_sequence(
    model: BinaryHMM,
    sequence: tuple[ObservableIndex, ...],
) -> SequenceAnalysis:
    """Analyze every prefix, including depth zero and the full sequence."""
    predictive = initial_prediction(model)
    rows = [
        _prefix_result(
            _PrefixInputs(
                depth=0,
                observed_index=None,
                posterior=None,
                next_hidden=model.initial,
                predictive=predictive,
                actual_target_index=sequence[0] if sequence else None,
            )
        )
    ]
    prior = model.initial
    for depth, observed_index in enumerate(sequence, start=1):
        step = filter_observation(
            model,
            prior,
            ObservedSymbol(index=observed_index, position=depth),
        )
        actual_target_index = sequence[depth] if depth < len(sequence) else None
        rows.append(
            _prefix_result(
                _PrefixInputs(
                    depth=depth,
                    observed_index=observed_index,
                    posterior=step.posterior,
                    next_hidden=step.next_hidden,
                    predictive=step.predictive,
                    actual_target_index=actual_target_index,
                )
            )
        )
        prior = step.next_hidden
    observed_entropy = (
        binary_entropy(sequence.count(0) / len(sequence)) if sequence else None
    )
    return SequenceAnalysis(
        sequence=sequence,
        rows=tuple(rows),
        observed_entropy_bits=observed_entropy,
    )


def _prefix_result(inputs: _PrefixInputs) -> PrefixResult:
    probability_0, probability_1 = float_values(inputs.predictive)
    predicted_index: ObservableIndex = 0 if probability_0 >= probability_1 else 1
    if inputs.actual_target_index is None:
        classification = None
        target_probability = None
        target_surprisal = None
    else:
        assessment = assess_target(inputs.predictive, inputs.actual_target_index)
        target_probability = assessment.probability
        target_surprisal = assessment.surprisal_bits
        classification = assessment.classification
    return PrefixResult(
        depth=inputs.depth,
        observed_index=inputs.observed_index,
        posterior=inputs.posterior,
        next_hidden=inputs.next_hidden,
        predictive=inputs.predictive,
        entropy_bits=binary_entropy(probability_0),
        predicted_index=predicted_index,
        actual_target_index=inputs.actual_target_index,
        target_classification=classification,
        actual_target_probability=target_probability,
        actual_target_surprisal_bits=target_surprisal,
    )


def assess_target(
    predictive: FloatArray,
    actual_target_index: ObservableIndex,
) -> TargetAssessment:
    """Classify one actual target under a normalized binary prediction."""
    probability_0, probability_1 = float_values(predictive)
    probability = probability_0 if actual_target_index == 0 else probability_1
    predicted_index: ObservableIndex = 0 if probability_0 >= probability_1 else 1
    probability_gap = abs(probability_0 - probability_1)
    if probability_gap <= PROBABILITY_TOLERANCE:
        classification = TargetClassification.TIED
    elif actual_target_index == predicted_index:
        classification = TargetClassification.MODAL
    else:
        classification = TargetClassification.LOWER_PROBABILITY
    return TargetAssessment(
        actual_target_index=actual_target_index,
        probability=probability,
        surprisal_bits=surprisal(probability),
        classification=classification,
    )
