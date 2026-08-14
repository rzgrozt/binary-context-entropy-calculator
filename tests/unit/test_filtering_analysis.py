import math

import numpy as np
import pytest

from binary_entropy.analysis import analyze_sequence
from binary_entropy.domain import (
    BinaryHMM,
    BinaryLabels,
    ObservableIndex,
    TargetClassification,
    float_values,
)
from binary_entropy.errors import ZeroLikelihoodError
from binary_entropy.filtering import (
    ObservedSymbol,
    filter_observation,
    initial_prediction,
)
from binary_entropy.information import surprisal
from tests.unit.helpers import hand_model, hand_sequence, load_hand_fixture


def test_initial_prediction_when_using_hand_model_has_no_transition() -> None:
    model = hand_model()

    result = initial_prediction(model)

    np.testing.assert_allclose(result, [0.62, 0.38], atol=1e-15, rtol=0.0)


def test_filter_observation_when_first_symbol_is_a_matches_manual_values() -> None:
    model = hand_model()

    result = filter_observation(
        model, model.initial, ObservedSymbol(index=0, position=1)
    )

    np.testing.assert_allclose(
        result.posterior,
        [27 / 31, 4 / 31],
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.predictive,
        [0.6448387096774194, 0.3551612903225807],
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.next_hidden,
        [0.6354838709677418, 0.36451612903225805],
        atol=1e-15,
        rtol=0.0,
    )


def test_filter_observation_when_symbol_is_impossible() -> None:
    model = BinaryHMM(
        labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
        initial=[1.0, 0.0],
        transition=[[1.0, 0.0], [0.0, 1.0]],
        emission=[[1.0, 0.0], [1.0, 0.0]],
    )

    with pytest.raises(ZeroLikelihoodError) as captured:
        _ = filter_observation(
            model,
            model.initial,
            ObservedSymbol(index=1, position=1),
        )

    assert captured.value.observable_index == 1
    assert captured.value.position == 1


def test_analyze_sequence_when_empty_has_only_depth_zero() -> None:
    model = hand_model()

    result = analyze_sequence(model, ())

    assert len(result.rows) == 1
    assert result.rows[0].depth == 0
    assert result.rows[0].posterior is None
    np.testing.assert_allclose(result.rows[0].next_hidden, model.initial)
    assert result.observed_entropy_bits is None


def test_analyze_sequence_when_using_hand_fixture_matches_every_row() -> None:
    fixture = load_hand_fixture()

    result = analyze_sequence(hand_model(), hand_sequence())

    assert len(result.rows) == len(fixture.rows)
    for actual, expected in zip(result.rows, fixture.rows, strict=True):
        assert actual.depth == expected.depth
        assert actual.observed_index == expected.observed_index
        assert actual.predicted_index == expected.predicted_index
        if expected.posterior is None:
            assert actual.posterior is None
        else:
            assert actual.posterior is not None
            np.testing.assert_allclose(
                actual.posterior,
                expected.posterior,
                atol=1e-14,
                rtol=0.0,
            )
        np.testing.assert_allclose(
            actual.next_hidden,
            expected.next_hidden,
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            actual.predictive,
            expected.predictive,
            atol=1e-14,
            rtol=0.0,
        )
        assert actual.entropy_bits == pytest.approx(expected.entropy_bits, abs=1e-14)
        predictive_0, predictive_1 = float_values(actual.predictive)
        assert surprisal(predictive_0) == pytest.approx(
            expected.surprisal_bits[0], abs=1e-14
        )
        assert surprisal(predictive_1) == pytest.approx(
            expected.surprisal_bits[1], abs=1e-14
        )
    assert result.observed_entropy_bits == pytest.approx(
        fixture.observed_entropy_bits,
        abs=1e-14,
    )


@pytest.mark.parametrize("length", [1, 2, 5, 8, 20, 100, 10_000])
def test_analyze_sequence_when_length_varies_has_normalized_n_plus_one_rows(
    length: int,
) -> None:
    sequence: tuple[ObservableIndex, ...] = tuple(
        0 if index % 2 == 0 else 1 for index in range(length)
    )

    result = analyze_sequence(hand_model(), sequence)

    assert len(result.rows) == length + 1
    for row in result.rows:
        assert math.isclose(row.predictive.sum(), 1.0, abs_tol=1e-12, rel_tol=0.0)
        assert math.isclose(row.next_hidden.sum(), 1.0, abs_tol=1e-12, rel_tol=0.0)
        if row.posterior is not None:
            assert math.isclose(row.posterior.sum(), 1.0, abs_tol=1e-12, rel_tol=0.0)


@pytest.mark.parametrize(
    ("predictive_emission", "target", "classification"),
    [
        ((0.8, 0.2), 0, TargetClassification.MODAL),
        ((0.8, 0.2), 1, TargetClassification.LOWER_PROBABILITY),
        ((0.5, 0.5), 1, TargetClassification.TIED),
    ],
)
def test_target_classification_when_distribution_varies(
    predictive_emission: tuple[float, float],
    target: ObservableIndex,
    classification: TargetClassification,
) -> None:
    model = BinaryHMM(
        labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
        initial=[0.5, 0.5],
        transition=[[1.0, 0.0], [0.0, 1.0]],
        emission=[predictive_emission, predictive_emission],
    )

    result = analyze_sequence(model, (target,))

    assert result.rows[0].target_classification is classification
    assert result.rows[0].predicted_index == 0
    assert result.rows[0].actual_target_probability == predictive_emission[target]


def test_analyze_sequence_when_target_probability_is_zero_has_infinite_surprisal() -> (
    None
):
    model = BinaryHMM(
        labels=BinaryLabels(states=("S1", "S2"), observables=("A", "B")),
        initial=[1.0, 0.0],
        transition=[[1.0, 0.0], [0.0, 1.0]],
        emission=[[1.0, 0.0], [0.0, 1.0]],
    )

    with pytest.raises(ZeroLikelihoodError):
        _ = analyze_sequence(model, (1,))
