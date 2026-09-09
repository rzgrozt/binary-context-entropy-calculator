import pytest

from binary_entropy import KTSmoothing, VMMConfig
from binary_entropy.stimulus_generation import generate_candidate_records
from binary_entropy.stimulus_metrics import sequence_metrics
from binary_entropy.stimulus_search_types import (
    InclusiveRange,
    InvalidStimulusSearchConfigurationError,
    StimulusConstraints,
    StimulusSearchConfig,
)


def _config(*, length: int = 6, limit: int = 12) -> StimulusSearchConfig:
    return StimulusSearchConfig(
        sequence_length=length,
        desired_stimuli=4,
        seed=23,
        candidate_limit=limit,
        vmm_config=VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sequence_length", 0), ("desired_stimuli", 0), ("candidate_limit", 0)],
)
def test_config_when_positive_count_is_violated_raises_typed_error(
    field: str, value: int
) -> None:
    # Given
    values = {"sequence_length": 4, "desired_stimuli": 2, "candidate_limit": 4}
    values[field] = value

    # When / Then
    with pytest.raises(InvalidStimulusSearchConfigurationError) as error:
        _ = StimulusSearchConfig(
            sequence_length=values["sequence_length"],
            desired_stimuli=values["desired_stimuli"],
            seed=1,
            candidate_limit=values["candidate_limit"],
            vmm_config=VMMConfig(),
        )
    assert error.value.parameter == field


def test_range_when_bounds_are_reversed_raises_typed_error() -> None:
    # Given / When / Then
    with pytest.raises(InvalidStimulusSearchConfigurationError):
        _ = InclusiveRange(0.8, 0.2)


@pytest.mark.parametrize(
    ("parameter", "sequence_length", "desired", "seed", "limit"),
    [
        ("sequence_length", 33, 1, 0, 1),
        ("candidate_limit", 4, 1, 0, 5001),
        ("seed", 4, 1, -1, 1),
        ("seed", 4, 1, 2**63, 1),
        ("desired_stimuli", 4, 2, 0, 1),
    ],
)
def test_config_when_safety_bound_is_violated_raises_typed_error(
    parameter: str,
    sequence_length: int,
    desired: int,
    seed: int,
    limit: int,
) -> None:
    # Given / When / Then
    with pytest.raises(InvalidStimulusSearchConfigurationError) as error:
        _ = StimulusSearchConfig(
            sequence_length=sequence_length,
            desired_stimuli=desired,
            seed=seed,
            candidate_limit=limit,
            vmm_config=VMMConfig(),
        )
    assert error.value.parameter == parameter


def test_config_when_values_are_on_safety_boundaries_accepts_and_trims_labels() -> None:
    # Given / When
    config = StimulusSearchConfig(
        sequence_length=32,
        desired_stimuli=5000,
        seed=2**63 - 1,
        candidate_limit=5000,
        vmm_config=VMMConfig(),
        symbol_mapping=(" left ", " right "),
    )

    # Then
    assert config.symbol_mapping == ("left", "right")


@pytest.mark.parametrize(
    "mapping",
    [
        ("", "B"),
        ("   ", "B"),
        ("same", " same "),
        ("A,B", "B"),
        ("A\n", "B"),
        ("A\r", "B"),
    ],
)
def test_config_when_symbol_mapping_is_unsafe_raises_typed_error(
    mapping: tuple[str, str],
) -> None:
    # Given / When / Then
    with pytest.raises(InvalidStimulusSearchConfigurationError) as error:
        _ = StimulusSearchConfig(4, 1, 0, 1, VMMConfig(), symbol_mapping=mapping)
    assert error.value.parameter == "symbol_mapping"


@pytest.mark.parametrize(
    "constraints",
    [
        StimulusConstraints(predicted_probability=InclusiveRange(-0.1, 0.5)),
        StimulusConstraints(predictive_entropy=InclusiveRange(0.0, 1.1)),
        StimulusConstraints(a_proportion=InclusiveRange(-0.1, 1.0)),
        StimulusConstraints(switch_rate=InclusiveRange(0.0, 1.1)),
        StimulusConstraints(a_count=InclusiveRange(-1, 2)),
        StimulusConstraints(effective_depth=InclusiveRange(-1, 2)),
        StimulusConstraints(context_support=InclusiveRange(-1, 2)),
        StimulusConstraints(switches=InclusiveRange(-1, 2)),
        StimulusConstraints(longest_a_run=InclusiveRange(-1, 2)),
        StimulusConstraints(longest_b_run=InclusiveRange(-1, 2)),
        StimulusConstraints(longest_run=InclusiveRange(-1, 2)),
    ],
)
def test_constraints_when_semantic_range_is_invalid_raises_typed_error(
    constraints: StimulusConstraints,
) -> None:
    # Given / When / Then
    with pytest.raises(InvalidStimulusSearchConfigurationError):
        _ = StimulusSearchConfig(4, 1, 0, 1, VMMConfig(), constraints=constraints)


def test_constraints_when_semantic_ranges_use_boundaries_are_accepted() -> None:
    # Given
    constraints = StimulusConstraints(
        predicted_probability=InclusiveRange(0, 1),
        predictive_entropy=InclusiveRange(0, 1),
        a_proportion=InclusiveRange(0, 1),
        switch_rate=InclusiveRange(0, 1),
        a_count=InclusiveRange(0, 4),
        effective_depth=InclusiveRange(0, 4),
        context_support=InclusiveRange(0, 4),
        switches=InclusiveRange(0, 3),
        longest_a_run=InclusiveRange(0, 4),
        longest_b_run=InclusiveRange(0, 4),
        longest_run=InclusiveRange(0, 4),
    )

    # When
    config = StimulusSearchConfig(4, 1, 0, 1, VMMConfig(), constraints=constraints)

    # Then
    assert config.constraints == constraints


def test_generation_when_seed_is_reused_returns_exact_sequences_and_ids() -> None:
    # Given
    config = _config()

    # When
    first = generate_candidate_records(config)
    second = generate_candidate_records(config)

    # Then
    assert first == second
    assert len(first) == config.candidate_limit
    assert len({record.sequence for record in first}) == len(first)
    assert tuple(record.sequence_id for record in first) == tuple(
        f"stimulus-{index:06d}" for index in range(1, 13)
    )


def test_generation_when_limit_exceeds_universe_samples_without_replacement() -> None:
    # Given
    config = _config(length=2, limit=20)

    # When
    records = generate_candidate_records(config)

    # Then
    assert len(records) == 4
    assert {record.sequence for record in records} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_sequence_metrics_when_known_sequence_returns_exact_values() -> None:
    # Given
    sequence = (0, 0, 1, 0, 1, 1, 1)

    # When
    metrics = sequence_metrics(sequence)

    # Then
    assert metrics.length == 7
    assert (metrics.a_count, metrics.b_count) == (3, 4)
    assert metrics.a_proportion == 3 / 7
    assert metrics.b_proportion == 4 / 7
    assert metrics.switches == 3
    assert metrics.switch_rate == 0.5
    assert metrics.alternation_tendency == metrics.switch_rate
    assert (metrics.longest_a_run, metrics.longest_b_run) == (2, 3)
    assert metrics.longest_run == 3
