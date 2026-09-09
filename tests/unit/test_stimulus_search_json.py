from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from binary_entropy import KTSmoothing, VMMConfig
from binary_entropy.stimulus_search import search_stimuli
from binary_entropy.stimulus_search_json import stimulus_generator_config_json
from binary_entropy.stimulus_search_results import (
    MatchTolerances,
    MetricSummary,
    SearchResult,
)
from binary_entropy.stimulus_search_types import StimulusSearchConfig
from binary_entropy.stimulus_targets import assign_targets, validate_stimuli


class _MetricSummaryView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    minimum: float | str
    mean: float | str
    maximum: float | str


class _MatchTolerancesView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    entropy: float | str
    prediction_strength: float | str
    a_proportion: float | str
    switch_rate: float | str
    run_structure: float | str
    effective_depth: float | str


class _ValidationReportView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    stimulus_count: int
    predicted_a_count: int
    predicted_b_count: int
    tie_count: int
    unavailable_count: int
    expected_target_count: int
    unexpected_target_count: int
    tie_target_count: int
    unassigned_target_count: int
    a_count: _MetricSummaryView | None
    b_count: _MetricSummaryView | None
    a_proportion: _MetricSummaryView | None
    b_proportion: _MetricSummaryView | None
    switches: _MetricSummaryView | None
    switch_rate: _MetricSummaryView | None
    longest_a_run: _MetricSummaryView | None
    longest_b_run: _MetricSummaryView | None
    longest_run: _MetricSummaryView | None
    predicted_probability: _MetricSummaryView | None
    predictive_entropy: _MetricSummaryView | None
    effective_depth: _MetricSummaryView | None
    context_support: _MetricSummaryView | None
    target_surprisal: _MetricSummaryView | None
    imbalance_flags: tuple[str, ...]


class _GeneratorConfigView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    package_version: str
    timestamp_utc: str
    seed: int
    minimum_support: int
    smoothing_alpha: float
    assignment_seed: int | None
    match_tolerances: _MatchTolerancesView | None
    validation_report: _ValidationReportView | None


def _result() -> SearchResult:
    config = StimulusSearchConfig(
        sequence_length=8,
        desired_stimuli=3,
        seed=4,
        candidate_limit=8,
        vmm_config=VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )
    return search_stimuli(config, timestamp=datetime(2026, 9, 8, tzinfo=UTC))


def test_config_json_when_stage_data_is_absent_serializes_explicit_nulls() -> None:
    # Given
    result = _result()

    # When
    first = stimulus_generator_config_json(result)
    second = stimulus_generator_config_json(result)
    payload = _GeneratorConfigView.model_validate_json(first, strict=True)

    # Then
    assert first == second
    assert payload.package_version == "0.1.0"
    assert payload.timestamp_utc == result.timestamp_utc
    assert payload.seed == result.config.seed
    assert payload.minimum_support == result.config.vmm_config.minimum_support
    assert payload.smoothing_alpha == result.config.vmm_config.smoothing.alpha
    assert payload.assignment_seed is None
    assert payload.match_tolerances is None
    assert payload.validation_report is None


def test_config_json_when_stage_data_is_supplied_captures_reproducibility() -> None:
    # Given
    result = _result()
    assigned = assign_targets(result.selected, seed=29)
    tolerances = MatchTolerances(entropy=0.2, prediction_strength=0.3)
    report = validate_stimuli(assigned)

    # When
    first = stimulus_generator_config_json(
        result,
        match_tolerances=tolerances,
        assignment_seed=29,
        validation_report=report,
    )
    second = stimulus_generator_config_json(
        result,
        match_tolerances=tolerances,
        assignment_seed=29,
        validation_report=report,
    )
    payload = _GeneratorConfigView.model_validate_json(first, strict=True)

    # Then
    assert first == second
    assert payload.assignment_seed == 29
    assert payload.match_tolerances is not None
    assert payload.match_tolerances.model_dump() == {
        "a_proportion": 1.0,
        "effective_depth": "infinity",
        "entropy": 0.2,
        "prediction_strength": 0.3,
        "run_structure": 1.0,
        "switch_rate": 1.0,
    }
    assert payload.validation_report is not None
    serialized = payload.validation_report
    assert (
        serialized.stimulus_count,
        serialized.predicted_a_count,
        serialized.predicted_b_count,
        serialized.tie_count,
        serialized.unavailable_count,
        serialized.expected_target_count,
        serialized.unexpected_target_count,
        serialized.tie_target_count,
        serialized.unassigned_target_count,
    ) == (
        report.stimulus_count,
        report.predicted_a_count,
        report.predicted_b_count,
        report.tie_count,
        report.unavailable_count,
        report.expected_target_count,
        report.unexpected_target_count,
        report.tie_target_count,
        report.unassigned_target_count,
    )
    assert serialized.imbalance_flags == report.imbalance_flags
    summaries: tuple[tuple[_MetricSummaryView | None, MetricSummary | None], ...] = (
        (serialized.a_count, report.a_count),
        (serialized.b_count, report.b_count),
        (serialized.a_proportion, report.a_proportion),
        (serialized.b_proportion, report.b_proportion),
        (serialized.switches, report.switches),
        (serialized.switch_rate, report.switch_rate),
        (serialized.longest_a_run, report.longest_a_run),
        (serialized.longest_b_run, report.longest_b_run),
        (serialized.longest_run, report.longest_run),
        (serialized.predicted_probability, report.predicted_probability),
        (serialized.predictive_entropy, report.predictive_entropy),
        (serialized.effective_depth, report.effective_depth),
        (serialized.context_support, report.context_support),
        (serialized.target_surprisal, report.target_surprisal),
    )
    for exported, original in summaries:
        assert exported is not None
        assert original is not None
        assert exported.model_dump() == {
            "minimum": original.minimum,
            "mean": original.mean,
            "maximum": original.maximum,
        }
