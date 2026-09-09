"""Scientific invariants across search and experimental design stages."""

import json
from dataclasses import replace

import pytest

from binary_entropy import VMMConfig
from binary_entropy.information import binary_entropy
from binary_entropy.stimulus_matching import create_complement_candidates, match_stimuli
from binary_entropy.stimulus_search import evaluate_candidate, search_stimuli
from binary_entropy.stimulus_search_json import stimulus_generator_config_json
from binary_entropy.stimulus_search_results import MatchTolerances
from binary_entropy.stimulus_search_types import (
    InclusiveRange,
    PredictedSymbol,
    StimulusConstraints,
    StimulusSearchConfig,
)


def _config(length: int = 8) -> StimulusSearchConfig:
    return StimulusSearchConfig(length, 16, 17, 256, VMMConfig())


def test_predictive_entropy_and_complements_use_the_same_binary_distribution() -> None:
    config = _config()
    result = search_stimuli(config)
    complements = create_complement_candidates(result.selected, config)
    for original, complement in zip(result.selected, complements, strict=True):
        assert original.probability_a is not None
        assert original.probability_b is not None
        assert original.probability_a + original.probability_b == pytest.approx(1)
        assert original.predictive_entropy_bits == binary_entropy(
            original.probability_a
        )
        assert complement.probability_a == original.probability_b
        assert complement.probability_b == original.probability_a
        assert complement.predictive_entropy_bits == pytest.approx(
            original.predictive_entropy_bits
        )
        assert complement.effective_context_depth == original.effective_context_depth
        assert complement.metrics.switches == original.metrics.switches
        assert complement.metrics.longest_a_run == original.metrics.longest_b_run


@pytest.mark.parametrize(
    "field",
    [
        "predicted_probability",
        "predictive_entropy",
        "effective_depth",
        "context_support",
        "a_count",
        "a_proportion",
        "switches",
        "switch_rate",
        "longest_a_run",
        "longest_b_run",
        "longest_run",
    ],
)
def test_each_hard_filter_reports_failure_independently(field: str) -> None:
    candidate = search_stimuli(_config()).selected[0]
    # Test the evaluator separately from configuration-domain validation.
    constraints = replace(StimulusConstraints(), **{field: InclusiveRange(100, 101)})
    assert evaluate_candidate(candidate, constraints) == (field,)


def test_matching_never_pairs_different_lengths_even_with_unbounded_tolerances() -> (
    None
):
    first = search_stimuli(_config(7))
    second = search_stimuli(_config(8))
    candidates = tuple(
        row for row in first.accepted if row.predicted_target_index == 0
    ) + tuple(row for row in second.accepted if row.predicted_target_index == 1)
    assert candidates
    result = match_stimuli(
        candidates,
        MatchTolerances(run_structure=float("inf"), effective_depth=float("inf")),
    )
    assert not result.pairs
    assert result.unmatched == candidates


def test_final_export_tracks_complements_that_violate_original_symbol_filter() -> None:
    config = replace(
        _config(), constraints=StimulusConstraints(predicted_symbol=PredictedSymbol.A)
    )
    result = search_stimuli(config)
    complements = create_complement_candidates(result.selected, config)
    payload = json.loads(
        stimulus_generator_config_json(result, final_candidates=complements)
    )
    assert len(payload["final_stimuli"]) == len(complements)
    assert all(
        row["original_constraint_failures"] == ["predicted_symbol"]
        for row in payload["final_stimuli"]
    )
    assert payload["final_stimuli"][0]["target"] is None
    assert payload["final_stimuli"][0]["sequence"] == "".join(
        "AB"[symbol] for symbol in complements[0].sequence
    )
