import csv
import io
from dataclasses import replace
from datetime import UTC, datetime

from binary_entropy import KTSmoothing, VMMConfig
from binary_entropy.stimulus_search import search_stimuli
from binary_entropy.stimulus_search_csv import (
    stimulus_candidate_csv,
    stimulus_experiment_csv,
    stimulus_scientific_csv,
)
from binary_entropy.stimulus_search_results import SearchResult
from binary_entropy.stimulus_search_types import StimulusSearchConfig
from binary_entropy.stimulus_targets import assign_targets


def _result() -> SearchResult:
    config = StimulusSearchConfig(
        sequence_length=8,
        desired_stimuli=3,
        seed=4,
        candidate_limit=8,
        vmm_config=VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
        symbol_mapping=("left", "right"),
    )
    return search_stimuli(config, timestamp=datetime(2026, 9, 8, tzinfo=UTC))


def test_candidate_csv_when_unassigned_keeps_metadata_blank_and_round_trips() -> None:
    # Given
    result = _result()

    # When
    payload = stimulus_candidate_csv(result)
    rows = tuple(csv.DictReader(io.StringIO(payload)))

    # Then
    assert len(rows) == result.accepted_count
    assert tuple(row["stimulus_id"] for row in rows) == tuple(
        candidate.stimulus_id for candidate in result.accepted
    )
    assert sum(row["selected"] == "True" for row in rows) == result.selected_count
    assert rows[0]["sequence"].translate(str.maketrans({"A": "0", "B": "1"}))
    assert rows[0]["symbol_mapping"] == "left|right"
    assert rows[0]["target_symbol"] == ""
    assert float(rows[0]["probability_a"]) == result.accepted[0].probability_a


def test_scientific_csv_when_exported_includes_every_vmm_evidence_row() -> None:
    # Given
    result = _result()

    # When
    rows = tuple(csv.DictReader(io.StringIO(stimulus_scientific_csv(result))))

    # Then
    assert len(rows) == sum(len(candidate.depth_rows) for candidate in result.accepted)
    selected_ids = {candidate.stimulus_id for candidate in result.selected}
    assert all(
        (row["selected"] == "True") == (row["stimulus_id"] in selected_ids)
        for row in rows
    )
    assert {row["result_scope"] for row in rows} == {"per_sequence"}
    first = result.accepted[0]
    first_row = rows[0]
    assert first_row["context_used"] == "".join(
        "A" if value == 0 else "B" for value in first.context_used or ()
    )
    assert first_row["support_count"] == str(first.support_count)
    assert float(first_row["surprisal_a_bits"]) == first.surprisal_a_bits
    assert float(first_row["surprisal_b_bits"]) == first.surprisal_b_bits
    assert first_row["group_id"] == ""
    assert first_row["pair_id"] == ""
    assert first_row["condition"] == ""
    assert first_row["target_symbol"] == ""
    assert "nan" not in stimulus_scientific_csv(result).lower()


def test_scientific_csv_when_stage_metadata_exists_copies_every_final_field() -> None:
    # Given
    result = _result()
    staged = tuple(
        replace(candidate, group_id="group-1", pair_id="pair-1")
        for candidate in assign_targets(result.selected, seed=7)
    )
    snapshot = replace(
        result,
        accepted=staged,
        selected=staged,
        accepted_count=len(staged),
        selected_count=len(staged),
    )

    # When
    row = next(csv.DictReader(io.StringIO(stimulus_scientific_csv(snapshot))))

    # Then
    candidate = staged[0]
    assert row["group_id"] == candidate.group_id
    assert row["pair_id"] == candidate.pair_id
    assert row["condition"] == candidate.condition
    assert row["target_symbol"] in {"A", "B"}
    assert float(row["target_probability"]) == candidate.target_probability
    assert float(row["target_surprisal_bits"]) == candidate.target_surprisal_bits
    assert row["target_congruency"] == candidate.target_congruency


def test_experiment_csv_when_targets_are_assigned_contains_ready_metadata() -> None:
    # Given
    result = _result()
    assigned = assign_targets(result.selected, seed=2)

    # When
    rows = tuple(csv.DictReader(io.StringIO(stimulus_experiment_csv(assigned))))

    # Then
    assert len(rows) == len(assigned)
    assert all(row["target_symbol"] in {"A", "B"} for row in rows)
    assert all(
        row["target_congruency"] in {"expected", "unexpected", "tie"} for row in rows
    )
