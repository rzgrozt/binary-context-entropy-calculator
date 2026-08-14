import csv
import io
import json
import math

import pytest

from binary_entropy.analysis import analyze_sequence
from binary_entropy.errors import PresetDecodeError, PresetSchemaError
from binary_entropy.serialization import (
    CandidateMetadata,
    candidate_summary_csv,
    model_from_preset,
    parse_preset_json,
    prefix_csv,
    preset_from_model,
    preset_json,
)
from tests.unit.helpers import hand_model, hand_sequence

type JsonScalar = str | float | bool | None
type JsonValue = JsonScalar | list[JsonScalar] | list[list[float]]
type JsonMapping = dict[str, JsonValue]


def test_preset_json_when_round_tripped_is_stable_utf8() -> None:
    preset = preset_from_model(hand_model(), "Hand model")

    first = preset_json(preset)
    parsed = parse_preset_json(first)
    second = preset_json(parsed)

    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first.decode("utf-8"))["schema_version"] == 1
    assert model_from_preset(parsed).labels == hand_model().labels


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"{not-json}",
        b'{"schema_version": 1, "preset_name": "x"}',
    ],
)
def test_parse_preset_json_when_payload_is_invalid(payload: bytes) -> None:
    expected_error = (
        PresetDecodeError
        if payload != b'{"schema_version": 1, "preset_name": "x"}'
        else PresetSchemaError
    )

    with pytest.raises(expected_error):
        _ = parse_preset_json(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra", True),
        ("schema_version", 2),
        ("initial", [math.nan, 0.0]),
        ("initial", [math.inf, 0.0]),
        ("state_labels", ["same", "same"]),
        ("transition", [[0.7, 0.4], [0.2, 0.8]]),
    ],
)
def test_parse_preset_json_when_schema_contract_is_violated(
    field: str,
    value: JsonValue,
) -> None:
    valid: JsonMapping = preset_from_model(hand_model(), "Hand model").model_dump()
    valid[field] = value
    payload = json.dumps(valid, allow_nan=True)

    with pytest.raises(PresetSchemaError):
        _ = parse_preset_json(payload)


def test_prefix_csv_when_using_hand_sequence_has_stable_columns_and_precision() -> None:
    model = hand_model()
    analysis = analyze_sequence(model, hand_sequence())

    result = prefix_csv(analysis, model)
    rows = list(csv.reader(io.StringIO(result)))

    assert rows[0] == [
        "depth",
        "observed_symbol",
        "next_target_symbol",
        "predictive_probability_A",
        "predictive_probability_B",
        "predictive_entropy_bits",
        "predicted_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "target_classification",
        "posterior_State 1",
        "posterior_State 2",
        "next_hidden_State 1",
        "next_hidden_State 2",
    ]
    assert rows[1][3:6] == ["0.620000000000", "0.380000000000", "0.958042022226"]
    assert len(rows) == 9


def test_candidate_summary_csv_when_actual_target_is_supplied_has_one_stable_row() -> (
    None
):
    model = hand_model()
    analysis = analyze_sequence(model, hand_sequence())
    metadata = CandidateMetadata(
        sequence_id="sequence-001",
        preset_name="Hand model",
        actual_target_index=0,
    )

    result = candidate_summary_csv(analysis, model, metadata)
    rows = list(csv.DictReader(io.StringIO(result)))

    assert len(rows) == 1
    assert list(rows[0]) == [
        "sequence_id",
        "preset_name",
        "state_label_0",
        "state_label_1",
        "observable_label_0",
        "observable_label_1",
        "sequence",
        "initial_0",
        "initial_1",
        "transition_00",
        "transition_01",
        "transition_10",
        "transition_11",
        "emission_00",
        "emission_01",
        "emission_10",
        "emission_11",
        "sequence_length",
        "observed_entropy_bits",
        "final_predictive_probability_0",
        "final_predictive_probability_1",
        "final_predictive_entropy_bits",
        "final_predicted_symbol",
        "actual_target_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "actual_target_classification",
    ]
    assert rows[0]["sequence_id"] == "sequence-001"
    assert rows[0]["sequence"] == "A,B,B,A,A,A,B"
    assert rows[0]["actual_target_classification"] == "lower_probability"
    assert not math.isnan(float(rows[0]["observed_entropy_bits"]))
