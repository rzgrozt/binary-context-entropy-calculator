import csv
import io
import math
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from binary_entropy import (
    BinaryLabels,
    KTSmoothing,
    MLESmoothing,
    SequenceDataset,
    SequenceRecord,
    VMMConfig,
    analyze_vmm,
    analyze_vmm_per_sequence,
    vmm_context_evidence_csv,
    vmm_context_model_json,
    vmm_evaluation_csv,
)


class _ContextDistributionView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    displayed_context: tuple[str, ...]
    context_depth: int
    context_occurrence_count: int
    next_a_count: int
    next_b_count: int
    next_a_probability: float
    next_b_probability: float


class _ModelView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    model_identifier: str
    source_record_identifiers: tuple[str, ...]
    source_orders: tuple[int, ...]
    source_sequence_count: int
    fitted_context_distributions: tuple[_ContextDistributionView, ...]


class _RecordView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    record_identifier: str
    source_order: int
    sequence_stimulus: tuple[str, ...]
    consumed_prefix_stimulus: tuple[str, ...]
    requested_depth: int
    actual_depth: int | None


class _ContextModelView(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal[1]
    artifact_name: Literal["Context model export"]
    experimental_status: Literal["experimental"]
    experimental_notice: str
    method: Literal["vmm"]
    dataset_role: Literal["training"]
    training_dataset_identifier: str
    evaluation_dataset_identifier: None
    workflow: Literal["variable_order_markov"]
    result_scope: Literal["pooled", "per_sequence"]
    configured_depth_selection: str
    minimum_support: int
    estimation_rule: str
    smoothing_alpha: float
    records: tuple[_RecordView, ...]
    models: tuple[_ModelView, ...]


def _dataset() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (
            SequenceRecord("aa", (0, 0), actual_target_index=1),
            SequenceRecord("ba", (1, 0)),
        ),
    )


def test_vmm_context_model_json_when_pooled_is_stable_and_boundary_preserving() -> None:
    # Given
    dataset = _dataset()
    analysis = analyze_vmm(
        dataset,
        VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )

    # When
    payload = vmm_context_model_json(analysis, dataset)
    result = _ContextModelView.model_validate_json(payload, strict=True)

    # Then
    assert payload == vmm_context_model_json(analysis, dataset)
    assert payload.endswith(b"\n")
    assert result.experimental_notice
    assert result.training_dataset_identifier.startswith("sha256:")
    assert result.workflow == "variable_order_markov"
    assert result.result_scope == "pooled"
    assert result.configured_depth_selection == "deepest_supported_suffix"
    assert result.estimation_rule == "krichevsky_trofimov"
    assert tuple(record.record_identifier for record in result.records) == ("aa", "ba")
    assert tuple(record.source_order for record in result.records) == (1, 2)
    assert result.records[0].sequence_stimulus == ("A", "A")
    assert result.records[0].consumed_prefix_stimulus == ("A", "A")
    assert len(result.models) == 1
    model = result.models[0]
    assert model.model_identifier == "pooled"
    assert model.source_record_identifiers == ("aa", "ba")
    assert model.source_orders == (1, 2)
    assert tuple(
        (row.displayed_context, row.next_a_count, row.next_b_count)
        for row in model.fitted_context_distributions
    ) == (((), 3, 1), (("A",), 1, 0), (("B",), 1, 0))


def test_vmm_context_model_json_when_per_sequence_preserves_model_order() -> None:
    # Given
    dataset = _dataset()
    analysis = analyze_vmm_per_sequence(
        dataset,
        VMMConfig(smoothing=KTSmoothing(), minimum_support=1),
    )

    # When
    result = _ContextModelView.model_validate_json(
        vmm_context_model_json(analysis, dataset),
        strict=True,
    )

    # Then
    assert result.workflow == "variable_order_markov"
    assert result.result_scope == "per_sequence"
    assert tuple(model.model_identifier for model in result.models) == ("aa", "ba")
    assert tuple(model.source_record_identifiers for model in result.models) == (
        ("aa",),
        ("ba",),
    )
    assert tuple(model.source_orders for model in result.models) == ((1,), (2,))
    assert tuple(
        tuple(row.displayed_context for row in model.fitted_context_distributions)
        for model in result.models
    ) == (((), ("A",)), ((), ("B",)))


def test_vmm_context_evidence_csv_when_mle_context_is_unseen_is_explicit() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    dataset = SequenceDataset(
        labels,
        (SequenceRecord("mle-unseen", (0, 1, 1), actual_target_index=0),),
    )
    analysis = analyze_vmm_per_sequence(
        dataset,
        VMMConfig(smoothing=MLESmoothing(), minimum_support=1),
    )

    # When
    payload = vmm_context_evidence_csv(analysis, dataset)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert list(rows[0])[18:20] == ["workflow", "result_scope"]
    assert {row["workflow"] for row in rows} == {"variable_order_markov"}
    assert {row["result_scope"] for row in rows} == {"per_sequence"}
    assert tuple(row["requested_depth"] for row in rows) == ("0", "1", "2", "3")
    assert {row["actual_depth"] for row in rows} == {"1"}
    assert rows[0]["observed_target"] == "A"
    assert float(rows[0]["target_probability"] or "") == 1 / 3
    assert float(rows[0]["target_surprisal_bits"] or "") == -math.log2(1 / 3)
    assert float(rows[1]["target_probability"] or "") == 0.0
    assert math.isinf(float(rows[1]["target_surprisal_bits"] or ""))
    assert rows[2]["support_status"] == "unavailable"
    assert rows[2]["sparse_status"] == "unavailable"
    assert rows[2]["suffix_backoff_reason"] == (
        "MLE unavailable: unseen context has no occurrences in the training dataset."
    )
    assert rows[2]["next_A_probability"] == ""
    assert rows[2]["predictive_entropy_bits"] == ""
    assert rows[2]["observed_target"] == "A"
    assert rows[2]["target_probability"] == ""
    assert rows[2]["target_surprisal_bits"] == ""
    assert rows[2]["evaluation_status"] == "In-sample evaluation unavailable"
    probability_text = rows[0]["next_A_probability"]
    assert probability_text is not None
    assert float(probability_text) == 1 / 3
    assert len(probability_text.split(".", maxsplit=1)[1]) >= 12
    assert "nan" not in payload.lower()


def test_vmm_evaluation_csv_when_targets_are_training_data_labels_in_sample() -> None:
    # Given
    labels = BinaryLabels(states=("S1", "S2"), observables=("=A", "+B"))
    dataset = SequenceDataset(
        labels,
        (
            SequenceRecord(
                "@alpha",
                (0, 0, 1, 0, 0, 1, 0, 0),
                actual_target_index=1,
            ),
            SequenceRecord("-empty", ()),
        ),
    )
    analysis = analyze_vmm_per_sequence(
        dataset,
        VMMConfig(smoothing=KTSmoothing(), minimum_support=2),
    )

    # When
    payload = vmm_evaluation_csv(analysis, dataset)
    rows = list(csv.DictReader(io.StringIO(payload)))

    # Then
    assert list(rows[0])[18:20] == ["workflow", "result_scope"]
    assert {row["workflow"] for row in rows} == {"variable_order_markov"}
    assert {row["result_scope"] for row in rows} == {"per_sequence"}
    assert tuple(row["source_order"] for row in rows) == ("1", "2")
    assert rows[0]["record_identifier"] == "'@alpha"
    assert rows[0]["sequence_stimulus"].startswith("'=A")
    assert rows[0]["observed_target"] == "'+B"
    assert rows[0]["evaluation_dataset_identifier"] == ""
    assert rows[0]["evaluation_status"] == "In-sample evaluation, not held out"
    assert float(rows[0]["target_probability"] or "") == 5 / 6
    assert len(rows[0]["target_probability"].split(".", maxsplit=1)[1]) >= 12
    assert rows[1]["record_identifier"] == "'-empty"
    assert rows[1]["evaluation_status"] == "Not supplied"
    assert rows[1]["next_A_probability"] == ""
    assert rows[1]["target_probability"] == ""
    assert "nan" not in payload.lower()
