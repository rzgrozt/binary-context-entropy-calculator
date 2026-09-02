from dataclasses import replace

import pytest

from binary_entropy.markov_types import MarkovBatchAnalysis, MarkovResultScope
from binary_entropy.methods.hmm import HMMBatchAnalysis
from binary_entropy.methods.shannon import ShannonBatchAnalysis
from binary_entropy.ui.state import ActualTargetChoice
from binary_entropy.ui.workbench_state import (
    InputMode,
    MarkovControls,
    MarkovWorkflow,
    MethodChoice,
    VMMSmoothingChoice,
    WorkbenchCalculationFailure,
    WorkbenchCalculationSuccess,
    calculate_workbench,
    default_workbench_form,
    parse_workbench_dataset,
)
from binary_entropy.vmm_types import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMAnalysis,
    VMMResultScope,
)


def test_default_workbench_when_created_selects_only_markov() -> None:
    # Given / When
    form = default_workbench_form()

    # Then
    assert form.methods == (MethodChoice.MARKOV,)
    assert tuple(MethodChoice) == (
        MethodChoice.MARKOV,
        MethodChoice.HMM,
        MethodChoice.SHANNON,
    )
    assert form.intake.mode is InputMode.SINGLE
    assert form.intake.text == "A, B, B, A, A, A, B"
    assert form.markov.workflow is MarkovWorkflow.VMM
    assert form.markov.vmm_smoothing_choice is VMMSmoothingChoice.KT
    assert form.markov.vmm_custom_alpha == 0.5
    assert form.markov.minimum_support == 2


@pytest.mark.parametrize(
    ("smoothing_choice", "custom_alpha", "expected"),
    [
        (VMMSmoothingChoice.KT, 0.75, KTSmoothing()),
        (VMMSmoothingChoice.MLE, 0.75, MLESmoothing()),
        (VMMSmoothingChoice.ADDITIVE, 0.75, AdditiveSmoothing(0.75)),
    ],
)
def test_vmm_smoothing_when_choice_changes_constructs_typed_value(
    smoothing_choice: VMMSmoothingChoice,
    custom_alpha: float,
    expected: KTSmoothing | AdditiveSmoothing,
) -> None:
    # Given
    controls = default_workbench_form().markov
    configured = replace(
        controls,
        vmm_smoothing_choice=smoothing_choice,
        vmm_custom_alpha=custom_alpha,
    )

    # When
    smoothing = configured.vmm_smoothing()

    # Then
    assert smoothing == expected


def test_default_workbench_when_calculated_routes_to_vmm() -> None:
    # Given
    form = default_workbench_form()

    # When
    outcome = calculate_workbench(form)

    # Then
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    result = outcome.results[0]
    assert isinstance(result, VMMAnalysis)
    assert result.config.smoothing == KTSmoothing()
    assert result.config.minimum_support == 2
    assert result.result_scope is VMMResultScope.POOLED


def test_single_input_when_text_contains_newlines_remains_one_record() -> None:
    # Given
    form = default_workbench_form()
    intake = replace(form.intake, text="A, B\nA B")

    # When
    dataset = parse_workbench_dataset(replace(form, intake=intake))

    # Then
    assert len(dataset.records) == 1
    assert dataset.records[0].sequence == (0, 1, 0, 1)


def test_batch_input_when_lines_are_submitted_keeps_record_boundaries() -> None:
    # Given
    form = default_workbench_form()
    intake = replace(form.intake, mode=InputMode.BATCH, text="A,A\nB,B")

    # When
    dataset = parse_workbench_dataset(replace(form, intake=intake))

    # Then
    assert tuple(record.sequence for record in dataset.records) == ((0, 0), (1, 1))


def test_first_order_target_when_changed_does_not_change_markov_fit() -> None:
    # Given
    default = default_workbench_form()
    form = replace(
        default,
        markov=replace(default.markov, workflow=MarkovWorkflow.FIRST_ORDER),
    )
    target_form = replace(
        form,
        intake=replace(form.intake, actual_target=ActualTargetChoice.FIRST),
    )

    # When
    without_target = calculate_workbench(form)
    with_target = calculate_workbench(target_form)

    # Then
    assert isinstance(without_target, WorkbenchCalculationSuccess)
    assert isinstance(with_target, WorkbenchCalculationSuccess)
    first = without_target.results[0]
    second = with_target.results[0]
    assert isinstance(first, MarkovBatchAnalysis)
    assert isinstance(second, MarkovBatchAnalysis)
    assert first.model.transition_counts == second.model.transition_counts
    assert first.model.transition_matrix[0] == pytest.approx(
        second.model.transition_matrix[0]
    )


def test_first_order_workflow_when_calculated_returns_markov_batch() -> None:
    # Given
    form = default_workbench_form()
    first_order = replace(
        form,
        markov=replace(form.markov, workflow=MarkovWorkflow.FIRST_ORDER),
    )

    # When
    outcome = calculate_workbench(first_order)

    # Then
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    assert isinstance(outcome.results[0], MarkovBatchAnalysis)


def test_vmm_workflow_when_scope_is_per_sequence_maps_scope() -> None:
    # Given
    form = default_workbench_form()
    per_sequence = replace(
        form,
        markov=replace(form.markov, result_scope=MarkovResultScope.PER_SEQUENCE),
    )

    # When
    outcome = calculate_workbench(per_sequence)

    # Then
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    result = outcome.results[0]
    assert isinstance(result, VMMAnalysis)
    assert result.result_scope is VMMResultScope.PER_SEQUENCE


def test_all_methods_when_calculated_preserve_workbench_order() -> None:
    # Given
    form = default_workbench_form()
    selected = replace(
        form,
        methods=(
            MethodChoice.MARKOV,
            MethodChoice.HMM,
            MethodChoice.SHANNON,
        ),
    )

    # When
    outcome = calculate_workbench(selected)

    # Then
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    assert tuple(type(result) for result in outcome.results) == (
        VMMAnalysis,
        HMMBatchAnalysis,
        ShannonBatchAnalysis,
    )


def test_method_fingerprint_when_hmm_changes_keeps_markov_current() -> None:
    # Given
    form = default_workbench_form()
    changed_hmm = replace(
        form,
        hmm_model=replace(form.hmm_model, initial=(0.25, 0.75)),
    )

    # When / Then
    assert changed_hmm.method_fingerprint(MethodChoice.MARKOV) == (
        form.method_fingerprint(MethodChoice.MARKOV)
    )
    assert changed_hmm.method_fingerprint(MethodChoice.HMM) != (
        form.method_fingerprint(MethodChoice.HMM)
    )


def test_method_fingerprint_when_preset_name_changes_only_hmm_changes() -> None:
    # Given
    form = default_workbench_form()
    renamed = replace(form, preset_name="renamed model")

    # When / Then
    assert renamed.method_fingerprint(MethodChoice.MARKOV) == (
        form.method_fingerprint(MethodChoice.MARKOV)
    )
    assert renamed.method_fingerprint(MethodChoice.HMM) != (
        form.method_fingerprint(MethodChoice.HMM)
    )
    assert renamed.method_fingerprint(MethodChoice.SHANNON) == (
        form.method_fingerprint(MethodChoice.SHANNON)
    )


@pytest.mark.parametrize(
    "changed_markov",
    [
        replace(
            default_workbench_form().markov,
            workflow=MarkovWorkflow.FIRST_ORDER,
        ),
        replace(
            default_workbench_form().markov,
            vmm_smoothing_choice=VMMSmoothingChoice.MLE,
        ),
        replace(
            default_workbench_form().markov,
            vmm_smoothing_choice=VMMSmoothingChoice.ADDITIVE,
        ),
        replace(default_workbench_form().markov, vmm_custom_alpha=0.75),
        replace(default_workbench_form().markov, minimum_support=3),
        replace(
            default_workbench_form().markov,
            result_scope=MarkovResultScope.PER_SEQUENCE,
        ),
    ],
)
def test_method_fingerprint_when_vmm_control_changes_only_markov_changes(
    changed_markov: MarkovControls,
) -> None:
    # Given
    form = default_workbench_form()
    changed = replace(form, markov=changed_markov)

    # When
    changed_markov_fingerprint = changed.method_fingerprint(MethodChoice.MARKOV)

    # Then
    assert changed_markov_fingerprint != form.method_fingerprint(MethodChoice.MARKOV)
    assert changed.method_fingerprint(MethodChoice.HMM) == form.method_fingerprint(
        MethodChoice.HMM
    )
    assert changed.method_fingerprint(MethodChoice.SHANNON) == form.method_fingerprint(
        MethodChoice.SHANNON
    )


def test_invalid_shared_labels_when_calculated_fail_before_any_method() -> None:
    # Given
    form = default_workbench_form()
    invalid = replace(form, intake=replace(form.intake, observable_labels=("A", " A ")))

    # When
    outcome = calculate_workbench(invalid)

    # Then
    assert isinstance(outcome, WorkbenchCalculationFailure)
    assert "distinct" in outcome.message
