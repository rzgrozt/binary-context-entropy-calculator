from binary_entropy import (
    BinaryLabels,
    SequenceDataset,
    SequenceRecord,
    VMMConfig,
    analyze_vmm,
    vmm_context_evidence_csv,
    vmm_context_model_json,
    vmm_evaluation_csv,
)
from binary_entropy.ui.vmm_artifacts import (
    VMM_EXPERIMENTAL_NOTICE,
    vmm_download_artifacts,
    vmm_reproducibility_lines,
)


def _dataset() -> SequenceDataset:
    labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
    return SequenceDataset(
        labels,
        (
            SequenceRecord(
                "recurrent-aa",
                (0, 0, 1, 0, 0, 1, 0, 0),
                actual_target_index=1,
            ),
        ),
    )


def test_vmm_download_artifacts_when_result_is_current_use_raw_public_exports() -> None:
    # Given
    dataset = _dataset()
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))

    # When
    artifacts = vmm_download_artifacts(analysis, dataset)

    # Then
    assert tuple(artifact.name for artifact in artifacts) == (
        "Context model export",
        "Context evidence export",
        "Evaluation export",
    )
    assert tuple(artifact.label for artifact in artifacts) == (
        "Download Context model export (JSON)",
        "Download Context evidence export (CSV)",
        "Download Evaluation export (CSV)",
    )
    assert tuple(artifact.file_name for artifact in artifacts) == (
        "vmm-pooled-context-model.json",
        "vmm-pooled-context-evidence.csv",
        "vmm-pooled-evaluation.csv",
    )
    assert tuple(artifact.mime for artifact in artifacts) == (
        "application/json",
        "text/csv; charset=utf-8",
        "text/csv; charset=utf-8",
    )
    assert artifacts[0].data == vmm_context_model_json(analysis, dataset)
    assert artifacts[1].data == vmm_context_evidence_csv(analysis, dataset)
    assert artifacts[2].data == vmm_evaluation_csv(analysis, dataset)
    assert "Experimental" in VMM_EXPERIMENTAL_NOTICE


def test_vmm_reproducibility_when_target_is_training_data_is_explicit() -> None:
    # Given
    dataset = _dataset()
    analysis = analyze_vmm(dataset, VMMConfig(minimum_support=2))

    # When
    lines = vmm_reproducibility_lines(analysis, dataset)
    text = "\n".join(lines)

    # Then
    assert "Estimator: krichevsky_trofimov" in text
    assert "Smoothing alpha: 0.500" in text
    assert "Support rule: minimum_support=2" in text
    assert "Sparse rule: support below 2 is sparse" in text
    assert "Result scope: pooled" in text
    pooled_rule = (
        "Pooled rule: sum within-record context counts "
        "without crossing record boundaries"
    )
    assert pooled_rule in text
    assert "Training dataset role: training" in text
    assert "Evaluation dataset identifier: not present" in text
    assert "Parsed record count: 1" in text
    assert "Record IDs in source order: recurrent-aa" in text
    assert "Sequence lengths in source order: recurrent-aa=8" in text
    assert "Requested depths: recurrent-aa=0..8" in text
    assert "Actual selected depths: recurrent-aa=2" in text
    assert "Backoff selections: recurrent-aa=backed_off_to_shorter_suffix" in text
    assert "Target evaluation: recurrent-aa=In-sample evaluation, not held out" in text
    assert "Visible precision: exactly 3 decimal places" in text
    assert "Raw export precision: unrounded float64" in text
    assert "Ordering: submitted record order, then ascending requested depth" in text
