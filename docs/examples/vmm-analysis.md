# VMM Analysis Example

Complete working example of Variable-Order Markov analysis with different configurations.

## Basic VMM Analysis

```python
from bspe import (
    BinaryLabels,
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
    parse_manual_batch,
)

# Define labels
labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B"),
)

# Parse a simple sequence
dataset = parse_manual_batch("A, B, B, A, A, B, A, A", labels)

print("=" * 70)
print("EXAMPLE 1: Pooled VMM with KT Smoothing (Default)")
print("=" * 70)

# Default configuration: KT smoothing, minimum support 2
result_kt = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    ),
)

record = result_kt.records[0]
print(f"Sequence: {record.sequence_str}")
print(f"Effective context depth: {record.effective_context_depth}")
print(f"Context used: {record.context_used}")
print(f"Support count: {record.support_count}")
print(f"Predicted target index: {record.predicted_target_index}")
print(f"  (0=A, 1=B, None=tie)")
print(f"Probability A: {record.probability_a:.6f}")
print(f"Probability B: {record.probability_b:.6f}")
print(f"Predictive entropy: {record.predictive_entropy_bits:.6f} bits")

print("\n" + "=" * 70)
print("EXAMPLE 2: Pooled VMM with MLE Smoothing")
print("=" * 70)

from bspe import MLESmoothing

# MLE smoothing (alpha=0, no pseudocount)
result_mle = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(
            minimum_support=2,
            smoothing=MLESmoothing(),
        ),
        result_scope=VMMResultScope.POOLED,
    ),
)

record_mle = result_mle.records[0]
print(f"Sequence: {record_mle.sequence_str}")
print(f"Predictive entropy (MLE): {record_mle.predictive_entropy_bits:.6f} bits")
print(f"Probability A (MLE): {record_mle.probability_a}")
print(f"Probability B (MLE): {record_mle.probability_b}")

print("\n" + "=" * 70)
print("EXAMPLE 3: Per-Sequence VMM Analysis")
print("=" * 70)

# Per-sequence analysis: each record fits its own model
result_per_seq = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.PER_SEQUENCE,
    ),
)

record_per = result_per_seq.records[0]
print(f"Sequence: {record_per.sequence_str}")
print(f"Per-sequence effective depth: {record_per.effective_context_depth}")
print(f"Per-sequence entropy: {record_per.predictive_entropy_bits:.6f} bits")

print("\n" + "=" * 70)
print("EXAMPLE 4: Context Depth Evidence Table")
print("=" * 70)

# Examine the depth-evidence table to see backoff behavior
print("Depth-evidence table (examined suffixes):")
print(f"{'Depth':<6} {'Context':<15} {'Support':<8} {'Entropy':<10} {'Used':<5}")
print("-" * 50)

for evidence in record.depth_evidence:
    context_str = evidence.context if evidence.context else "(none)"
    used_marker = "✓" if evidence.used else ""
    print(
        f"{evidence.depth:<6} {context_str:<15} "
        f"{evidence.support_count:<8} {evidence.entropy_bits:<10.6f} {used_marker:<5}"
    )

print("\n" + "=" * 70)
print("EXAMPLE 5: Multiple Sequences (Batch)")
print("=" * 70)

# Parse multiple sequences
batch_dataset = parse_manual_batch(
    "A, B, B, A, A\nB, A, A, B, B\nA, A, B, B, A",
    labels,
)

result_batch = analyze_dataset(
    batch_dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    ),
)

print(f"Analyzed {len(result_batch.records)} sequences:")
for i, rec in enumerate(result_batch.records, 1):
    print(
        f"  Sequence {i}: {rec.sequence_str} "
        f"→ entropy={rec.predictive_entropy_bits:.6f} bits"
    )

print("\n" + "=" * 70)
print("EXAMPLE 6: With Optional Target Assessment")
print("=" * 70)

# Parse with an optional target (observed next symbol)
dataset_with_target = parse_manual_batch(
    "A, B, B, A, A, B, A, A",
    labels,
    target_index=1,  # Target is B (index 1)
)

result_with_target = analyze_dataset(
    dataset_with_target,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    ),
)

record_target = result_with_target.records[0]
print(f"Sequence: {record_target.sequence_str}")
print(f"Predicted target index: {record_target.predicted_target_index}")
print(f"Observed target index: {record_target.observed_target_index}")
print(f"Target surprisal: {record_target.target_assessment.surprisal_bits:.6f} bits")
print("  (surprisal = -log2(probability of observed target))")

print("\n" + "=" * 70)
print("EXAMPLE 7: Minimum Support Comparison")
print("=" * 70)

for min_sup in [1, 2, 3]:
    result = analyze_dataset(
        dataset,
        VMMAnalysisRequest(
            config=VMMConfig(minimum_support=min_sup),
            result_scope=VMMResultScope.POOLED,
        ),
    )
    rec = result.records[0]
    print(
        f"minimum_support={min_sup}: "
        f"depth={rec.effective_context_depth}, "
        f"entropy={rec.predictive_entropy_bits:.6f}"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
```

## Key Takeaways

1. **VMMConfig**: Control smoothing, minimum support
2. **VMMResultScope**: Choose pooled vs. per-sequence
3. **Depth Evidence**: Inspect which contexts are used and why
4. **Target Assessment**: Optional evaluation of observed outcomes
5. **Batching**: Analyze multiple sequences simultaneously

See [Variable-order Markov](../user-guide/methods/vmm.md) for detailed explanation.
