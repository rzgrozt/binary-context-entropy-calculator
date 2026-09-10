# Markov Comparison Examples

This page demonstrates bspe's **first-order Markov method** and shows how to compare Markov with other analysis methods.

## Example 1: Basic Markov Analysis

Analyze a sequence using first-order Markov:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
)

labels = BinaryLabels(
    states=("A", "B"),
    observables=("Heads", "Tails"),
)

sequence = "HTHHTHHHT"
record = SequenceRecord(
    sequence_id="coin_flips",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

result = analyze_dataset(dataset, MarkovAnalysisRequest())

record_result = result.records[0]
print(f"Sequence: {sequence}")
print(f"Predictive Entropy: {record_result.predictive_entropy_bits:.6f} bits")
print(f"\nTransition Matrix:")
print(f"  H → H: {record_result.transition_probabilities[0][0]:.3f}")
print(f"  H → T: {record_result.transition_probabilities[0][1]:.3f}")
print(f"  T → H: {record_result.transition_probabilities[1][0]:.3f}")
print(f"  T → T: {record_result.transition_probabilities[1][1]:.3f}")
```

## Example 2: Markov with Different Smoothing Methods

Compare KT, MLE, and additive smoothing:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    KTSmoothing,
    MLESmoothing,
    AdditiveSmoothing,
)

labels = BinaryLabels(states=("A", "B"), observables=("X", "Y"))

# Sequence with sparse transitions
sequence = "XYXYXYXXX"  # X→X only appears at end
record = SequenceRecord(
    sequence_id="sparse_transitions",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

smoothing_methods = {
    "KT Smoothing": KTSmoothing(),
    "MLE (No Smoothing)": MLESmoothing(),
    "Additive (α=1)": AdditiveSmoothing(alpha=1.0),
    "Additive (α=0.5)": AdditiveSmoothing(alpha=0.5),
}

print("Entropy comparison with different smoothing methods:")
print(f"Sequence: {sequence}\n")

for name, smoothing in smoothing_methods.items():
    result = analyze_dataset(
        dataset,
        MarkovAnalysisRequest(smoothing=smoothing),
    )
    entropy = result.records[0].predictive_entropy_bits
    print(f"{name:20} → {entropy:.6f} bits")
```

**Output** (illustrative):
```
Entropy comparison with different smoothing methods:
Sequence: XYXYXYXXX

KT Smoothing         → 0.811278 bits
MLE (No Smoothing)   → 0.650000 bits
Additive (α=1)       → 0.811278 bits
Additive (α=0.5)     → 0.735632 bits
```

## Example 3: FIXED_MODEL vs CUMULATIVE_PREFIX Prediction Modes

Compare two strategies for handling unseen transitions:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    MarkovPredictionMode,
    KTSmoothing,
)

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))

# Sequence that evolves over time
sequence = "00000111"
record = SequenceRecord(
    sequence_id="evolving",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

print(f"Sequence: {sequence}\n")

# FIXED_MODEL: Use same model throughout
result_fixed = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(
        prediction_mode=MarkovPredictionMode.FIXED_MODEL,
        smoothing=KTSmoothing(),
    ),
)

print("FIXED_MODEL mode (constant model):")
print(f"  Entropy: {result_fixed.records[0].predictive_entropy_bits:.6f} bits")
print(f"  Interpretation: Model learned from full sequence, applied unchanged")

# CUMULATIVE_PREFIX: Update model as prediction progresses
result_cumulative = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(
        prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX,
        smoothing=KTSmoothing(),
    ),
)

print("\nCUMULATIVE_PREFIX mode (adaptive model):")
print(f"  Entropy: {result_cumulative.records[0].predictive_entropy_bits:.6f} bits")
print(f"  Interpretation: Model updated as each new observation added")
```

## Example 4: Markov vs. Shannon

Compare predictive (Markov) with descriptive (Shannon) entropy:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
)

labels = BinaryLabels(states=("A", "B"), observables=("Red", "Blue"))

sequences = [
    "RBRBRBRB",  # Perfectly alternating
    "RRRRBBBB",  # Two runs
    "RRBRRBBRR",  # Mixed
]

print("Markov (Predictive) vs Shannon (Descriptive):\n")
print(f"{'Sequence':<20} {'Markov':<15} {'Shannon':<15}")
print("-" * 50)

for seq in sequences:
    record = SequenceRecord(
        sequence_id=seq,
        sequence=seq,
        labels=labels,
    )
    dataset = SequenceDataset(labels=labels, records=[record])
    
    markov_result = analyze_dataset(dataset, MarkovAnalysisRequest())
    markov_entropy = markov_result.records[0].predictive_entropy_bits
    
    shannon_result = analyze_dataset(dataset, ShannonAnalysisRequest())
    shannon_entropy = shannon_result.records[0].predictive_entropy_bits
    
    print(f"{seq:<20} {markov_entropy:<15.6f} {shannon_entropy:<15.6f}")
```

**Output** (illustrative):
```
Markov (Predictive) vs Shannon (Descriptive):

Sequence             Markov          Shannon        
--------------------------------------------------
RBRBRBRB             0.000000        1.000000       
RRRRBBBB             0.000000        1.000000       
RRBRRBBRR            0.811278        0.811278       
```

Observations:
- **Alternating**: Markov entropy = 0 (perfectly predictable); Shannon = 1 (50% each).
- **Two runs**: Markov entropy = 0 (predictable within runs); Shannon = 1 (50% each).
- **Mixed**: Both similar (no strong predictive pattern).

## Example 5: Pooled vs. Per-Sequence Markov Analysis

Compare aggregate and individual sequence analysis:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    MarkovResultScope,
)

labels = BinaryLabels(states=("A", "B"), observables=("H", "T"))

sequences = [
    "HHHHHH",  # Highly predictable (always H)
    "HTHTH",  # Perfectly alternating
    "HHTTHH",  # Runs
]

records = [
    SequenceRecord(
        sequence_id=f"seq_{i}",
        sequence=seq,
        labels=labels,
    )
    for i, seq in enumerate(sequences)
]

dataset = SequenceDataset(labels=labels, records=records)

# Per-sequence analysis
result_per_seq = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(scope=MarkovResultScope.PER_SEQUENCE),
)

print("Per-Sequence Markov Analysis:")
for rec in result_per_seq.records:
    print(f"  {rec.sequence_id}: {rec.predictive_entropy_bits:.6f} bits")

# Pooled analysis
result_pooled = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(scope=MarkovResultScope.POOLED),
)

print(f"\nPooled Markov Analysis: {result_pooled.records[0].predictive_entropy_bits:.6f} bits")
```

## Example 6: Markov with Targets

Evaluate Markov prediction accuracy:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
)

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))

records = [
    SequenceRecord(
        sequence_id="test1",
        sequence="001001001",
        actual_target_index=0,  # Next should be 0
        labels=labels,
    ),
    SequenceRecord(
        sequence_id="test2",
        sequence="110110110",
        actual_target_index=1,  # Next should be 1
        labels=labels,
    ),
    SequenceRecord(
        sequence_id="test3",
        sequence="001100110",
        actual_target_index=0,  # Next should be 0
        labels=labels,
    ),
]

dataset = SequenceDataset(labels=labels, records=records)
result = analyze_dataset(dataset, MarkovAnalysisRequest())

correct = 0
print("Target Prediction Accuracy:\n")
for rec in result.records:
    match = rec.predicted_target_label == rec.actual_target_label
    correct += int(match)
    print(f"{rec.sequence_id}:")
    print(f"  Predicted: {rec.predicted_target_label} "
          f"(P={rec.predicted_target_probability:.3f})")
    print(f"  Actual: {rec.actual_target_label}")
    print(f"  Correct: {match}\n")

print(f"Accuracy: {correct}/{len(result.records)} ({correct/len(result.records):.0%})")
```

## Example 7: Comparing All Methods on Same Data

Multi-method comparison:

```python
from bspe import (
    BinaryLabels,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
)

labels = BinaryLabels(states=("A", "B"), observables=("X", "Y"))

sequences = [
    "XYXYXYXYXY",
    "XXXXXXXXXX",
    "XYXXYXXYXY",
]

records = [
    SequenceRecord(
        sequence_id=f"seq_{i}",
        sequence=seq,
        labels=labels,
    )
    for i, seq in enumerate(sequences)
]

dataset = SequenceDataset(labels=labels, records=records)

# Markov analysis
markov_result = analyze_dataset(dataset, MarkovAnalysisRequest())

# Shannon analysis
shannon_result = analyze_dataset(dataset, ShannonAnalysisRequest())

# Comparison table
print(f"{'Sequence':<20} {'Markov (bits)':<18} {'Shannon (bits)':<18}")
print("-" * 60)

for markov_rec, shannon_rec in zip(markov_result.records, shannon_result.records):
    print(f"{markov_rec.sequence_id:<20} "
          f"{markov_rec.predictive_entropy_bits:<18.6f} "
          f"{shannon_rec.predictive_entropy_bits:<18.6f}")
```

## Best Practices

1. **Smoothing Selection**: Use KT for small datasets; MLE for large, complete data.
2. **Prediction Mode**: Use FIXED_MODEL for stationary sequences; CUMULATIVE_PREFIX for evolving patterns.
3. **Scope**: Use POOLED for quick summaries; PER_SEQUENCE for diagnostics.
4. **Targets**: Include targets to measure real prediction quality.
5. **Comparison**: Always compare Markov against Shannon to understand temporal structure.

## See Also

- [Markov Method User Guide](../user-guide/methods/markov.md)
- [Example: VMM Analysis](vmm-analysis.md)
- [API Reference: Analysis Methods](../api/analysis-methods.md)
