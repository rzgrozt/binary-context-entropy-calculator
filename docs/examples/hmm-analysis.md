# HMM Analysis Examples

This page demonstrates how to use bspe's **Hidden Markov Model (HMM)** method for sequence analysis. HMMs are useful when you have a theoretical model of hidden states and want to evaluate how well sequences fit that model.

## Example 1: Basic HMM Configuration

Define a simple HMM and analyze a sequence:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
)

# Create labels: 2 hidden states, 2 observables
labels = BinaryLabels(
    states=("Fair", "Biased"),
    observables=("Heads", "Tails"),
)

# Define an HMM:
# - Initial: 60% Fair, 40% Biased
# - Transition: Fair→Fair 90%, Fair→Biased 10%, etc.
# - Emission: Fair emits both equally; Biased favors Heads
model = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.9, 0.1], [0.2, 0.8]],
    emission=[[0.5, 0.5], [0.7, 0.3]],
)

# Create a sequence
sequence = "HHTHHHTHHHTHHTHH"  # Heads and Tails
record = SequenceRecord(
    sequence_id="test_sequence",
    sequence=sequence,
    labels=labels,
)

# Create dataset
dataset = SequenceDataset(labels=labels, records=[record])

# Analyze with HMM
result = analyze_dataset(dataset, HMMAnalysisRequest(model=model))

# Inspect results
record_result = result.records[0]
print(f"Predictive Entropy: {record_result.predictive_entropy_bits:.6f} bits")
print(f"Per-position predictions:")
for i, (pos_result, obs) in enumerate(
    zip(record_result.per_position_results, sequence)
):
    print(f"  Position {i} ({obs}): "
          f"P(Heads)={pos_result.probability_observable_a:.3f}, "
          f"Entropy={pos_result.entropy_bits:.3f} bits")
```

## Example 2: Multiple Models for Hypothesis Testing

Compare different HMM configurations to test competing hypotheses:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
)

labels = BinaryLabels(states=("S0", "S1"), observables=("0", "1"))

# Hypothesis 1: Weak state coupling (mostly stay in same state)
weak_coupling = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.95, 0.05], [0.05, 0.95]],
    emission=[[0.9, 0.1], [0.1, 0.9]],
)

# Hypothesis 2: Strong state coupling (frequent transitions)
strong_coupling = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.5, 0.5], [0.5, 0.5]],
    emission=[[0.9, 0.1], [0.1, 0.9]],
)

# Hypothesis 3: Noisy emissions (states less distinct)
noisy_emissions = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.9, 0.1], [0.1, 0.9]],
    emission=[[0.6, 0.4], [0.4, 0.6]],
)

# Same sequence for all hypotheses
sequence = "01010101"  # Alternating pattern
record = SequenceRecord(
    sequence_id="test",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

# Test each hypothesis
hypotheses = {
    "Weak Coupling": weak_coupling,
    "Strong Coupling": strong_coupling,
    "Noisy Emissions": noisy_emissions,
}

for name, model in hypotheses.items():
    result = analyze_dataset(dataset, HMMAnalysisRequest(model=model))
    entropy = result.records[0].predictive_entropy_bits
    print(f"{name:20} → Entropy: {entropy:.6f} bits")
```

**Output** (illustrative):
```
Weak Coupling            → Entropy: 0.145382 bits
Strong Coupling          → Entropy: 1.000000 bits
Noisy Emissions          → Entropy: 0.970951 bits
```

The weak coupling model produces lower entropy because it predicts alternation accurately.

## Example 3: HMM with Targets for Prediction Quality

Evaluate HMM predictions against known targets:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
)

labels = BinaryLabels(states=("Active", "Inactive"), observables=("On", "Off"))

model = BinaryHMM(
    labels=labels,
    initial=[0.8, 0.2],
    transition=[[0.7, 0.3], [0.4, 0.6]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)

# Sequence with known next value
records = [
    SequenceRecord(
        sequence_id="seq1",
        sequence="OnOnOffOnOn",
        actual_target_index=0,  # Next is "On"
        labels=labels,
    ),
    SequenceRecord(
        sequence_id="seq2",
        sequence="OffOffOnOffOff",
        actual_target_index=1,  # Next is "Off"
        labels=labels,
    ),
]

dataset = SequenceDataset(labels=labels, records=records)
result = analyze_dataset(dataset, HMMAnalysisRequest(model=model))

print("Target Agreement:\n")
for rec in result.records:
    print(f"{rec.sequence_id}:")
    print(f"  Predicted: {rec.predicted_target_label}")
    print(f"  Actual: {rec.actual_target_label}")
    print(f"  Match: {rec.predicted_target_label == rec.actual_target_label}")
    print(f"  P(predicted): {rec.predicted_target_probability:.3f}")
```

## Example 4: Forward Algorithm Details

Access the forward algorithm output for deeper inspection:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
)

labels = BinaryLabels(states=("H", "L"), observables=("A", "B"))

model = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.8, 0.2], [0.3, 0.7]],
    emission=[[0.7, 0.3], [0.3, 0.7]],
)

sequence = "AABAA"
record = SequenceRecord(
    sequence_id="example",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

result = analyze_dataset(dataset, HMMAnalysisRequest(model=model))
record_result = result.records[0]

print(f"Sequence: {sequence}\n")
print("Per-Position Analysis:")
for i, pos_result in enumerate(record_result.per_position_results):
    obs = sequence[i]
    print(f"\nPosition {i} (observed: {obs}):")
    print(f"  Predictive entropy: {pos_result.entropy_bits:.6f} bits")
    print(f"  P(A | history): {pos_result.probability_observable_a:.6f}")
    print(f"  P(B | history): {1 - pos_result.probability_observable_a:.6f}")
    if hasattr(pos_result, 'state_probabilities'):
        print(f"  State posteriors: H={pos_result.state_probabilities[0]:.4f}, "
              f"L={pos_result.state_probabilities[1]:.4f}")
```

## Example 5: Batch HMM Analysis

Analyze multiple sequences with the same model:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
    HMMResultScope,
)

labels = BinaryLabels(states=("S1", "S2"), observables=("X", "Y"))

model = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.7, 0.3], [0.4, 0.6]],
    emission=[[0.8, 0.2], [0.3, 0.7]],
)

sequences = [
    "XYXYXYXY",
    "XXXYYYYY",
    "XYXXYXYX",
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

# Analyze per-sequence
result_per_seq = analyze_dataset(
    dataset,
    HMMAnalysisRequest(model=model, scope=HMMResultScope.PER_SEQUENCE),
)

print("Per-Sequence Entropy:")
for rec in result_per_seq.records:
    print(f"  {rec.sequence_id}: {rec.predictive_entropy_bits:.6f} bits")

# Analyze pooled (aggregate)
result_pooled = analyze_dataset(
    dataset,
    HMMAnalysisRequest(model=model, scope=HMMResultScope.POOLED),
)

print(f"\nPooled Entropy: {result_pooled.records[0].predictive_entropy_bits:.6f} bits")
```

## Example 6: Deterministic HMM (No Entropy)

Create an HMM that perfectly predicts:

```python
from bspe import (
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    analyze_dataset,
    HMMAnalysisRequest,
)

labels = BinaryLabels(states=("Start", "Alt"), observables=("A", "B"))

# Deterministic model: alternates perfectly
deterministic_model = BinaryHMM(
    labels=labels,
    initial=[1.0, 0.0],  # Always start in "Start"
    transition=[[0.0, 1.0], [1.0, 0.0]],  # Always switch
    emission=[[1.0, 0.0], [0.0, 1.0]],  # Emit deterministically
)

sequence = "ABABAB"
record = SequenceRecord(
    sequence_id="deterministic",
    sequence=sequence,
    labels=labels,
)
dataset = SequenceDataset(labels=labels, records=[record])

result = analyze_dataset(dataset, HMMAnalysisRequest(model=deterministic_model))
record_result = result.records[0]

print(f"Sequence: {sequence}")
print(f"Predictive Entropy: {record_result.predictive_entropy_bits:.6f} bits")
print("(Zero entropy because model perfectly predicts each symbol)")
```

## Best Practices

1. **Model Specification**: Ensure probabilities sum to 1.0 in each row of transition and emission matrices.
2. **Hypothesis Clarity**: Document what each model represents (e.g., "alternation" vs. "clustering").
3. **Multiple Models**: Compare competing theories by testing several models on the same data.
4. **Scope Selection**: Use `PER_SEQUENCE` for individual diagnostics, `POOLED` for aggregate statistics.
5. **Target Evaluation**: Include targets to measure prediction quality against known outcomes.

## See Also

- [HMM Method Guide](../user-guide/methods/hmm.md)
- [Concepts](../getting-started/concepts.md)
- [API Reference: Analysis Methods](../api/analysis-methods.md)
