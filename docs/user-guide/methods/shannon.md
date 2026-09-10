# Observed Shannon Entropy

Observed Shannon Entropy analysis computes the **empirical entropy** of symbol frequencies without prediction or fitting.

## Overview

### What Shannon Entropy Does

1. **Counts** symbol frequencies in the data
2. **Computes** Shannon entropy: `H = -Σ p(x) log₂ p(x)`
3. **Reports** observed entropy in bits
4. **No prediction** or model fitting

### When to Use Shannon Entropy

- **Baseline measure**: How much information is in the data?
- **Descriptive analysis**: Simple empirical summary
- **Sanity check**: Verify data complexity
- **Comparison**: Compare against prediction methods

## Configuration

### ShannonAnalysisRequest

```python
from bspe import ShannonAnalysisRequest, analyze_dataset

request = ShannonAnalysisRequest()
result = analyze_dataset(dataset, request)
```

No configuration needed — Shannon entropy is deterministic.

## Results

### Pooled Summary

Overall entropy across all data:

```python
result = analyze_dataset(dataset, ShannonAnalysisRequest())

print(result.pooled.observable_symbols_count)  # Total symbols observed
print(result.pooled.outcome_a_count)           # Count of outcome A
print(result.pooled.outcome_b_count)           # Count of outcome B
print(result.pooled.outcome_a_proportion)      # Proportion of A
print(result.pooled.outcome_b_proportion)      # Proportion of B
print(result.pooled.entropy_bits)              # Shannon entropy
```

### Per-Record Summary

Individual entropy for each sequence:

```python
for record in result.records:
    print(f"Sequence: {record.sequence_id}")
    print(f"  A count: {record.outcome_a_count}")
    print(f"  B count: {record.outcome_b_count}")
    print(f"  Entropy: {record.entropy_bits:.4f} bits")
```

### Prefix-Level Analysis

Entropy at each position:

```python
for prefix_row in record.prefix_rows:
    print(f"Position {prefix_row.depth}")
    print(f"  A count: {prefix_row.outcome_a_count}")
    print(f"  B count: {prefix_row.outcome_b_count}")
    print(f"  Entropy: {prefix_row.entropy_bits:.4f}")
```

## Example

### Basic Analysis

```python
from bspe import (
    BinaryLabels,
    ShannonAnalysisRequest,
    analyze_dataset,
    parse_manual_batch,
)

# Setup
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B, A, A\nB, A, A, B", labels)

# Analyze
request = ShannonAnalysisRequest()
result = analyze_dataset(dataset, request)

# Pooled results
print(f"Total A: {result.pooled.outcome_a_count}")
print(f"Total B: {result.pooled.outcome_b_count}")
print(f"Pooled entropy: {result.pooled.entropy_bits:.4f} bits")

# Per-record results
for record in result.records:
    print(
        f"{record.sequence_id}: "
        f"entropy={record.entropy_bits:.4f} bits"
    )
```

### Prefix Analysis

```python
# Watch entropy grow as sequence builds up
for prefix in record.prefix_rows:
    print(f"After {prefix.depth} symbols: {prefix.entropy_bits:.4f} bits")
```

### Comparing Entropy Across Sequences

```python
entropies = [record.entropy_bits for record in result.records]
print(f"Min entropy: {min(entropies):.4f}")
print(f"Max entropy: {max(entropies):.4f}")
print(f"Mean entropy: {sum(entropies) / len(entropies):.4f}")
```

## Entropy Interpretation

### Entropy Range

**Shannon entropy** ranges from 0 to 1 bit for binary symbols:

- **0 bits**: All symbols are the same (certain)
  ```
  Sequence: A, A, A, A, A
  P(A) = 1.0, P(B) = 0.0
  H = -(1.0 * log₂(1.0)) - (0 * log₂(0)) = 0 bits
  ```

- **1 bit**: Equal split (maximum uncertainty)
  ```
  Sequence: A, B, A, B, A, B
  P(A) = 0.5, P(B) = 0.5
  H = -(0.5 * log₂(0.5)) - (0.5 * log₂(0.5)) = 1 bit
  ```

- **~0.47 bits**: Mild skew
  ```
  Sequence: A, A, B (2 A's, 1 B)
  P(A) = 0.67, P(B) = 0.33
  H ≈ 0.918 bits
  ```

### Interpretation

| Entropy | Meaning |
|---------|---------|
| 0 bits | Completely uniform (all same symbol) |
| 0.25 bits | Very skewed (e.g., 90-10 split) |
| 0.5 bits | Moderately skewed (e.g., 70-30 split) |
| 1.0 bits | Perfect balance (50-50 split) |

## Comparison with Predictive Methods

### Shannon vs. VMM

```python
# Shannon: What is the raw entropy?
shannon_result = analyze_dataset(dataset, ShannonAnalysisRequest())

# VMM: Can we predict better than random?
vmm_result = analyze_dataset(dataset, VMMAnalysisRequest(...))

shannon_entropy = shannon_result.records[0].entropy_bits
vmm_entropy = vmm_result.records[0].predictive_entropy_bits

print(f"Shannon (empirical): {shannon_entropy:.4f} bits")
print(f"VMM (predictive): {vmm_entropy:.4f} bits")

if vmm_entropy < shannon_entropy:
    print("VMM prediction is better than random")
else:
    print("VMM prediction is no better than random")
```

### Interpretation

- **VMM entropy < Shannon entropy**: VMM finds patterns
- **VMM entropy ≈ Shannon entropy**: VMM can't do better than symbol frequencies
- **VMM entropy > Shannon entropy**: Unlikely; would indicate overfitting

## Properties

### Additivity

Over independent sequences:

```python
# Entropy of combined sequences ≈ weighted average
seq1_entropy = 0.5
seq2_entropy = 0.9

combined_entropy = (0.5 + 0.9) / 2  # Rough approximation
```

### Conditional Entropy

Expected entropy at each position as sequence builds:

```python
for prefix in record.prefix_rows:
    print(f"Position {prefix.depth}: {prefix.entropy_bits:.4f}")
# Usually: entropy increases as more data accumulates
```

## Performance

### Speed

| Configuration | Time |
|---------------|------|
| Any dataset | ~1–5ms |

**Fastest method (pure counting)**

### Memory

- Fixed: ~1KB per record

## Troubleshooting

### Entropy is 0

**Cause:** Only one symbol in sequence (or all same symbol)

**Solution:** This is correct; sequence has no variety

```python
sequence = "A, A, A, A"  # Only A
# Entropy = 0 (certain)
```

### Entropy is 1 bit

**Cause:** Exactly 50-50 split

**Solution:** This is expected for balanced data

```python
sequence = "A, B, A, B, A, B"  # 50% each
# Entropy = 1 bit (maximum)
```

### Per-record entropy varies widely

**Cause:** Different sequences have different distributions

**Solution:** This is normal; investigate why

```python
# Maybe some sequences are more consistent (lower entropy)
# and others are more random (higher entropy)
```

## See Also

- **[Variable-order Markov](vmm.md)**: Prediction method
- **[First-order Markov](markov.md)**: Prediction method
- **[Input Formats](../input-formats.md)**: How to prepare data
- **[Glossary](../../reference/glossary.md)**: Entropy definition
