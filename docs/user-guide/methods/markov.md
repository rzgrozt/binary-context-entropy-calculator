# First-Order Markov

First-Order Markov analysis predicts the next symbol based only on the **previous symbol** using a fitted 2×2 transition matrix.

## Overview

### What First-Order Markov Does

1. **Fits a model** from observed symbol pairs (transitions)
2. **Builds a transition matrix** P(next | previous)
3. **Predicts** the next symbol from the previous one
4. **Calculates** stationary distribution and entropy rate (if applicable)

### When to Use First-Order Markov

- **Baseline comparison**: Compare against VMM
- **Simple patterns**: Only previous symbol matters
- **Speed**: Very fast (~1ms per record)
- **Stability**: Fewer parameters than VMM

## Configuration

### MarkovAnalysisRequest

```python
from bspe import MarkovAnalysisRequest, MarkovPredictionMode, MarkovResultScope

# Pooled analysis with fixed fitted model
request = MarkovAnalysisRequest(
    smoothing_alpha=0.5,
    prediction_mode=MarkovPredictionMode.FIXED_MODEL,
    result_scope=MarkovResultScope.POOLED,
)

result = analyze_dataset(dataset, request)
```

### Smoothing

```python
# No smoothing (MLE)
request = MarkovAnalysisRequest(smoothing_alpha=0.0)

# Default KT smoothing
request = MarkovAnalysisRequest(smoothing_alpha=0.5)

# Additive smoothing
request = MarkovAnalysisRequest(smoothing_alpha=0.1)
```

**Formula:** `P(next | current) = (count + alpha) / (total + 2*alpha)`

### Prediction Modes

#### FIXED_MODEL (Default)

One transition matrix fitted from all data, used for all predictions:

```python
request = MarkovAnalysisRequest(
    prediction_mode=MarkovPredictionMode.FIXED_MODEL
)
```

**Process:**
1. Fit one transition matrix from entire dataset
2. Use this matrix to predict for all records

**Use:** Assumes stationary process

#### CUMULATIVE_PREFIX

Re-estimate transition matrix at each position:

```python
request = MarkovAnalysisRequest(
    prediction_mode=MarkovPredictionMode.CUMULATIVE_PREFIX
)
```

**Process:**
1. For position N, use data from start to position N-1
2. Refit transition matrix with this subset
3. Use it to predict position N

**Use:** Non-stationary data or online learning scenarios

### Result Scopes

#### POOLED (Default)

One model from all records:

```python
request = MarkovAnalysisRequest(
    result_scope=MarkovResultScope.POOLED
)
```

**Pros:** More data, stable
**Cons:** Assumes homogeneity

#### PER_SEQUENCE

Independent model per record:

```python
request = MarkovAnalysisRequest(
    result_scope=MarkovResultScope.PER_SEQUENCE
)
```

**Pros:** Individual patterns
**Cons:** Less data per model

## Results

### Per-Record Analysis

```python
result = analyze_dataset(dataset, request)

for record in result.records:
    # Identifiers
    print(record.sequence_id)
    print(record.sequence)
    
    # Predictions
    print(record.probability_a)
    print(record.probability_b)
    print(record.predicted_target_index)
    
    # Uncertainty
    print(record.predictive_entropy_bits)
```

### Pooled Model

```python
# Only available in POOLED mode
print(result.model)  # Fitted 2x2 transition matrix

# Stationary distribution (if unique)
if result.stationary:
    print(result.stationary.probability_a)
    print(result.stationary.probability_b)

# Entropy rate (long-run average entropy per symbol)
print(result.entropy_rate_bits)
```

### Transition Matrix

Access the fitted model:

```python
model = result.model
print(model.transition)
# [[P(A|A), P(B|A)],
#  [P(A|B), P(B|B)]]

# Check for transitions
if model.transition[0][1] > 0:
    print(f"Can transition from A to B: {model.transition[0][1]:.4f}")
```

## Example

### Basic Analysis

```python
from bspe import (
    BinaryLabels,
    MarkovAnalysisRequest,
    analyze_dataset,
    parse_manual_batch,
)

# Setup
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B, A, A, B, A, A", labels)

# Analyze
request = MarkovAnalysisRequest(smoothing_alpha=0.5)
result = analyze_dataset(dataset, request)

# Results
record = result.records[0]
print(f"Sequence: {record.sequence_str}")
print(f"Predicted: {'A' if record.predicted_target_index == 0 else 'B'}")
print(f"P(A) = {record.probability_a:.4f}")
print(f"Entropy: {record.predictive_entropy_bits:.4f} bits")

# Model
print(f"\nTransition Matrix:")
print(result.model.transition)
```

### Comparing Smoothing Methods

```python
# MLE (no smoothing)
result_mle = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(smoothing_alpha=0.0)
)

# KT (default)
result_kt = analyze_dataset(
    dataset,
    MarkovAnalysisRequest(smoothing_alpha=0.5)
)

# Compare
print(f"MLE: P(A|B) = {result_mle.model.transition[1][0]:.4f}")
print(f"KT:  P(A|B) = {result_kt.model.transition[1][0]:.4f}")
```

### Stationary Distribution

```python
result = analyze_dataset(dataset, MarkovAnalysisRequest())

if result.stationary:
    print(f"Stationary P(A) = {result.stationary.probability_a:.4f}")
    print(f"Stationary P(B) = {result.stationary.probability_b:.4f}")
    print(f"Entropy Rate = {result.entropy_rate_bits:.4f} bits")
else:
    print("No unique stationary distribution (e.g., absorbing state)")
```

## Transition Matrix Properties

### Ergodic

Both states can reach each other (all entries > 0):

```python
# Ergodic: Can reach any state from any other
[[0.7, 0.3],    # From A, go to A or B
 [0.2, 0.8]]    # From B, go to A or B

# Has unique stationary distribution
```

### Absorbing

One or more states with self-loops (diagonal = 1.0):

```python
# Absorbing: Once in A, stay in A
[[1.0, 0.0],    # From A, always stay in A
 [0.5, 0.5]]    # From B, can leave

# No unique stationary distribution (depends on starting state)
```

## Performance

### Speed

| Configuration | Time |
|---------------|------|
| Any configuration | ~1–5ms |

**First-order Markov is ~100x faster than VMM**

### Memory

- Fixed: ~1KB (always the same 2×2 matrix)

### Comparison with VMM

| Aspect | VMM | Markov |
|--------|-----|--------|
| Speed | 100ms | 1ms |
| Accuracy | Higher | Lower |
| Contexts | ~1000 | 1 |
| Memory | ~100KB | ~1KB |

## Troubleshooting

### No stationary distribution

**Cause:** Absorbing state or unreachable state

**Solution:** Check transition matrix diagonals and structure

```python
print(result.model.transition)
# If any diagonal = 1.0, that's an absorbing state
```

### Entropy is 0

**Cause:** Model is deterministic (one outcome has P=1.0)

**Solution:** This is correct behavior; model has zero prediction uncertainty

### All predictions are the same

**Cause:** Transition matrix has same output for all previous symbols

**Solution:** Markov may not be the right model; try VMM

## See Also

- **[Variable-order Markov](vmm.md)**: More complex context-aware model
- **[HMM](hmm.md)**: Configured model without fitting
- **[Input Formats](../input-formats.md)**: How to prepare data
- **[Examples](../../examples/markov-comparison.md)**: Markov vs. VMM comparison
