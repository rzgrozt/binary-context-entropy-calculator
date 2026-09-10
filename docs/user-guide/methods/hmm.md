# Hidden Markov Model (HMM)

Hidden Markov Model (HMM) analysis filters observed sequences through a **configured two-state, two-observable model** to compute hidden state posteriors and predictive distributions.

## Overview

### What HMM Does

1. **Takes a configured model** (not fitted from data)
2. **Performs forward filtering** at each observation
3. **Computes posterior** distribution over hidden states
4. **Predicts** next symbol using posterior and emission probabilities

### When to Use HMM

- **Configured models**: You have a prior model to test
- **Schema validation**: Verify against schema-v1 presets
- **Interpretable states**: Hidden states have meaning
- **Comparison**: Compare against data-fitted methods

## Configuration

### BinaryHMM

```python
from bspe import BinaryHMM, BinaryLabels

labels = BinaryLabels(
    states=("State A", "State B"),
    observables=("Outcome 1", "Outcome 2")
)

hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],           # P(state at t=0)
    transition=[
        [0.7, 0.3],               # From state A: 70% stay, 30% go to B
        [0.2, 0.8],               # From state B: 20% go to A, 80% stay
    ],
    emission=[
        [0.9, 0.1],               # In state A: 90% emit outcome 1, 10% emit outcome 2
        [0.2, 0.8],               # In state B: 20% emit outcome 1, 80% emit outcome 2
    ],
    preset_name="my-model",       # Optional name
)
```

### Probability Constraints

All probabilities must satisfy:
- **Range**: Each probability in [0, 1]
- **Sum**: Initial distribution sums to 1.0
- **Sum**: Each row of transition matrix sums to 1.0
- **Sum**: Each row of emission matrix sums to 1.0

```python
# ✓ Valid
initial = [0.6, 0.4]           # Sums to 1.0
transition = [[0.7, 0.3], [0.2, 0.8]]

# ✗ Invalid
initial = [0.6, 0.3]           # Sums to 0.9, not 1.0
```

### Schema-v1 Presets

**bspe** supports preset models:

```python
from bspe import HMMAnalysisRequest

# Use a preset by name
request = HMMAnalysisRequest(
    model=load_preset_hmm("schema-v1-preset-name")
)
```

## Analysis

### HMMAnalysisRequest

```python
from bspe import HMMAnalysisRequest, analyze_dataset

request = HMMAnalysisRequest(model=hmm)
result = analyze_dataset(dataset, request)
```

### Forward Filtering

For each observation, HMM updates its belief about the hidden state:

```
t=0: Initial state
       ↓ Observe outcome 1
t=1: Posterior over {state A, state B}
       ↓ Observe outcome 2
t=2: Updated posterior
       ...
```

## Results

### Per-Record Analysis

```python
result = analyze_dataset(dataset, request)

for record in result.records:
    # Identifiers
    print(record.sequence_id)
    print(record.sequence)
    
    # Overall statistics
    print(record.observed_entropy_bits)
    
    # Target assessment (if provided)
    if record.target_assessment:
        print(record.target_assessment.probability)
        print(record.target_assessment.surprisal_bits)
```

### Depth-by-Depth Analysis

```python
# Each position in the sequence
for row in record.rows:
    print(f"Position {row.depth}")
    print(f"  Observed: {row.observed_index}")
    print(f"  Posterior: {row.posterior}")  # Belief over hidden states
    print(f"  Predictive: {row.predictive}")  # Prediction at this position
    print(f"  Entropy: {row.entropy_bits:.4f} bits")
```

### Hidden State Posterior

The **posterior** is the belief distribution over hidden states after observing a symbol:

```python
row = record.rows[0]
posterior = row.posterior  # [P(state 0 | obs), P(state 1 | obs)]

if posterior[0] > posterior[1]:
    print("More likely in state 0")
else:
    print("More likely in state 1")
```

### Predictive Distribution

The **predictive** is the distribution over next symbols:

```python
row = record.rows[-1]
predictive = row.predictive  # [P(outcome 0), P(outcome 1)]

print(f"Predicted outcome: {row.predicted_index}")
print(f"Probabilities: {predictive}")
```

## Example

### Basic Analysis

```python
from bspe import (
    BinaryHMM,
    BinaryLabels,
    HMMAnalysisRequest,
    analyze_dataset,
    parse_manual_batch,
)

# Setup
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B, A, A", labels)

# Configure model
hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.7, 0.3], [0.2, 0.8]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)

# Analyze
result = analyze_dataset(dataset, HMMAnalysisRequest(model=hmm))

# Results
record = result.records[0]
print(f"Sequence: {record.sequence_str}")
print(f"Observed entropy: {record.observed_entropy_bits:.4f} bits")

# Final prediction
final_row = record.rows[-1]
print(f"Final posterior: {final_row.posterior}")
print(f"Final predictive: {final_row.predictive}")
print(f"Predicted outcome: {final_row.predicted_index}")
```

### Forward Filtering Trace

```python
# See how belief changes with each observation
for i, row in enumerate(record.rows):
    obs = labels.observables[row.observed_index] if row.observed_index is not None else "—"
    print(
        f"t={i}: observed={obs}, "
        f"posterior={row.posterior}, "
        f"entropy={row.entropy_bits:.4f}"
    )
```

### Comparing Models

```python
# Model 1: Weak hidden states
hmm1 = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.6, 0.4], [0.4, 0.6]],  # Weak separation
    emission=[[0.6, 0.4], [0.4, 0.6]],
)

# Model 2: Strong hidden states
hmm2 = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.9, 0.1], [0.1, 0.9]],  # Strong separation
    emission=[[0.95, 0.05], [0.05, 0.95]],
)

result1 = analyze_dataset(dataset, HMMAnalysisRequest(model=hmm1))
result2 = analyze_dataset(dataset, HMMAnalysisRequest(model=hmm2))

print(f"Weak model entropy: {result1.records[0].observed_entropy_bits:.4f}")
print(f"Strong model entropy: {result2.records[0].observed_entropy_bits:.4f}")
```

## HMM Concepts

### Initial Distribution

Belief about hidden state at start of sequence:

```python
initial = [0.8, 0.2]  # Start in state 0 with high probability
```

### Transition Matrix

Probability of state change:

```
       to A  to B
from A [0.7  0.3]    # In A, 70% stay, 30% go to B
from B [0.2  0.8]    # In B, 20% go to A, 80% stay
```

### Emission Matrix

Probability of each observation given hidden state:

```
        emit A  emit B
state A [0.9    0.1]    # In A, 90% produce A, 10% produce B
state B [0.2    0.8]    # In B, 20% produce A, 80% produce B
```

### Filtering

Updating belief given new observation:

```
Before: P(state | history)
   ↓ Observe new symbol
After: P(state | history + observation)
```

## Performance

### Speed

| Configuration | Time |
|---------------|------|
| Any model | ~5–10ms |

**Similar to VMM, depends on sequence length**

### Memory

- Fixed: ~1KB (model is fixed size)

## Troubleshooting

### No prediction (predictive is uniform)

**Cause:** Model is uninformative or posterior is uniform

**Solution:** Check if model parameters make sense

```python
# Very uninformative emission matrix
emission = [[0.5, 0.5], [0.5, 0.5]]  # No preference for outcomes
```

### Posterior converges to one state

**Cause:** Model has strong hidden state structure

**Solution:** This is normal; check if interpretable

```python
# If posterior always converges to one state,
# hidden states may be redundant in explaining data
```

### Different results than VMM

**Cause:** HMM uses configured model, VMM fits from data

**Solution:** This is expected and useful for comparison

## See Also

- **[Variable-order Markov](vmm.md)**: Data-fitted prediction method
- **[First-order Markov](markov.md)**: Simpler baseline
- **[Input Formats](../input-formats.md)**: How to prepare data
- **[Examples](../../examples/hmm-analysis.md)**: Runnable examples
