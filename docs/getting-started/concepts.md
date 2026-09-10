# Core Concepts

## Binary Sequences

**bspe** analyzes binary sequences — ordered lists of two distinct symbols.

```python
# Represented as tuples of indices (0 or 1)
sequence = (0, 1, 1, 0, 0, 1, 0)  # A, B, B, A, A, B, A

# With labels
labels = BinaryLabels(observables=("A", "B"))
# 0 → "A", 1 → "B"
```

## Records and Boundaries

Each sequence is an independent **record**. Records are never concatenated or merged.

```python
dataset = SequenceDataset(
    labels=labels,
    records=(
        SequenceRecord("seq-001", (0, 1, 1, 0)),
        SequenceRecord("seq-002", (1, 0, 0, 1)),
    )
)
```

**Why?** Boundary preservation ensures:
- Scientific validity (no artificial transitions)
- Reproducibility (clear data ownership)
- Interpretability (per-record analysis)

## Prediction vs. Description

### Prediction Methods

**Predict** the next symbol given a sequence:

- **Variable-order Markov (VMM)**: Context-aware, automatic suffix backoff
- **First-order Markov**: Previous symbol only
- **Hidden Markov Model (HMM)**: Configured two-state model

All return:
- `probability_a`, `probability_b`: Next-symbol distribution
- `predictive_entropy_bits`: Uncertainty (0–1 bits)
- `predicted_target_index`: Modal prediction (0, 1, or None if tie)

### Description Methods

**Describe** observed symbol frequencies:

- **Observed Shannon Entropy**: Empirical symbol counts
- No prediction, just summary statistics

## Smoothing

When fitting models, we estimate probabilities from counts. **Smoothing** handles unseen contexts:

### Krichevsky-Trofimov (KT) — Default

```
P(next = x | context) = (count_x + 0.5) / (total + 1)
```

Balanced, principled default.

### Maximum Likelihood Estimation (MLE)

```
P(next = x | context) = count_x / total
```

No pseudocount. Fails on unseen contexts.

### Additive Smoothing

```
P(next = x | context) = (count_x + alpha) / (total + 2*alpha)
```

Custom positive `alpha`. Flexible but requires justification.

## Scopes

### Pooled Analysis

One model shared across all records:

```python
result = analyze_vmm(dataset, config, VMMResultScope.POOLED)
```

**Pros**: More data per model, stable estimates
**Cons**: Assumes homogeneity

### Per-Sequence Analysis

Independent model for each record:

```python
result = analyze_vmm_per_sequence(dataset, config)
```

**Pros**: Captures individual patterns
**Cons**: Less data per model, less stable

## Entropy

**Entropy** measures prediction uncertainty (in bits):

```
H(p) = -p * log2(p) - (1-p) * log2(1-p)
```

- **0 bits**: Certain (P=1.0 or P=0.0)
- **0.5 bits**: Moderate uncertainty (P≈0.71)
- **1 bit**: Maximum uncertainty (P=0.5)

## Target Assessment

Optional: evaluate an observed next symbol after prediction.

```python
record = SequenceRecord(
    "seq-001",
    (0, 1, 1, 0),
    actual_target_index=1  # Observed next was B
)
```

**Never influences fitting.** Assessed only after prediction.

## Stimulus Search

Generate and rank binary sequences matching criteria:

1. **Generate**: Seeded sample without replacement
2. **Analyze**: VMM analysis per sequence
3. **Filter**: Hard constraints (never relaxed)
4. **Rank**: Soft preferences (weighted distance)
5. **Match** (optional): Complement pairs
6. **Assign** (optional): Balanced targets

## Determinism

All operations are deterministic:

- **Seeded randomness**: Same seed → same results
- **Sorted output**: Consistent ordering
- **Reproducible exports**: Full configuration saved

## Next Steps

- **[User Guide](../user-guide/overview.md)**: Detailed method descriptions
- **[Examples](../examples/vmm-analysis.md)**: Code examples
- **[API Reference](../api/overview.md)**: Complete API
