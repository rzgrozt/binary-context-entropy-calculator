# Variable-Order Markov (VMM)

Variable-Order Markov (VMM) analysis predicts the next symbol based on a **context-aware suffix** that automatically backs off to shorter suffixes when needed.

## Overview

### What VMM Does

1. **Fits a model** from observed sequences using context counts
2. **Selects contexts** of varying length based on minimum support threshold
3. **Backs off** to shorter suffixes when a context has insufficient observations
4. **Predicts** the next symbol using the fitted distribution

### When to Use VMM

- **Detect recurrent patterns**: VMM finds multi-symbol dependencies
- **Exploratory analysis**: Understand what contexts matter
- **Primary analysis**: Usually more accurate than first-order Markov
- **Comparison**: Use as baseline against other methods

## Configuration

### VMMConfig

```python
from bspe import VMMConfig, KTSmoothing, MLESmoothing, AdditiveSmoothing

# Default: KT smoothing, minimum support 1
config = VMMConfig()

# Custom minimum support
config = VMMConfig(minimum_support=2)

# Custom smoothing: MLE (no pseudocount)
config = VMMConfig(smoothing=MLESmoothing())

# Custom smoothing: Additive with alpha=0.5
config = VMMConfig(smoothing=AdditiveSmoothing(alpha=0.5))
```

### Smoothing Methods

| Method | Alpha | Formula | Use When |
|--------|-------|---------|----------|
| **KT** (default) | 0.5 | `(count + 0.5) / (support + 1)` | Default choice |
| **MLE** | 0 | `count / support` | Want no pseudocount |
| **Additive** | Custom | `(count + α) / (support + 2α)` | Fine-tuning |

**Recommendation:** Start with KT (default), then experiment with others if needed.

### Minimum Support

**Minimum support** is the threshold for accepting a context:

```python
# Accept all contexts (may be noisy)
config = VMMConfig(minimum_support=1)

# Require at least 2 observations (recommended)
config = VMMConfig(minimum_support=2)

# Require at least 3+ observations (most conservative)
config = VMMConfig(minimum_support=3)
```

**Effect:** Higher minimum support → fewer contexts → more stable → less specific predictions

## Analysis Modes

### Pooled Analysis

One model shared across all records:

```python
from bspe import VMMAnalysisRequest, VMMResultScope, analyze_dataset

result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    )
)

# One fitted model, applied to all records
print(f"Records analyzed: {len(result.records)}")
print(f"Config: {result.config}")
```

**Pros:**
- More data per model
- Stable, confident estimates
- Good for similar sequences

**Cons:**
- Assumes all sequences come from same distribution
- May miss individual patterns

### Per-Sequence Analysis

Independent model for each record:

```python
from bspe import VMMResultScope

result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.PER_SEQUENCE,
    )
)

# Each record has independent model
for record in result.records:
    print(f"{record.sequence_id}: depth={record.effective_context_depth}")
```

**Pros:**
- Captures individual patterns
- No pooling assumptions
- Fair comparison across different sequences

**Cons:**
- Less data per model
- Less stable estimates
- May be noisy with short sequences

## Results

### Per-Record Analysis

```python
result = analyze_dataset(dataset, request)

for record in result.records:
    # Identifiers
    print(record.sequence_id)
    print(record.sequence)
    
    # Context information
    print(record.effective_context_depth)  # Depth used (or None)
    print(record.context_used)              # Actual context suffix
    print(record.support_count)             # Observations of context
    
    # Predictions
    print(record.probability_a)
    print(record.probability_b)
    print(record.predicted_target_index)   # 0, 1, or None (tie)
    
    # Uncertainty
    print(record.predictive_entropy_bits)
    
    # Target assessment (if provided)
    if record.target_assessment:
        print(record.target_assessment.surprisal_bits)
    
    # Depth evidence table
    for evidence in record.depth_evidence:
        print(evidence.depth, evidence.context, evidence.support_count)
```

### Depth Evidence Table

The **depth evidence table** shows all examined suffixes and backoff decisions:

```python
for evidence in record.depth_evidence:
    print(f"Depth: {evidence.depth}")
    print(f"Context: {evidence.context}")
    print(f"Support: {evidence.support_count}")
    print(f"Entropy: {evidence.entropy_bits}")
    print(f"Used: {evidence.used}")
```

**Interpretation:**
- **Depth 0**: No context (marginal distribution)
- **Depth 1**: Previous symbol
- **Depth 2+**: Longer suffixes
- **Used**: Whether this level was selected for prediction

## Example

### Basic Analysis

```python
from bspe import (
    BinaryLabels,
    VMMConfig,
    VMMAnalysisRequest,
    VMMResultScope,
    analyze_dataset,
    parse_manual_batch,
)

# Setup
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B, A, A, B, A, A", labels)

# Analyze
result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    )
)

# Results
record = result.records[0]
print(f"Sequence: {record.sequence_str}")
print(f"Predicted: {'A' if record.predicted_target_index == 0 else 'B'}")
print(f"P(A) = {record.probability_a:.4f}")
print(f"P(B) = {record.probability_b:.4f}")
print(f"Entropy: {record.predictive_entropy_bits:.4f} bits")
```

### Inspecting Backoff

```python
# See which context depths were examined
for evidence in record.depth_evidence:
    status = "✓" if evidence.used else "✗"
    print(
        f"{status} Depth {evidence.depth}: "
        f"context={evidence.context}, "
        f"support={evidence.support_count}"
    )
```

### Comparing Configurations

```python
# Conservative (higher minimum support)
result_conservative = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=3),
        result_scope=VMMResultScope.POOLED,
    )
)

# Exploratory (lower minimum support)
result_exploratory = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=1),
        result_scope=VMMResultScope.POOLED,
    )
)

# Compare predictions
record_c = result_conservative.records[0]
record_e = result_exploratory.records[0]

print(f"Conservative: depth={record_c.effective_context_depth}, P(A)={record_c.probability_a:.4f}")
print(f"Exploratory: depth={record_e.effective_context_depth}, P(A)={record_e.probability_a:.4f}")
```

## Key Concepts

### Automatic Suffix Backoff

VMM automatically backs off to shorter suffixes:

```
Full context: "AAA" (support=1, too low)
    ↓ Back off
Shorter: "AA" (support=3, acceptable)
    ↓ Use this
Prediction: P(next | AA)
```

### Effective Context Depth

The actual depth (suffix length) used for prediction:

```python
record.effective_context_depth  # int or None
# 0 = no context (marginal)
# 1 = previous symbol
# 2 = previous 2 symbols
# None = no valid context found
```

### Context Support

Number of times the selected context was observed:

```python
record.support_count  # How many times was context_used observed?
```

Higher support → more confident predictions.

## Performance Considerations

### Speed

| Configuration | Time |
|---------------|------|
| minimum_support=1 | ~100ms (100 records) |
| minimum_support=2 | ~50ms (100 records) |
| minimum_support=3 | ~30ms (100 records) |

**Trade-off:** Higher minimum support is faster (fewer contexts).

### Memory

- Each unique context: ~100 bytes
- 100-symbol sequence: ~1000 contexts (typical)
- Per-sequence: ~100KB per record

### Optimization

```python
# Faster: Higher minimum support
config = VMMConfig(minimum_support=3)

# Faster: Per-sequence mode (smaller models)
result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=config,
        result_scope=VMMResultScope.PER_SEQUENCE,
    )
)
```

## Troubleshooting

### Prediction is just marginal distribution (depth=0)

**Cause:** No context has sufficient support (minimum_support too high)

**Solution:** Lower minimum_support or use per-sequence analysis

```python
# Try lower threshold
config = VMMConfig(minimum_support=1)
```

### Entropy is very low (near 0)

**Cause:** Model is very confident (either accurate or overfitting)

**Solution:** Check depth evidence table to see which contexts are used

### Results differ between pooled and per-sequence

**Cause:** This is expected! Different models can give different predictions

**Solution:** Use comparison to understand which method fits better

## See Also

- **[First-order Markov](markov.md)**: Simpler baseline
- **[Input Formats](../input-formats.md)**: How to prepare data
- **[Stimulus Search](../stimulus-search.md)**: Generate sequences with specific VMM properties
- **[Examples](../../examples/vmm-analysis.md)**: Runnable examples
