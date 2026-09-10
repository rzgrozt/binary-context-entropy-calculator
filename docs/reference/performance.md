# Performance Guide

## Benchmarks

Typical performance on a modern laptop (Intel i7, 16GB RAM):

### Analysis Methods

| Method | Sequence Length | Dataset Size | Time |
|--------|-----------------|--------------|------|
| VMM | 12 symbols | 10 records | ~100ms |
| VMM | 12 symbols | 100 records | ~1s |
| First-order Markov | 12 symbols | 100 records | ~10ms |
| HMM | 12 symbols | 100 records | ~50ms |
| Shannon | 12 symbols | 100 records | ~5ms |

### Stimulus Search

| Sequence Length | Candidate Limit | Time | Memory |
|-----------------|-----------------|------|--------|
| 8 | 256 | ~50ms | ~5MB |
| 12 | 1000 | ~1s | ~20MB |
| 16 | 5000 | ~10s | ~100MB |
| 20 | 5000 | ~60s | ~200MB |

**Note:** Times are approximate and depend on:
- Minimum support threshold
- Constraint complexity
- System specifications

## Optimization Strategies

### For VMM Analysis

**Increase minimum_support:**
```python
# Slower: Accept all contexts
config = VMMConfig(minimum_support=1)

# Faster: Require at least 3 observations
config = VMMConfig(minimum_support=3)
```

**Effect:** Higher minimum support reduces context count exponentially.

**Use per-sequence analysis:**
```python
# Slower: Pooled (more data per model)
result = analyze_vmm(dataset, config)

# Faster: Per-sequence (smaller models)
result = analyze_vmm_per_sequence(dataset, config)
```

**Effect:** Per-sequence models are smaller but less stable.

### For Stimulus Search

**Reduce sequence_length:**
```python
# Slower: 20-symbol sequences (2^20 = 1M candidates)
config = StimulusSearchConfig(sequence_length=20, ...)

# Faster: 12-symbol sequences (2^12 = 4K candidates)
config = StimulusSearchConfig(sequence_length=12, ...)
```

**Effect:** Exponential reduction in search space.

**Reduce candidate_limit:**
```python
# Slower: Search 5000 candidates
config = StimulusSearchConfig(candidate_limit=5000, ...)

# Faster: Search 1000 candidates
config = StimulusSearchConfig(candidate_limit=1000, ...)
```

**Effect:** Linear reduction in search time.

**Relax constraints:**
```python
# Slower: Tight constraints reject many candidates
constraints = StimulusConstraints(
    predicted_probability=InclusiveRange(0.7, 0.9),
    predictive_entropy=InclusiveRange(0.2, 0.3),
)

# Faster: Loose constraints accept more candidates
constraints = StimulusConstraints(
    predicted_probability=InclusiveRange(0.5, 1.0),
    predictive_entropy=InclusiveRange(0.0, 1.0),
)
```

**Effect:** Fewer candidates rejected, faster filtering.

## Memory Usage

### Analysis Methods

| Method | Typical Memory | Scaling |
|--------|----------------|---------|
| VMM | 1KB per context | O(contexts) |
| Markov | ~1KB | O(1) |
| HMM | ~1KB | O(1) |
| Shannon | ~1KB | O(1) |

**VMM context count:**
- 12-symbol sequence: ~100–1000 contexts
- 20-symbol sequence: ~1000–10000 contexts
- Minimum support=1: Maximum contexts
- Minimum support=3: ~50% fewer contexts

### Stimulus Search

**Batching:** Stimulus search processes candidates in 64-candidate batches to bound memory.

```python
# Memory usage is bounded regardless of candidate_limit
config = StimulusSearchConfig(
    candidate_limit=5000,  # Only 64 in memory at a time
    ...
)
```

**Typical memory:**
- 1000 candidates: ~20MB
- 5000 candidates: ~100MB

## Profiling

### Using Python's cProfile

```python
import cProfile
import pstats
from bspe import analyze_vmm, parse_manual_batch, BinaryLabels, VMMConfig

labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, A, B\n" * 100, labels)

profiler = cProfile.Profile()
profiler.enable()

result = analyze_vmm(dataset, VMMConfig(minimum_support=2))

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(10)
```

### Using timeit

```python
import timeit
from bspe import analyze_vmm, parse_manual_batch, BinaryLabels, VMMConfig

labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, A, B\n" * 100, labels)
config = VMMConfig(minimum_support=2)

time_ms = timeit.timeit(
    lambda: analyze_vmm(dataset, config),
    number=10,
) / 10 * 1000

print(f"Average time: {time_ms:.1f}ms")
```

## Scaling Limits

### Hard Limits

- **Sequence length (stimulus search):** ≤ 32 symbols
- **Candidate limit (stimulus search):** ≤ 5000
- **Dataset size:** No hard limit (tested up to 10K records)
- **Sequence length (analysis):** No hard limit (tested up to 1000 symbols)

### Practical Limits

| Metric | Practical Limit | Reason |
|--------|-----------------|--------|
| VMM contexts | ~100K | Memory and fitting time |
| Dataset records | ~10K | Fitting time |
| Sequence length | ~500 | Context explosion |
| Stimulus search length | ~20 | Exponential search space |

## Recommendations

### For Interactive Use (Streamlit)

- **Sequence length:** 8–16 symbols
- **Dataset size:** 10–100 records
- **Minimum support:** 2–3
- **Candidate limit:** 1000

**Expected response time:** < 1 second

### For Batch Processing

- **Sequence length:** 12–32 symbols
- **Dataset size:** 100–10K records
- **Minimum support:** 1–5
- **Candidate limit:** 5000

**Expected time:** 1–60 seconds

### For Research

- **Sequence length:** 12–20 symbols
- **Dataset size:** 100–1K records
- **Minimum support:** 2–3
- **Candidate limit:** 5000

**Expected time:** 10–120 seconds

## Troubleshooting Performance

### Analysis is slow

**Checklist:**
1. Increase `minimum_support` (reduces contexts)
2. Use per-sequence analysis (smaller models)
3. Reduce dataset size
4. Reduce sequence length

### Stimulus search is slow

**Checklist:**
1. Reduce `sequence_length` (exponential effect)
2. Reduce `candidate_limit`
3. Relax hard constraints
4. Increase `minimum_support` in VMM config

### Memory usage is high

**Checklist:**
1. Reduce `sequence_length`
2. Increase `minimum_support`
3. Reduce dataset size
4. Use per-sequence analysis

## Next Steps

- **[Glossary](glossary.md)**: Quick reference for terms
- **[FAQ](../faq.md)**: Common questions
- **[Troubleshooting](../troubleshooting.md)**: Error solutions
