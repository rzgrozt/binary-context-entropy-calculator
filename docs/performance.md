# Performance Guide

Optimization tips, benchmarks, and scaling considerations for **bspe**.

## Table of Contents

- [Benchmarks](#benchmarks)
- [Optimization Tips](#optimization-tips)
- [Memory Management](#memory-management)
- [Profiling](#profiling)
- [Scaling](#scaling)

---

## Benchmarks

### VMM Analysis

**Setup:** Single sequence, KT smoothing, minimum support 2

| Sequence Length | Contexts | Time (ms) | Memory (MB) |
|-----------------|----------|-----------|------------|
| 10 | ~50 | 2 | 0.1 |
| 50 | ~500 | 10 | 0.5 |
| 100 | ~2000 | 40 | 2 |
| 500 | ~10000 | 200 | 10 |

**Notes:**
- Time scales roughly O(N²) with sequence length
- Memory scales with number of unique contexts
- Minimum support significantly affects context count

### Markov Analysis

**Setup:** Single sequence, MLE smoothing

| Sequence Length | Time (ms) | Memory (MB) |
|-----------------|-----------|------------|
| 10 | 0.5 | 0.01 |
| 100 | 0.5 | 0.01 |
| 1000 | 0.5 | 0.01 |
| 10000 | 0.5 | 0.01 |

**Notes:**
- Time is constant (only counts transitions)
- Memory is constant (fixed 2×2 matrix)
- Markov is ~100x faster than VMM

### HMM Analysis

**Setup:** Single sequence, configured model

| Sequence Length | Time (ms) | Memory (MB) |
|-----------------|-----------|------------|
| 10 | 1 | 0.1 |
| 100 | 10 | 0.1 |
| 1000 | 100 | 0.1 |
| 10000 | 1000 | 0.1 |

**Notes:**
- Time scales linearly O(N) with sequence length
- Memory is constant (fixed model size)
- Filtering is straightforward matrix multiplication

### Stimulus Search

**Setup:** Seeded generation, VMM analysis, no constraints

| Sequence Length | Candidates | Time (s) | Memory (MB) |
|-----------------|------------|----------|------------|
| 8 | 256 | 0.1 | 5 |
| 10 | 1024 | 0.5 | 20 |
| 12 | 4096 | 2 | 80 |
| 14 | 16384 | 8 | 320 |

**Notes:**
- Time scales with 2^length (exponential)
- Memory scales with 2^length (batched in 64-candidate chunks)
- Constraints reduce time (fewer candidates analyzed)

---

## Optimization Tips

### 1. Choose the Right Method

**For speed:**
```python
# ✓ Fast: Markov (constant time)
result = analyze_markov(dataset, smoothing_alpha=0.0)

# ✗ Slow: VMM (quadratic time)
result = analyze_vmm(dataset, VMMConfig(minimum_support=1))
```

**For accuracy:**
```python
# ✓ Accurate: VMM (captures patterns)
result = analyze_vmm(dataset, VMMConfig(minimum_support=2))

# ✗ Less accurate: Markov (only previous symbol)
result = analyze_markov(dataset)
```

### 2. Increase Minimum Support

Higher minimum support reduces context count:

```python
# Slow: Many contexts
config = VMMConfig(minimum_support=1)  # ~10000 contexts

# Fast: Fewer contexts
config = VMMConfig(minimum_support=5)  # ~1000 contexts
```

**Trade-off:** Higher support = fewer contexts = less specific predictions

### 3. Use Per-Sequence Analysis

Per-sequence analysis fits smaller models:

```python
# Slower: One large pooled model
result = analyze_vmm(dataset, VMMConfig())

# Faster: Many small per-sequence models
result = analyze_vmm_per_sequence(dataset, VMMConfig())
```

**Trade-off:** Per-sequence = less data per model = less stable

### 4. Reduce Sequence Length

Stimulus search is exponential in sequence length:

```python
# Slow: 2^20 = 1M candidates
config = StimulusSearchConfig(sequence_length=20, ...)

# Fast: 2^12 = 4K candidates
config = StimulusSearchConfig(sequence_length=12, ...)
```

**Trade-off:** Shorter sequences = less expressive

### 5. Reduce Candidate Limit

Fewer candidates = faster search:

```python
# Slow: Search 5000 candidates
config = StimulusSearchConfig(candidate_limit=5000, ...)

# Fast: Search 1000 candidates
config = StimulusSearchConfig(candidate_limit=1000, ...)
```

**Trade-off:** Smaller limit = fewer options

### 6. Relax Constraints

Fewer constraints = faster filtering:

```python
# Slow: Many constraints
constraints = StimulusConstraints(
    predicted_probability=InclusiveRange(0.6, 0.9),
    predictive_entropy=InclusiveRange(0.3, 0.8),
    effective_depth=InclusiveRange(1, 5),
    context_support=InclusiveRange(3, 100),
    a_count=InclusiveRange(4, 8),
    switches=InclusiveRange(2, 5),
)

# Fast: Few constraints
constraints = StimulusConstraints(
    predicted_symbol=PredictedSymbol.A,
)
```

**Trade-off:** Fewer constraints = less control

### 7. Use Caching

Streamlit caches results automatically:

```python
@st.cache_data
def analyze_dataset_cached(dataset, config):
    return analyze_dataset(dataset, config)

result = analyze_dataset_cached(dataset, config)
```

**Note:** Cache is invalidated if inputs change

---

## Memory Management

### VMM Context Counts

Memory usage depends on unique contexts:

```python
# Estimate contexts
sequence_length = 100
unique_contexts ≈ sequence_length ** 2 / 2  # ~5000

# Each context: ~100 bytes
memory_mb = unique_contexts * 100 / 1_000_000  # ~0.5 MB
```

**Optimization:**
- Increase `minimum_support` to reduce contexts
- Use shorter sequences
- Use per-sequence analysis (smaller models)

### Stimulus Search Batching

Stimulus search processes in 64-candidate batches:

```python
# Automatic batching (no action needed)
for batch in batched(generate_candidate_records(config), 64):
    for candidate in analyze_stimulus_records(batch, config):
        # Process one candidate
```

**Memory usage:**
- ~64 VMM models in memory at once
- ~5-10 MB per batch (typical)
- Total: ~5-10 MB (not cumulative)

### NumPy Array Ownership

All arrays are copied and owned:

```python
# Each result owns its arrays
result1 = analyze_dataset(dataset1, config)
result2 = analyze_dataset(dataset2, config)
# Both results can be kept in memory independently
```

**Note:** Arrays are read-only to prevent accidental mutation

---

## Profiling

### Profile VMM Analysis

```python
import cProfile
import pstats
from io import StringIO

from bspe import VMMAnalysisRequest, VMMConfig, VMMResultScope, analyze_dataset, parse_manual_batch

labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, A, B, A, B" * 100, labels)

pr = cProfile.Profile()
pr.enable()

result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    ),
)

pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())
```

**Output:**
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.001   0.001    0.050   0.050 vmm.py:50(_fit_records)
   10    0.010   0.001    0.040   0.004 vmm.py:100(_analyze_record)
  100    0.005   0.000    0.020   0.000 vmm.py:150(_smoothed_probabilities)
```

### Profile Stimulus Search

```python
import cProfile
import pstats
from io import StringIO

from bspe import StimulusSearchConfig, VMMConfig, search_stimuli

pr = cProfile.Profile()
pr.enable()

result = search_stimuli(
    StimulusSearchConfig(
        sequence_length=12,
        desired_stimuli=20,
        candidate_limit=1000,
        seed=2026,
        vmm_config=VMMConfig(minimum_support=2),
    )
)

pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def analyze_large_dataset():
    dataset = parse_manual_batch("A, B" * 10000, labels)
    result = analyze_vmm(dataset, VMMConfig(minimum_support=2))
    return result

analyze_large_dataset()
```

**Run:**
```bash
python -m memory_profiler script.py
```

---

## Scaling

### Scaling VMM

**Bottleneck:** Context count (O(N²))

**Strategies:**
1. Increase `minimum_support` (reduce contexts)
2. Use per-sequence analysis (smaller models)
3. Split dataset into smaller batches
4. Use Markov as baseline (constant time)

**Example:**
```python
# For large dataset, use per-sequence
if len(dataset.records) > 100:
    result = analyze_vmm_per_sequence(dataset, config)
else:
    result = analyze_vmm(dataset, config)
```

### Scaling Stimulus Search

**Bottleneck:** Exponential in sequence length (2^N)

**Strategies:**
1. Reduce `sequence_length` (exponential effect)
2. Reduce `candidate_limit`
3. Increase `minimum_support` (faster analysis)
4. Relax constraints (faster filtering)

**Example:**
```python
# For long sequences, reduce candidate limit
if config.sequence_length > 14:
    config = replace(config, candidate_limit=500)
```

### Scaling Streamlit

**Bottleneck:** Re-running entire script on interaction

**Strategies:**
1. Use `@st.cache_data` for expensive operations
2. Use `@st.cache_resource` for models
3. Minimize re-renders with `st.session_state`
4. Use `st.spinner()` for long operations

**Example:**
```python
@st.cache_data
def cached_analysis(dataset_hash, config_hash):
    return analyze_dataset(dataset, config)

result = cached_analysis(hash(dataset), hash(config))
```

---

## Benchmarking Your Data

### Template

```python
import time
from bspe import analyze_vmm, VMMConfig, parse_manual_batch

labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch(your_data, labels)

start = time.perf_counter()
result = analyze_vmm(dataset, VMMConfig(minimum_support=2))
elapsed = time.perf_counter() - start

print(f"Time: {elapsed:.3f}s")
print(f"Records: {len(dataset.records)}")
print(f"Contexts: {len(result.records[0].model.context_counts)}")
print(f"Time per record: {elapsed / len(dataset.records) * 1000:.1f}ms")
```

### Interpreting Results

- **Time per record < 10ms**: Good performance
- **Time per record 10-100ms**: Acceptable
- **Time per record > 100ms**: Consider optimization

---

## References

- [Python cProfile](https://docs.python.org/3/library/profile.html)
- [memory_profiler](https://github.com/pythonprofilers/memory_profiler)
- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching)
- [NumPy Performance](https://numpy.org/doc/stable/user/basics.broadcasting.html)
