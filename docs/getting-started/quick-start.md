# Quick Start

Get up and running with **bspe** in 5 minutes.

## 1. Parse a Sequence

```python
from bspe import BinaryLabels, parse_manual_batch

# Define observable labels
labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B")
)

# Parse sequences (one per line)
dataset = parse_manual_batch(
    "A, B, B, A, A\nB, A, A, B",
    labels
)

print(f"Parsed {len(dataset.records)} sequences")
```

## 2. Analyze with VMM

```python
from bspe import (
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
)

# Configure VMM
config = VMMConfig(minimum_support=2)

# Analyze
result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=config,
        result_scope=VMMResultScope.POOLED,
    )
)

# Inspect results
for record in result.records:
    print(f"Sequence: {record.sequence}")
    print(f"Predicted: {'A' if record.predicted_target_index == 0 else 'B'}")
    print(f"Entropy: {record.predictive_entropy_bits:.4f} bits")
    print()
```

## 3. Compare Methods

```python
from bspe import (
    MarkovAnalysisRequest,
    ShannonAnalysisRequest,
    compare_methods,
)

# Compare VMM, Markov, and Shannon
comparison = compare_methods(
    dataset,
    [
        VMMAnalysisRequest(
            config=VMMConfig(minimum_support=2),
            result_scope=VMMResultScope.POOLED,
        ),
        MarkovAnalysisRequest(smoothing_alpha=0.5),
        ShannonAnalysisRequest(),
    ]
)

# Results in order
for result in comparison.results:
    print(f"Method: {result.method}")
    print(f"Records: {len(result.records)}")
```

## 4. Stimulus Search

```python
from bspe import (
    StimulusSearchConfig,
    VMMConfig,
    search_stimuli,
    stimulus_candidate_csv,
)

# Configure search
config = StimulusSearchConfig(
    sequence_length=12,
    desired_stimuli=20,
    seed=2026,
    candidate_limit=1000,
    vmm_config=VMMConfig(minimum_support=2),
)

# Run search
result = search_stimuli(config)

print(f"Status: {result.status}")
print(f"Selected: {result.selected_count} / {result.config.desired_stimuli}")

# Export results
csv = stimulus_candidate_csv(result)
with open("candidates.csv", "w") as f:
    f.write(csv)
```

## 5. Use the Streamlit Workbench

```bash
uv run streamlit run streamlit_app.py
```

Then:

1. **Analyzer Mode**: Select methods, upload data, compare results
2. **Stimulus Search Mode**: Configure search, inspect candidates, assign targets

## Next Steps

- **[Concepts](concepts.md)**: Understand VMM, Markov, HMM, Shannon
- **[User Guide](../user-guide/overview.md)**: Detailed method descriptions
- **[Examples](../examples/vmm-analysis.md)**: More code examples
- **[API Reference](../api/overview.md)**: Complete API documentation
