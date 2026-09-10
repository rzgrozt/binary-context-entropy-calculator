# Analysis Methods API

Statistical and probabilistic analysis of binary sequences.

## Main Entry Point

### analyze_dataset

::: bspe.workbench.analyze_dataset

## Markov Analysis

### MarkovAnalysisRequest

::: bspe.methods.markov.MarkovAnalysisRequest

### MarkovResult

::: bspe.methods.markov.MarkovResult

### MarkovRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
transition_matrix = record.transition_probabilities  # 2x2 matrix
```

### MarkovPredictionMode

::: bspe.markov_types.MarkovPredictionMode

### MarkovResultScope

::: bspe.markov_types.MarkovResultScope

## VMM (Variable-Order Markov) Analysis

### VMMAnalysisRequest

::: bspe.methods.vmm.VMMAnalysisRequest

### VMMConfig

::: bspe.vmm_types.VMMConfig

### VMMSmoothing

::: bspe.vmm_types.VMMSmoothing

(Abstract base; use `KTSmoothing`, `MLESmoothing`, or `AdditiveSmoothing`)

### KTSmoothing

::: bspe.vmm_types.KTSmoothing

### MLESmoothing

::: bspe.vmm_types.MLESmoothing

### AdditiveSmoothing

::: bspe.vmm_types.AdditiveSmoothing

### VMMResult

::: bspe.methods.vmm.VMMResult

### VMMRecordAnalysis

Result for a single sequence or pooled aggregate. Extends base result with depth evidence:

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
depth_evidence = record.depth_evidence  # Per-position depth metrics
```

### VMMResultScope

::: bspe.vmm_types.VMMResultScope

## HMM (Hidden Markov Model) Analysis

### HMMAnalysisRequest

::: bspe.methods.hmm.HMMAnalysisRequest

### HMMResult

::: bspe.methods.hmm.HMMResult

### HMMRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
posterior_prob = record.state_probabilities  # Hidden state posteriors
```

### BinaryHMM

::: bspe.domain.BinaryHMM

## Shannon Analysis (Descriptive)

### ShannonAnalysisRequest

::: bspe.methods.shannon.ShannonAnalysisRequest

### ShannonResult

::: bspe.methods.shannon.ShannonResult

### ShannonRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits  # Observed Shannon entropy
observable_frequencies = record.observable_frequencies
```

## Smoothing Classes

All smoothing classes are in `bspe.vmm_types`:

- `KTSmoothing`: Bayesian Krichevsky-Trofimov (α=0.5 pseudo-count)
- `MLESmoothing`: No smoothing; maximum likelihood only
- `AdditiveSmoothing(alpha: float)`: Laplace-like smoothing

### Usage

```python
from bspe import VMMAnalysisRequest, VMMConfig, KTSmoothing

# Default (KT smoothing)
request = VMMAnalysisRequest()

# Explicit smoothing
request = VMMAnalysisRequest(
    smoothing=KTSmoothing(),
    config=VMMConfig(minimum_support=2),
)
```

## Result Scope

Analysis can return results at different aggregation levels:

- **PER_SEQUENCE**: One result per input sequence
- **POOLED**: Single aggregate result across all sequences
- **UNION**: Both per-sequence and pooled (where applicable)

### Selection by Method

| Method | Scopes |
|--------|--------|
| Markov | PER_SEQUENCE, POOLED, UNION |
| VMM | PER_SEQUENCE, POOLED |
| HMM | PER_SEQUENCE, POOLED |
| Shannon | PER_SEQUENCE, POOLED, UNION |

### Default Scope

If not specified, defaults vary by method:

```python
# Markov: defaults to UNION (both per-sequence and pooled)
markov_result = analyze_dataset(dataset, MarkovAnalysisRequest())

# VMM: defaults to POOLED
vmm_result = analyze_dataset(dataset, VMMAnalysisRequest())

# HMM: defaults to PER_SEQUENCE
hmm_result = analyze_dataset(dataset, HMMAnalysisRequest(model=model))
```

## Common Result Fields

All analysis results share base fields:

```python
record = result.records[0]

# Predictive entropy (bits)
entropy_bits = record.predictive_entropy_bits

# Next observable probabilities
prob_a = record.probability_observable_a
prob_b = 1.0 - prob_a

# Per-position analysis (if available)
for pos_result in record.per_position_results:
    position_entropy = pos_result.entropy_bits
    position_prob_a = pos_result.probability_observable_a
```

## Probability Utilities

### binary_entropy

::: bspe.information.binary_entropy

Calculate Shannon entropy of a Bernoulli distribution:

```python
from bspe import binary_entropy

h_fair_coin = binary_entropy(0.5)  # → 1.0 bits
h_biased_coin = binary_entropy(0.8)  # → 0.722 bits
h_certain = binary_entropy(1.0)  # → 0.0 bits
```

## Error Handling

Analysis operations raise domain-specific exceptions:

```python
from bspe import analyze_dataset, NumericalInvariantError

try:
    result = analyze_dataset(dataset, request)
except NumericalInvariantError as e:
    print(f"Numerical error: {e}")
```

See [Exceptions](exceptions.md) for full error reference.

## See Also

- [User Guide: Methods](../user-guide/methods/overview.md)
- [Examples: Markov](../examples/markov-comparison.md)
- [Examples: VMM](../examples/vmm-analysis.md)
- [Examples: HMM](../examples/hmm-analysis.md)
- [Glossary](../reference/glossary.md)
