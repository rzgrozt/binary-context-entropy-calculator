# Analysis Methods API

Statistical and probabilistic analysis of binary sequences.

## Main Entry Point

### analyze_dataset

::: bspe.workbench.analyze_dataset

## Markov Analysis

### MarkovAnalysisRequest

Configuration for Markov analysis:

```python
from bspe import MarkovAnalysisRequest, MarkovPredictionMode, MarkovResultScope

request = MarkovAnalysisRequest(
    smoothing=KTSmoothing(),  # Default
    prediction_mode=MarkovPredictionMode.FIXED_MODEL,
    scope=MarkovResultScope.UNION,  # Default: both per-seq and pooled
)
```

**Fields:**
- `smoothing`: KTSmoothing, MLESmoothing, or AdditiveSmoothing
- `prediction_mode`: FIXED_MODEL or CUMULATIVE_PREFIX
- `scope`: PER_SEQUENCE, POOLED, or UNION

### MarkovResult

Result container with `records: tuple[MarkovRecordAnalysis, ...]`

### MarkovRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
transition_matrix = record.transition_probabilities  # 2x2 matrix
```

### MarkovPredictionMode

Enum for target prediction strategy:
- `FIXED_MODEL`: Use model learned from full sequence (default)
- `CUMULATIVE_PREFIX`: Update model as prediction progresses

### MarkovResultScope

Enum for result aggregation:
- `PER_SEQUENCE`: One result per input sequence
- `POOLED`: Single aggregate result
- `UNION`: Both per-sequence and pooled (default)

## VMM (Variable-Order Markov) Analysis

### VMMAnalysisRequest

Configuration for VMM analysis:

```python
from bspe import VMMAnalysisRequest, VMMConfig, VMMResultScope, KTSmoothing

request = VMMAnalysisRequest(
    smoothing=KTSmoothing(),  # Default
    config=VMMConfig(
        minimum_support=2,
        maximum_depth=4,
    ),
    scope=VMMResultScope.POOLED,  # Default
)
```

**Fields:**
- `smoothing`: Smoothing strategy (KTSmoothing, MLESmoothing, AdditiveSmoothing)
- `config`: VMM configuration (minimum_support, maximum_depth)
- `scope`: PER_SEQUENCE or POOLED

### VMMConfig

Configuration for VMM algorithm:

- `minimum_support` (int): Minimum observations required for a context (default: 2)
- `maximum_depth` (int): Maximum context depth to explore (default: 4)
- `depth_evidence_limit` (float): Threshold for depth evidence (default: 0.1)

### VMMSmoothing

Base class for smoothing strategies. Use one of:

- `KTSmoothing()`: Krichevsky-Trofimov (Bayesian, α=0.5)
- `MLESmoothing()`: Maximum Likelihood (no smoothing, α=0)
- `AdditiveSmoothing(alpha=1.0)`: Laplace-like (α=1.0)

### KTSmoothing

Krichevsky-Trofimov smoothing (default). Uses universal prior with pseudo-count 0.5.

```python
from bspe import KTSmoothing
smoothing = KTSmoothing()
```

### MLESmoothing

Maximum Likelihood Estimation (no smoothing). Zero probability for unseen transitions.

```python
from bspe import MLESmoothing
smoothing = MLESmoothing()
```

### AdditiveSmoothing

Laplace-like smoothing with configurable pseudo-count.

```python
from bspe import AdditiveSmoothing
smoothing = AdditiveSmoothing(alpha=1.0)  # Standard Laplace
smoothing = AdditiveSmoothing(alpha=0.5)  # Custom
```

### VMMResult

Result container with `records: tuple[VMMRecordAnalysis, ...]`

### VMMRecordAnalysis

Result for a single sequence or pooled aggregate. Extends base result with depth evidence:

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
depth_evidence = record.depth_evidence  # Per-position depth metrics
```

### VMMResultScope

Enum for result aggregation:
- `PER_SEQUENCE`: One result per input sequence
- `POOLED`: Single aggregate result (default)

## HMM (Hidden Markov Model) Analysis

### HMMAnalysisRequest

Configuration for HMM analysis:

```python
from bspe import HMMAnalysisRequest

request = HMMAnalysisRequest(
    model=model,  # BinaryHMM instance
    scope=HMMResultScope.PER_SEQUENCE,  # Default
)
```

**Fields:**
- `model`: Configured BinaryHMM
- `scope`: PER_SEQUENCE or POOLED

### HMMResult

Result container with `records: tuple[HMMRecordAnalysis, ...]`

### HMMRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits
posterior_prob = record.state_probabilities  # Hidden state posteriors (per position)
```

### BinaryHMM

Configured two-state, two-observable Hidden Markov Model:

```python
from bspe import BinaryHMM, BinaryLabels

model = BinaryHMM(
    labels=BinaryLabels(states=("Fair", "Biased"), observables=("H", "T")),
    initial=[0.6, 0.4],           # Initial state distribution
    transition=[[0.9, 0.1],        # State transition matrix (2x2)
                [0.2, 0.8]],
    emission=[[0.5, 0.5],          # Emission matrix (2x2)
              [0.7, 0.3]],
)
```

**Raises ProbabilityRangeError, ProbabilitySumError, ProbabilityShapeError if invalid.**

## Shannon Analysis (Descriptive)

### ShannonAnalysisRequest

Configuration for Shannon entropy analysis (no temporal model):

```python
from bspe import ShannonAnalysisRequest

request = ShannonAnalysisRequest(
    scope=ShannonResultScope.UNION,  # Default: both per-seq and pooled
)
```

**Fields:**
- `scope`: PER_SEQUENCE, POOLED, or UNION

### ShannonResult

Result container with `records: tuple[ShannonRecordAnalysis, ...]`

### ShannonRecordAnalysis

Result for a single sequence or pooled aggregate.

```python
record = result.records[0]
entropy_bits = record.predictive_entropy_bits  # Observed Shannon entropy
observable_frequencies = record.observable_frequencies  # (freq_a, freq_b)
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

- [User Guide: Methods](../user-guide/overview.md)
- [Examples: Markov](../examples/markov-comparison.md)
- [Examples: VMM](../examples/vmm-analysis.md)
- [Examples: HMM](../examples/hmm-analysis.md)
- [Glossary](../reference/glossary.md)
