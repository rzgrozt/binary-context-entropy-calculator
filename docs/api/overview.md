# API Reference Overview

This section provides comprehensive documentation of bspe's public API, automatically extracted from source code.

## API Organization

### Core Domain Objects

- **[Records & Parsing](records-parsing.md)** — Data structures for sequences, datasets, and labels
  - `BinaryLabels`: Immutable label pairs for states and observables
  - `BinarySequence`: Sequence of observations
  - `SequenceRecord`: Single sequence with optional target
  - `SequenceDataset`: Collection of records with shared labels
  - Parsing functions for TXT and CSV formats

### Analysis Methods

- **[Analysis Methods](analysis-methods.md)** — Statistical and probabilistic analysis functions
  - Markov analysis
  - Variable-Order Markov (VMM)
  - Hidden Markov Model (HMM)
  - Shannon entropy
  - All result types and configuration classes

### Stimulus Search

- **[Stimulus Search](stimulus-search.md)** — Synthetic sequence generation and optimization
  - `search_stimulus()`: Main entry point
  - Constraints and preferences
  - Candidate ranking and export

### Serialization

- **[Serialization](serialization.md)** — Export and persistence
  - CSV export functions
  - JSON export for stimulus results
  - Batch parsing from files

### Error Handling

- **[Exceptions](exceptions.md)** — All error types and diagnostics
  - Configuration errors
  - Parsing errors
  - Numerical errors
  - Dataset validation errors

## Common Workflows

### Quick Analysis

```python
from bspe import analyze_dataset, SequenceDataset, SequenceRecord
from bspe import BinaryLabels, MarkovAnalysisRequest

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))
records = [SequenceRecord("id", "01010", labels=labels)]
dataset = SequenceDataset(labels, records)
result = analyze_dataset(dataset, MarkovAnalysisRequest())
```

### Multi-Method Workbench

```python
from bspe import workbench

result = workbench.analyze_dataset(dataset, analysis_request)
```

### Stimulus Generation

```python
from bspe import search_stimulus, StimulusConstraints, StimulusPreferences

result = search_stimulus(
    labels=labels,
    constraints=StimulusConstraints(minimum_length=20),
    preferences=StimulusPreferences(target_entropy_bits=0.5),
)
```

## Module Map

```
bspe/
├── __init__.py           ← Public API exports
├── domain.py             ← Core domain objects (labels, HMM, sequences)
├── records.py            ← SequenceRecord, SequenceDataset
├── parsing.py            ← Sequence parsing (text → binary)
├── batch_parsing.py      ← CSV/TXT batch parsing
├── methods/
│   ├── markov.py         ← First-order Markov analysis
│   ├── vmm.py            ← Variable-order Markov analysis
│   ├── hmm.py            ← Hidden Markov Model analysis
│   └── shannon.py        ← Shannon entropy (descriptive)
├── workbench.py          ← Multi-method orchestration
├── stimulus_search.py    ← Stimulus generation and search
├── stimulus_*.py         ← Stimulus constraints, preferences, results
├── errors.py             ← All exception types
└── markov_csv.py         ← CSV export
```

## Naming Conventions

### Classes

- **Domain Objects**: `Binary*` (e.g., `BinaryLabels`, `BinaryHMM`)
- **Configuration**: `*Request`, `*Config` (e.g., `MarkovAnalysisRequest`, `VMMConfig`)
- **Results**: `*Result`, `*Record` (e.g., `MarkovResult`, `MarkovRecordAnalysis`)
- **Errors**: `*Error` (e.g., `ProbabilityRangeError`)

### Functions

- **Analysis**: `analyze_*` (e.g., `analyze_dataset`, `analyze_markov`)
- **Parsing**: `parse_*` (e.g., `parse_sequence`, `parse_batch_csv`)
- **Search**: `search_*`, `generate_*` (e.g., `search_stimulus`)

### Parameters

- Observable/state indices: `*_index` (0 or 1)
- Probability values: `*_probability` or `*_prob`
- Entropy values: `*_entropy_bits` or `entropy_bits`
- Scope/mode selectors: `*_scope`, `*_mode`

## Type Hints

All public APIs use full type hints. Common types:

```python
FloatArray = npt.NDArray[np.floating[Any]]  # NumPy float array
ObservableIndex = Literal[0, 1]             # Binary observable (0 or 1)
BinarySequence = tuple[ObservableIndex, ...] # Immutable sequence
ProbabilityVector = tuple[float, float]     # (p0, p1) summing to 1.0
```

## Documentation Style

- **Docstrings**: Google style
- **Type hints**: Present on all public functions/methods
- **Examples**: Integrated in method docstrings where helpful
- **Errors**: Documented in Raises section

## Accessing the API

### Public Exports from `bspe`

All public classes and functions are exported from the top-level `bspe` module:

```python
from bspe import (
    # Domain
    BinaryLabels,
    BinaryHMM,
    SequenceRecord,
    SequenceDataset,
    # Analysis
    analyze_dataset,
    MarkovAnalysisRequest,
    VMMAnalysisRequest,
    HMMAnalysisRequest,
    ShannonAnalysisRequest,
    # Stimulus search
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)
```

For full list, see `src/bspe/__init__.py`.

## Version Compatibility

- **Python**: 3.13+
- **NumPy**: 1.20+
- **Dataclasses**: Built-in (Python 3.13)

## See Also

- [Getting Started](../getting-started/overview.md)
- [Concepts](../getting-started/concepts.md)
- [Examples](../examples/vmm-analysis.md)
- [Troubleshooting](../reference/troubleshooting.md)
