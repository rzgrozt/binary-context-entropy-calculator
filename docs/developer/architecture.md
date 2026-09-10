# Architecture Guide

This document describes bspe's overall architecture, component organization, and design decisions.

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   User Applications                  │
│          (Streamlit Workbench, Python Scripts)       │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼───┐  ┌────▼────┐  ┌──▼──────┐
    │ Markov │  │   VMM   │  │   HMM   │
    │ Method │  │  Method │  │ Method  │
    └────┬───┘  └────┬────┘  └──┬──────┘
         │           │           │
    ┌────▼───────────▼───────────▼────┐
    │   Workbench (Multi-method Ops)   │
    └────┬──────────────────────────┬──┘
         │                          │
    ┌────▼────────┐         ┌──────▼──────┐
    │  Analysis   │         │   Stimulus  │
    │   Results   │         │   Search    │
    └─────────────┘         └─────────────┘
         │                          │
    ┌────▼──────────────────────────▼────┐
    │      Domain Objects & Validation    │
    │  (Labels, Sequences, HMM, etc)      │
    └─────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   Parsing & Serialization          │
    │   (CSV, TXT, JSON Export)          │
    └────────────────────────────────────┘
```

## Layered Architecture

### Layer 1: Domain (Core)

**Files**: `domain.py`, `records.py`, `errors.py`

Immutable, validated domain objects:

- `BinaryLabels`: Frozen pair of distinct state/observable labels
- `BinarySequence`: Tuple of 0 or 1 (ObservableIndex)
- `SequenceRecord`: Single sequence with optional target
- `SequenceDataset`: Collection of records + shared labels
- `BinaryHMM`: Fully-specified HMM configuration

**Invariants**:
- All objects frozen (no mutation after construction)
- All probabilities validated at construction
- Labels immutable (enforced via dataclass slots)
- Sequences preserve boundaries (first/last have special significance)

**Validation**:
- Probability values in [0, 1]
- Matrix rows sum to 1.0 ± tolerance
- Labels distinct within category
- Sequences use only recognized labels

### Layer 2: Parsing & Serialization

**Files**: `parsing.py`, `batch_parsing.py`, `markov_csv.py`

Bidirectional conversion:

- **Input**: Text → BinarySequence (via label mapping)
- **Batch Input**: CSV/TXT → SequenceRecords
- **Output**: Results → CSV/JSON

**Key Functions**:
- `parse_sequence()`: String → BinarySequence
- `parse_batch_from_csv()`: File → SequenceRecords (with diagnostics)
- `markov_csv_text()`: Results → CSV string

**Error Handling**:
- Returns (valid_records, issues) tuple
- Recoverable per-record errors reported separately
- File-level errors raised as exceptions

### Layer 3: Analysis Methods

**Files**: `methods/*.py` (markov, vmm, hmm, shannon)

Independent analysis implementations:

#### Markov (`methods/markov.py`)

First-order Markov model learned from data:

- **Input**: SequenceDataset + MarkovAnalysisRequest
- **Output**: MarkovResult (per-sequence and/or pooled)
- **Scope**: PER_SEQUENCE, POOLED, UNION
- **Modes**: FIXED_MODEL, CUMULATIVE_PREFIX (affects target prediction)

#### VMM (`methods/vmm.py`)

Variable-order Markov with adaptive depth:

- **Input**: SequenceDataset + VMMAnalysisRequest + VMMConfig + VMMSmoothing
- **Output**: VMMResult (with depth evidence per position)
- **Scope**: PER_SEQUENCE, POOLED
- **Config**: minimum_support, maximum_depth, backoff strategy
- **Smoothing**: KT (default), MLE, or Additive(α)

#### HMM (`methods/hmm.py`)

Configured (not fitted) Hidden Markov Model:

- **Input**: SequenceDataset + HMMAnalysisRequest + BinaryHMM
- **Output**: HMMResult (with state posteriors)
- **Scope**: PER_SEQUENCE, POOLED
- **Algorithm**: Forward algorithm for filtering

#### Shannon (`methods/shannon.py`)

Descriptive Shannon entropy (no temporal model):

- **Input**: SequenceDataset + ShannonAnalysisRequest
- **Output**: ShannonResult
- **Scope**: PER_SEQUENCE, POOLED, UNION
- **Metric**: Frequency-based, not predictive

### Layer 4: Orchestration

**Files**: `workbench.py`

Multi-method coordination:

- `analyze_dataset()`: Single polymorphic entry point
- Dispatches to appropriate method based on request type
- Returns method-specific result

**Responsibility**: Route only; methods are independent

### Layer 5: Stimulus Search

**Files**: `stimulus_search.py`, `stimulus_*.py`

Synthetic sequence generation and optimization:

- `search_stimulus()`: Generate candidates satisfying constraints
- Candidates ranked by preference score
- Independent from analysis methods

**Constraint Types**:
- Length bounds
- Entropy bounds
- Observable frequency
- Transition frequency
- Boundary values

**Preference Types**:
- Target entropy
- Required/forbidden transitions
- Observable preference

## Key Design Patterns

### 1. Frozen Domain Objects

All domain classes use `@dataclass(frozen=True, slots=True)`:

```python
@dataclass(frozen=True, slots=True)
class BinaryLabels:
    states: LabelPair
    observables: LabelPair
    # Cannot be modified after creation
```

**Benefits**:
- Thread-safe
- Hashable (cacheable)
- Prevents accidental mutation
- Reduced memory (slots)

**Trade-off**: Slight overhead at construction

### 2. Validation at Construction

Configuration errors caught immediately:

```python
model = BinaryHMM(...)  # Raises ProbabilityRangeError immediately
# Not deferred to analysis time
```

**Benefits**:
- Fail fast
- Clear error messages
- No silent bugs

### 3. Request-Response Pattern

Analysis via immutable requests:

```python
request = MarkovAnalysisRequest(scope=MarkovResultScope.POOLED)
result = analyze_dataset(dataset, request)
```

**Benefits**:
- Configuration explicit and immutable
- Easy to log/audit
- Supports polymorphism via overload

### 4. Per-Sequence vs. Pooled

Two aggregation strategies:

| Mode | Use Case | Complexity |
|------|----------|-----------|
| **PER_SEQUENCE** | Individual diagnostics | O(N × n) |
| **POOLED** | Aggregate statistics | O(1) space per sequence |
| **UNION** | Both (where applicable) | O(N × n) |

Selection at request time (not analysis time).

### 5. Scope-Aware Results

Result type varies by analysis method and scope:

- Markov: Returns per-sequence records + pooled record
- VMM: Returns per-sequence records (or pooled only)
- HMM: Returns per-sequence records (or pooled only)

Clients access via `result.records` (uniform interface).

### 6. Error Stratification

Three error categories:

| Category | Timing | Recovery |
|----------|--------|----------|
| **Configuration** | Construction | Retry with valid config |
| **Parsing** | Batch parse | Skip invalid records |
| **Numerical** | Analysis | Check data/smoothing |

Parsed errors returned separately (don't interrupt batch).

### 7. Immutable Sequences

Binary sequences are tuples (immutable):

```python
sequence: BinarySequence = (0, 1, 0, 1)  # Hashable
sequence[0] = 1  # TypeError: tuples are immutable
```

**Benefits**:
- Cache-friendly (hashable)
- Prevents accidental modification
- Clear semantics

## Module Dependencies

### Dependency Graph (simplified)

```
domain.py
  ↓
records.py
  ↓
parsing.py ← batch_parsing.py
  ↓
methods/markov.py ┐
methods/vmm.py    ├→ workbench.py
methods/hmm.py    │
methods/shannon.py┘
  ↓
stimulus_search.py (independent)
  ↓
markov_csv.py (export)
```

### Acyclic Dependency Flow

- Core layers don't depend on higher layers
- Methods are independent (no cross-method calls)
- Workbench is thin (dispatcher only)
- Stimulus search isolated

**Benefit**: Easy to understand and maintain; supports parallel development.

## Type System

### Use of Type Hints

All public APIs have complete type hints:

```python
def analyze_dataset(
    dataset: SequenceDataset,
    request: WorkbenchRequest,
) -> WorkbenchResult:
    """..."""
```

**Tools**:
- **basedpyright**: Full strict type checking
- **Runtime**: Type hints available via `__annotations__`

### Generic Types

Limited use of generics; mostly concrete:

```python
# Concrete
records: tuple[SequenceRecord, ...]

# Rare generic (usually unnecessary for binary system)
```

**Rationale**: Binary system is simple; unnecessary generics reduce clarity.

## Error Handling Strategy

### Three-Tier Error Model

1. **Configuration Errors** (immediate)
   - Invalid probabilities
   - Mismatched matrix shapes
   - Duplicate labels
   - Raised at construction

2. **Parsing Errors** (recoverable)
   - Invalid tokens
   - Malformed CSV
   - Returned in (records, issues) tuple

3. **Analytical Errors** (rare)
   - Zero likelihood
   - Numerical instability
   - Raised during analysis

### Exception Hierarchy

```
BinaryEntropyError (base)
├── Configuration errors (Probability*, Label*, Invalid*)
├── Parsing errors (Invalid*, Batch*)
├── Numerical errors (Zero*, Numerical*)
└── Search errors (InvalidStimulusSearch*, InvalidVMM*)
```

All inherit from `ValueError` for compatibility.

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Parse sequence | O(n) | n = sequence length |
| Markov analysis | O(N × n) | N = # sequences |
| VMM analysis | O(N × n × d²) | d = depth |
| HMM forward | O(N × n) | Constant per position |
| Shannon | O(N × n) | Frequency only |
| Stimulus search | O(C × iterations × ?) | C = candidates |

### Space Optimization

- **Sequences**: Tuples (compact, hashable)
- **Results**: Per-position only if needed
- **Pooled**: Aggregate statistics (O(1) space)
- **Frozen objects**: Slots reduce overhead by ~30%

## Extensibility Points

### Adding a New Analysis Method

1. Create `methods/newmethod.py`
2. Define `NewMethodAnalysisRequest` class
3. Implement `analyze_newmethod()` function
4. Define `NewMethodResult`, `NewMethodRecordAnalysis` classes
5. Add overload to `analyze_dataset()` in workbench
6. Export from `__init__.py`

### Adding Analysis Options

1. Add field to request dataclass
2. Update method implementation
3. Add field to result classes
4. Update tests and documentation

### Adding Smoothing Strategies

1. Extend `VMMSmoothing` protocol/ABC
2. Implement required methods
3. Add to `VMMConfig` validation
4. Update documentation

## Testing Strategy

### Test Organization

- **Unit tests** (`tests/unit/`): Scientific correctness, isolated components
- **UI tests** (`tests/ui/`): State management, form validation
- **Integration tests** (`tests/integration/`): End-to-end workflows (Streamlit)

### Coverage Targets

- **Line coverage**: 95%+
- **Branch coverage**: ~80%+
- **Edge cases**: Explicit (zero probability, empty dataset, etc.)

## Documentation

### Docstring Requirements

- **Module**: Describe purpose, key classes
- **Class**: Immutability, invariants, usage
- **Function**: Args, returns, raises (Google style)
- **Complex logic**: Inline comments explaining "why"

### User Documentation

- **Guides**: Conceptual ("what is entropy?", "when to use VMM?")
- **Tutorials**: Step-by-step examples
- **API reference**: Auto-generated from docstrings

## Future Optimization Opportunities

1. **Parallelization**: Per-sequence analysis across cores
2. **Caching**: Intermediate VMM context results
3. **Approximations**: Sampling-based for very large datasets
4. **GPU acceleration**: Limited benefit for 2×2 matrices
5. **Alternative backends**: NumPy → Polars, JAX

## See Also

- [Contributing Guide](contributing.md)
- [Testing](testing.md)
- [API Reference](../api/overview.md)
