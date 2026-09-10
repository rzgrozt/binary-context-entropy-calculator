# Exceptions API

Error types and diagnostics.

## Exception Hierarchy

```
ValueError
└── BinaryEntropyError (base exception)
    ├── ProbabilityRangeError
    ├── ProbabilitySumError
    ├── ProbabilityShapeError
    ├── InvalidLabelError
    ├── DuplicateLabelError
    ├── InvalidSequenceTokenError
    ├── NumericalInvariantError
    ├── PresetDecodeError
    ├── PresetSchemaError
    ├── ZeroLikelihoodError
    ├── DatasetValidationError
    ├── BatchRecordError
    ├── BatchParseError
    ├── InvalidStimulusSearchConfigurationError
    └── InvalidVMMConfigurationError
```

## Base Exception

### BinaryEntropyError

Base exception for all bspe errors. Inherits from `ValueError`:

```python
from bspe import BinaryEntropyError

try:
    # ... bspe operation ...
except BinaryEntropyError as e:
    print(f"bspe error: {e}")
```

All domain-specific errors inherit from this.

## Configuration Errors

Raised when domain parameters are invalid.

### ProbabilityRangeError

Probability value outside [0, 1].

**Example:**
```python
# Raises ProbabilityRangeError
model = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[1.2, -0.2], [0.4, 0.6]],  # 1.2 is invalid
    emission=[[0.5, 0.5], [0.5, 0.5]],
)
```

### ProbabilitySumError

Probability vector/matrix row doesn't sum to 1.0 (within tolerance).

**Example:**
```python
# Raises ProbabilitySumError
model = BinaryHMM(
    labels=labels,
    initial=[0.4, 0.4],  # Sums to 0.8, not 1.0
    transition=[[0.7, 0.3], [0.6, 0.4]],
    emission=[[0.5, 0.5], [0.5, 0.5]],
)
```

### ProbabilityShapeError

Matrix dimensions incorrect (must be 2×2).

**Example:**
```python
# Raises ProbabilityShapeError
model = BinaryHMM(
    labels=labels,
    initial=[0.5, 0.5],
    transition=[[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],  # 3 rows
    emission=[[0.5, 0.5], [0.5, 0.5]],
)
```

## Label Errors

Raised when label definitions are invalid.

### InvalidLabelError

Observable/state label not recognized during parsing.

**Example:**
```python
labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))
sequence = parse_sequence("0 1 2 0", labels)  # "2" not in labels
# Raises InvalidLabelError
```

### DuplicateLabelError

Duplicate labels in BinaryLabels definition.

**Example:**
```python
# Raises DuplicateLabelError
labels = BinaryLabels(
    states=("A", "A"),  # Duplicate
    observables=("0", "1"),
)
```

## Parsing Errors

Raised during sequence/batch parsing.

### InvalidSequenceTokenError

Invalid token in sequence text (not recognized in labels).

**Example:**
```python
labels = BinaryLabels(states=("A", "B"), observables=("H", "T"))
sequence = parse_sequence("H T X H", labels, separator=" ")
# Raises InvalidSequenceTokenError for "X"
```

### BatchRecordError

Collection of record-level issues during batch parsing.

**Example:**
```python
records, issues = parse_batch_from_csv("data.csv", columns, labels)
if issues:
    raise BatchRecordError(tuple(issues))
```

### BatchParseError

File-level parse error (missing columns, file not found, encoding).

**Example:**
```python
try:
    records, issues = parse_batch_from_csv("missing.csv", columns, labels)
except BatchParseError as e:
    print(f"Parse error: {e}")
```

## Dataset Errors

Raised during dataset construction.

### DatasetValidationError

Dataset constraint violated (duplicate IDs, mismatched labels, empty).

**Example:**
```python
records = [
    SequenceRecord("seq1", "010", labels),
    SequenceRecord("seq1", "101", labels),  # Duplicate ID
]
dataset = SequenceDataset(labels, records)
# Raises DatasetValidationError
```

## Numerical Errors

Raised during analysis when numerical invariants fail.

### ZeroLikelihoodError

Observable has zero probability in given context (unseen transition).

**Example:**
```python
# Sequence never shows A→B transition
sequence = "AAAAABBB"
# Markov model will have P(B|A) = 0
# If analyzing with this model, ZeroLikelihoodError may be raised
```

### NumericalInvariantError

Entropy or probability estimate failed numerical checks.

**Example:**
```python
# Raised if entropy calculation produces NaN or Inf
try:
    result = analyze_dataset(dataset, request)
except NumericalInvariantError as e:
    print(f"Numerical error: {e}")
```

## Stimulus Search Errors

Raised during stimulus generation.

### InvalidStimulusSearchConfigurationError

Constraints are contradictory or infeasible (covered in [Stimulus Search API](stimulus-search.md#constraint-validation-errors)).

**Example:**
```python
# Raises InvalidStimulusSearchConfigurationError
constraints = StimulusConstraints(
    entropy_bits_minimum=0.95,
    entropy_bits_maximum=0.1,  # Contradicts minimum
)
result = search_stimulus(labels, constraints, preferences)
```

## VMM Configuration Errors

Raised during VMM setup.

### InvalidVMMConfigurationError

::: bspe.vmm_types.InvalidVMMConfigurationError

VMM parameter out of valid range.

**Example:**
```python
# Raises InvalidVMMConfigurationError
config = VMMConfig(
    minimum_support=-1,  # Must be non-negative
)
```

## Error Handling Best Practices

### Catch Specific Errors

```python
from bspe import (
    parse_sequence,
    InvalidSequenceTokenError,
    InvalidLabelError,
)

try:
    sequence = parse_sequence(text, labels, separator=" ")
except InvalidSequenceTokenError as e:
    print(f"Invalid token: {e.token}")
except InvalidLabelError as e:
    print(f"Label error in {e.category}: {e}")
```

### Inspect Batch Issues

```python
from bspe import parse_batch_from_csv, BatchRecordError

records, issues = parse_batch_from_csv("data.csv", columns, labels)

if issues:
    print(f"Parsing issues ({len(issues)} total):")
    for issue in issues[:5]:  # Show first 5
        print(f"  Row {issue.record_index}: {issue.code} - {issue.detail}")
    
    if len(issues) > 5:
        print(f"  ... and {len(issues) - 5} more")

# Proceed with valid records only
valid_dataset = SequenceDataset(labels, records)
```

### Validate Configuration Before Analysis

```python
from bspe import (
    BinaryHMM,
    ProbabilitySumError,
    ProbabilityRangeError,
)

try:
    model = BinaryHMM(
        labels=labels,
        initial=initial,
        transition=transition,
        emission=emission,
    )
except (ProbabilitySumError, ProbabilityRangeError) as e:
    print(f"Invalid model configuration: {e}")
    print(f"Check field: {e.field}")
```

### Handle Analysis Failures

```python
from bspe import analyze_dataset, NumericalInvariantError, ZeroLikelihoodError

try:
    result = analyze_dataset(dataset, request)
except ZeroLikelihoodError as e:
    print(f"Unseen observable {e.observable_index} in context")
    # Increase smoothing or check data
except NumericalInvariantError as e:
    print(f"Numerical failure: {e.quantity}")
    # Check for degenerate data or invalid parameters
```

## Exception String Formatting

All exceptions provide detailed string representations with context:

```python
try:
    model = BinaryHMM(...)  # Invalid configuration
except ProbabilityRangeError as e:
    print(str(e))  # Detailed message about which field and value was invalid
```

## See Also

- [User Guide: Troubleshooting](../reference/troubleshooting.md)
- [API: Records & Parsing](records-parsing.md)
- [API: Analysis Methods](analysis-methods.md)
