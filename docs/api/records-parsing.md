# Records & Parsing API

Data structures for binary sequences and dataset management.

## Core Classes

### BinaryLabels

Immutable pair of distinct state and observable labels:

```python
from bspe import BinaryLabels

labels = BinaryLabels(
    states=("Fair", "Biased"),
    observables=("Heads", "Tails"),
)
```

- `states`: Tuple of two distinct state names (strings)
- `observables`: Tuple of two distinct observable names (strings)

**Frozen and hashable; raises DuplicateLabelError if labels repeat.**

### BinarySequence

A binary sequence is a tuple of integers (0 or 1):

```python
BinarySequence = tuple[ObservableIndex, ...]  # ObservableIndex is Literal[0, 1]
```

Immutable and hashable. Created by parsing text or direct construction.

**Example:** `(0, 1, 0, 1, 0, 1)` represents "Heads Tails Heads ..." when observables are ("Heads", "Tails")

### SequenceRecord

Single independently observed sequence with optional evaluation target:

```python
from bspe import SequenceRecord

record = SequenceRecord(
    sequence_id="measurement_001",
    sequence="0010101",  # String or tuple of ints
    actual_target_index=1,  # Optional: ground truth for next observable
    labels=labels,
)
```

- `sequence_id`: Unique identifier (string)
- `sequence`: BinarySequence (string parsed or tuple)
- `actual_target_index`: Optional 0 or 1 for evaluation
- `labels`: BinaryLabels for parsing

**Frozen; raised InvalidSequenceTokenError if sequence contains invalid tokens.**

### SequenceDataset

Collection of SequenceRecords with shared BinaryLabels:

```python
from bspe import SequenceDataset

dataset = SequenceDataset(
    labels=labels,
    records=[record1, record2, record3],
)
```

- `labels`: Shared BinaryLabels for all records
- `records`: Sequence of SequenceRecords

**Frozen; raises DatasetValidationError if records have duplicate IDs or mismatched labels.**

## Parsing Functions

### parse_sequence

Convert text to BinarySequence using label mapping:

```python
from bspe import parse_sequence

text = "heads tails heads heads tails"
sequence = parse_sequence(text, labels, separator=" ")
# Result: (0, 1, 0, 0, 1)
```

- `text`: String of space/comma/hyphen-separated or continuous tokens
- `labels`: BinaryLabels for token mapping
- `separator`: Optional separator ("space", "comma", "hyphen", "underscore", or None for continuous)

**Raises InvalidSequenceTokenError if text contains unrecognized tokens.**

### Batch Parsing

Parse multiple sequences from CSV or TXT files:

#### CsvBatchColumns

Configuration for CSV column names:

```python
from bspe import CsvBatchColumns

columns = CsvBatchColumns(
    id_column="seq_id",
    sequence_column="values",
    actual_target_column="next_value",  # Optional
)
```

#### parse_batch_from_csv

Parse SequenceRecords from CSV file:

```python
from bspe import parse_batch_from_csv

records, issues = parse_batch_from_csv(
    filepath="data.csv",
    columns=columns,
    labels=labels,
)

if issues:
    for issue in issues:
        print(f"Row {issue.record_index}: {issue.detail}")

# Use valid records
dataset = SequenceDataset(labels, records)
```

Returns `(records, issues)` tuple. Issues are **recoverable**; use only valid records.

#### parse_batch_from_txt

Parse SequenceRecords from TXT file (one sequence per line):

```python
from bspe import parse_batch_from_txt

records, issues = parse_batch_from_txt(
    filepath="sequences.txt",
    labels=labels,
    id_prefix="seq_",  # Auto-generate IDs as "seq_0", "seq_1", ...
    separator=" ",
)

dataset = SequenceDataset(labels, records)
```

- `filepath`: Path to TXT file
- `labels`: BinaryLabels for parsing
- `id_prefix`: Prefix for auto-generated IDs
- `separator`: Token separator

## Usage Examples

### Creating Labels and Records

```python
from bspe import BinaryLabels, SequenceRecord, SequenceDataset

# Define labels
labels = BinaryLabels(
    states=("State_A", "State_B"),
    observables=("Zero", "One"),
)

# Create a single record
record = SequenceRecord(
    sequence_id="measurement_001",
    sequence="0010101",  # String parsed automatically
    labels=labels,
)

# Create dataset
dataset = SequenceDataset(
    labels=labels,
    records=[record],
)
```

### Parsing from Text

```python
from bspe import parse_sequence

text = "heads tails heads heads tails"
labels = BinaryLabels(
    states=("Coin", "Table"),
    observables=("H", "T"),
)

sequence = parse_sequence(text, labels, separator=" ")
# Result: (0, 1, 0, 0, 1)
```

### Parsing Batch from CSV

```python
from bspe import parse_batch_from_csv, CsvBatchColumns

columns = CsvBatchColumns(
    id_column="sequence_id",
    sequence_column="sequence_values",
    actual_target_column="next_value",
)

records, issues = parse_batch_from_csv(
    filepath="data.csv",
    columns=columns,
    labels=labels,
)
```

### Parsing Batch from TXT

```python
from bspe import parse_batch_from_txt

records, issues = parse_batch_from_txt(
    filepath="sequences.txt",
    labels=labels,
    id_prefix="seq_",
)
```

## Error Handling

Parsing operations return issues separately from records:

```python
records, issues = parse_batch_from_csv(...)

if issues:
    for issue in issues:
        print(f"Row {issue.record_index}: {issue.detail}")

# Use only valid records
valid_dataset = SequenceDataset(labels=labels, records=records)
```

## Immutability and Validation

All record and dataset objects are frozen (immutable):

```python
record = SequenceRecord("id", "010", labels=labels)
record.sequence = "101"  # Raises FrozenInstanceError
```

Validation occurs at construction:

```python
# Invalid: observable not in labels
try:
    SequenceRecord("id", "xyz", labels=labels)
except InvalidSequenceTokenError:
    print("Token 'x' not recognized")
```

## See Also

- [Concepts: Sequences & Records](../getting-started/concepts.md)
- [API: Analysis Methods](analysis-methods.md)
- [Examples: VMM Analysis](../examples/vmm-analysis.md)
