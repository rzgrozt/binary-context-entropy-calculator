# Records & Parsing API

Data structures for binary sequences and dataset management.

## Core Classes

### BinaryLabels

::: bspe.domain.BinaryLabels

### BinarySequence

A binary sequence is a tuple of exactly two distinct observable values:

```python
BinarySequence = tuple[ObservableIndex, ...]  # Where ObservableIndex is 0 or 1
```

Immutable and hashable. Created by parsing text or direct construction.

### SequenceRecord

::: bspe.records.SequenceRecord

### SequenceDataset

::: bspe.records.SequenceDataset

## Parsing Functions

### parse_sequence

::: bspe.parsing.parse_sequence

### Batch Parsing

::: bspe.batch_parsing.CsvBatchColumns

::: bspe.batch_parsing.parse_batch_from_csv

::: bspe.batch_parsing.parse_batch_from_txt

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

- [Concepts: Records & Boundaries](../getting-started/concepts.md#records--boundaries)
- [API: Analysis Methods](analysis-methods.md)
- [Examples: VMM Analysis](../examples/vmm-analysis.md)
