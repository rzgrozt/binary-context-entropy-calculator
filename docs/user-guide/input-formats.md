# Input Formats

This guide covers how to prepare and parse binary sequences for **bspe**.

## Overview

**bspe** accepts binary sequences in three formats:

1. **Manual Batch**: Direct Python strings
2. **Text File (TXT)**: UTF-8 encoded text files
3. **Comma-Separated Values (CSV)**: Structured data with column mapping

All formats produce a `SequenceDataset` with independent records and optional targets.

## Manual Batch Parsing

### Format

```python
from bspe import parse_manual_batch, BinaryLabels

labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B")
)

# One sequence per line, symbols separated by commas or spaces
dataset = parse_manual_batch(
    """
    A, B, B, A, A
    B, A, A, B
    A, A, B, B, A
    """,
    labels
)
```

### Symbol Separators

All of these are valid:

```python
# Commas with spaces
"A, B, B, A"

# Commas without spaces
"A,B,B,A"

# Spaces only
"A B B A"

# Tabs
"A\tB\tB\tA"

# Newlines (multiline)
"""A
B
B
A"""
```

### Auto-Generated IDs

Records are automatically assigned IDs: `sequence-001`, `sequence-002`, etc.

```python
dataset = parse_manual_batch("A, B\nB, A", labels)
print(dataset.records[0].sequence_id)  # "sequence-001"
```

## Text File Parsing

### Format

```python
from bspe import parse_txt_batch

# Read file as bytes
with open("sequences.txt", "rb") as f:
    payload = f.read()

dataset = parse_txt_batch(payload, labels)
```

### File Requirements

- **Encoding**: UTF-8 (with or without BOM)
- **Line format**: One sequence per line
- **Symbol separator**: Commas or whitespace (same as manual batch)
- **Empty lines**: Ignored

### Example File

```
A, B, B, A, A
B, A, A, B
A, A, B, B, A
```

### UTF-8 Handling

```python
# UTF-8 with BOM (e.g., from Windows Notepad)
with open("sequences.txt", "rb") as f:
    payload = f.read()  # Starts with BOM bytes: EF BB BF
dataset = parse_txt_batch(payload, labels)  # Handles automatically

# UTF-8 without BOM (e.g., from Linux/Mac)
with open("sequences.txt", "rb") as f:
    payload = f.read()
dataset = parse_txt_batch(payload, labels)  # Also works
```

## CSV Parsing

### Format

```python
from bspe import parse_csv_batch, CsvBatchColumns

columns = CsvBatchColumns(
    id_column="record_id",
    sequence_column="binary_sequence",
    actual_target_column="observed_next"  # Optional
)

with open("data.csv", "rb") as f:
    payload = f.read()

dataset = parse_csv_batch(payload, labels, columns)
```

### CSV Structure

Minimal (ID and sequence):

```csv
record_id,binary_sequence
seq-001,A B A B
seq-002,B A A B
seq-003,A A B B A
```

With targets:

```csv
record_id,binary_sequence,observed_next
seq-001,A B A B,A
seq-002,B A A B,B
seq-003,A A B B A,A
```

### Column Configuration

```python
# Specify which columns to use
columns = CsvBatchColumns(
    id_column="participant_id",           # Maps to CSV column
    sequence_column="responses",          # Maps to CSV column
    actual_target_column="final_response" # Optional
)
```

**Note:** Column names are case-sensitive and must match CSV headers exactly.

### Symbol Separators in CSV

Sequences in CSV cells can use any separator:

```csv
record_id,binary_sequence
seq-001,"A, B, A, B"
seq-002,"A B A B"
seq-003,"A\tB\tA\tB"
```

## Optional Targets

### What is a Target?

A **target** is the observed next symbol after the sequence. It's optional and used only for **post-hoc assessment**, never for fitting.

### Format

```python
record = SequenceRecord(
    sequence_id="seq-001",
    sequence=(0, 1, 1, 0),      # A, B, B, A
    actual_target_index=1        # Observed next was B (index 1)
)
```

### In CSV

```csv
record_id,binary_sequence,observed_next
seq-001,A B A B,A
seq-002,B A A B,
seq-003,A A B B A,B
```

Empty cells are treated as "no target".

### Assessment

After analysis, targets are assessed:

```python
result = analyze_dataset(dataset, config)
for record in result.records:
    if record.target_assessment:
        print(f"Predicted: {record.predicted_target_index}")
        print(f"Observed: {record.observed_target_index}")
        print(f"Probability: {record.target_assessment.probability}")
        print(f"Surprisal: {record.target_assessment.surprisal_bits}")
```

## Observable Labels

### Single-Character Labels

```python
labels = BinaryLabels(
    states=("S1", "S2"),
    observables=("A", "B")
)

dataset = parse_manual_batch("A, B, A, B", labels)
```

### Multi-Character Labels

```python
labels = BinaryLabels(
    states=("Hidden State 1", "Hidden State 2"),
    observables=("Correct", "Incorrect")
)

dataset = parse_manual_batch("Correct, Incorrect, Correct", labels)
```

### Consistency

Labels must be consistent across your data:

```python
# ✓ Correct: Observable labels are "Heads" and "Tails"
labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("Heads", "Tails")
)

dataset = parse_manual_batch("Heads, Tails, Heads", labels)

# ✗ Wrong: "Coin" is not defined
dataset = parse_manual_batch("Heads, Coin, Heads", labels)
# → BatchRecordError: invalid_sequence
```

## Error Handling

### Invalid Sequence Token

```python
# Error: "C" is not in observables
dataset = parse_manual_batch("A, B, C, A", labels)
# → BatchRecordError: invalid_sequence token 'C'
```

**Solution:** Ensure all symbols match observable labels.

### Duplicate Record IDs

```python
# Error: "seq-001" appears twice
columns = CsvBatchColumns(
    id_column="id",
    sequence_column="seq"
)

csv_data = b"id,seq\nseq-001,A B\nseq-001,B A"
dataset = parse_csv_batch(csv_data, labels, columns)
# → BatchRecordError: duplicate_record_id
```

**Solution:** Ensure all record IDs are unique.

### Malformed CSV Row

```python
# Error: Row 2 has wrong number of columns
csv_data = b"id,seq,target\nseq-001,A B,A\nseq-002,B A"
# → BatchRecordError: malformed_row (expected 3 cells, got 2)
```

**Solution:** Ensure all CSV rows have the same number of columns.

## Best Practices

### Data Preparation

1. **Consistent labels**: Use same observable labels throughout
2. **Unique IDs**: Each record should have a unique ID (auto-generated if manual)
3. **Valid encoding**: Save files as UTF-8 (not Latin-1, UTF-16, etc.)
4. **Clear separators**: Use commas or spaces (not tabs in manual input)

### CSV Format

1. **Include headers**: Always include column names
2. **Escape values**: If sequences contain commas, quote them: `"A, B, C"`
3. **Consistent separators**: Use commas as column delimiters
4. **Consistent symbol separators**: Use same separator within sequence cells

### Error Prevention

1. **Validate manually first**: Parse small sample to check for errors
2. **Check line endings**: Use consistent line endings (LF or CRLF)
3. **Trim whitespace**: Remove extra spaces around symbols
4. **Test round-trip**: Export and re-import to verify

## Examples

### Example 1: Simple Manual Batch

```python
from bspe import BinaryLabels, parse_manual_batch

labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B")
)

dataset = parse_manual_batch(
    """
    A, B, B, A, A
    B, A, A, B
    A, A, B, B, A
    """,
    labels
)

print(f"Parsed {len(dataset.records)} sequences")
for record in dataset.records:
    print(f"  {record.sequence_id}: {record.sequence_str}")
```

### Example 2: CSV with Targets

```python
from bspe import parse_csv_batch, CsvBatchColumns

# File: responses.csv
# participant,responses,final_response
# p001,A B A B,A
# p002,B A A B,B
# p003,A A B B,A

columns = CsvBatchColumns(
    id_column="participant",
    sequence_column="responses",
    actual_target_column="final_response"
)

with open("responses.csv", "rb") as f:
    dataset = parse_csv_batch(f.read(), labels, columns)

print(f"Parsed {len(dataset.records)} sequences with targets")
for record in dataset.records:
    print(f"  {record.sequence_id}: {record.sequence_str} → {record.actual_target_index}")
```

### Example 3: Text File

```python
from bspe import parse_txt_batch

# File: sequences.txt
# A, B, B, A, A
# B, A, A, B
# A, A, B, B, A

with open("sequences.txt", "rb") as f:
    dataset = parse_txt_batch(f.read(), labels)

print(f"Parsed {len(dataset.records)} sequences from file")
```

## Next Steps

- **[Methods Overview](methods/vmm.md)**: Choose an analysis method
- **[Quick Start](../getting-started/quick-start.md)**: Run your first analysis
- **[Workbench](workbench.md)**: Use the interactive interface
