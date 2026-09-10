# Serialization API

Export and persistence of sequences, results, and stimulus candidates.

## CSV Export

### markov_csv_text

Export analysis results as CSV:

::: bspe.markov_csv.markov_csv_text

### CsvCell

Type alias for CSV cell values:

```python
CsvCell = str | float | int | None
```

## Result Export Workflows

### Export Markov Analysis Results

```python
from bspe import analyze_dataset, MarkovAnalysisRequest, markov_csv_text

result = analyze_dataset(dataset, MarkovAnalysisRequest())

# Prepare rows for export
rows = []
for record in result.records:
    rows.append([
        record.sequence_id,
        record.sequence,
        record.predictive_entropy_bits,
        record.probability_observable_a,
    ])

# Generate CSV text
csv_text = markov_csv_text(
    columns=["id", "sequence", "entropy_bits", "prob_a"],
    rows=rows,
)

with open("results.csv", "w") as f:
    f.write(csv_text)
```

### Export with Per-Position Details

```python
rows = []
for record in result.records:
    for pos_idx, pos_result in enumerate(record.per_position_results):
        rows.append([
            record.sequence_id,
            pos_idx,
            record.sequence[pos_idx],
            pos_result.entropy_bits,
            pos_result.probability_observable_a,
        ])

csv_text = markov_csv_text(
    columns=["id", "position", "observable", "entropy", "prob_a"],
    rows=rows,
)
```

## Stimulus Export

### JSON Export

Export stimulus candidates as JSON:

```python
import json

result = search_stimulus(labels, constraints, preferences)

export_data = {
    "metadata": {
        "labels": {
            "states": labels.states,
            "observables": labels.observables,
        },
        "constraints_summary": {
            "min_length": constraints.minimum_length,
            "max_length": constraints.maximum_length,
            "entropy_range": [
                constraints.entropy_bits_minimum,
                constraints.entropy_bits_maximum,
            ],
        },
        "total_candidates": result.total_candidates_generated,
        "valid_candidates": len(result.candidates),
    },
    "candidates": [
        {
            "sequence": c.sequence,
            "entropy_bits": float(c.entropy_bits),
            "ranking_score": float(c.ranking_score),
            "length": len(c.sequence),
        }
        for c in result.candidates
    ],
}

with open("stimuli.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

### CSV Export for Stimuli

```python
import csv

with open("stimuli.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "sequence",
        "length",
        "entropy_bits",
        "ranking_score",
        "observable_a_freq",
    ])
    
    for candidate in result.candidates:
        seq = candidate.sequence
        a_freq = seq.count(labels.observables[0]) / len(seq)
        writer.writerow([
            seq,
            len(seq),
            candidate.entropy_bits,
            candidate.ranking_score,
            a_freq,
        ])
```

## Batch Parsing Export

### parse_batch_from_csv

::: bspe.batch_parsing.parse_batch_from_csv

### parse_batch_from_txt

::: bspe.batch_parsing.parse_batch_from_txt

### CsvBatchColumns

::: bspe.batch_parsing.CsvBatchColumns

## Roundtrip Workflows

### Parse → Analyze → Export

```python
from bspe import (
    parse_batch_from_csv,
    CsvBatchColumns,
    SequenceDataset,
    analyze_dataset,
    MarkovAnalysisRequest,
    markov_csv_text,
)

# 1. Parse input
columns = CsvBatchColumns(
    id_column="seq_id",
    sequence_column="values",
    actual_target_column="next_val",
)
records, issues = parse_batch_from_csv("input.csv", columns, labels)

# 2. Create dataset
dataset = SequenceDataset(labels, records)

# 3. Analyze
result = analyze_dataset(dataset, MarkovAnalysisRequest())

# 4. Export results
export_rows = [
    [
        r.sequence_id,
        r.sequence,
        r.predictive_entropy_bits,
        r.probability_observable_a,
        r.predicted_target_label if r.actual_target_index is not None else None,
        r.actual_target_label if r.actual_target_index is not None else None,
    ]
    for r in result.records
]

csv_text = markov_csv_text(
    columns=[
        "id",
        "sequence",
        "entropy_bits",
        "prob_a",
        "predicted_target",
        "actual_target",
    ],
    rows=export_rows,
)

with open("output.csv", "w") as f:
    f.write(csv_text)
```

## Format Specifications

### CSV Format (Input)

Expected columns:
- `id_column`: Unique sequence identifier (string)
- `sequence_column`: Observable sequence (string, space/comma/hyphen/underscore-separated or continuous)
- `actual_target_column` (optional): Ground truth next observable (0 or 1, or label)

### CSV Format (Output)

Typical columns:
- `sequence_id`: String identifier
- `sequence`: Sequence of observables
- `entropy_bits`: Float, 0–1 bits
- `probability_observable_a`: Float, 0–1
- `per_position_*`: Optional per-position fields

### JSON Format (Stimulus Export)

```json
{
  "metadata": {
    "labels": { "states": [...], "observables": [...] },
    "constraints_summary": { ... }
  },
  "candidates": [
    {
      "sequence": "0101...",
      "entropy_bits": 0.5,
      "ranking_score": 0.95,
      "length": 20
    }
  ]
}
```

## Error Handling

Parsing and export operations may produce warnings/errors:

```python
records, issues = parse_batch_from_csv("file.csv", columns, labels)

if issues:
    print(f"Found {len(issues)} issues during parsing:")
    for issue in issues:
        print(f"  Row {issue.record_index}: {issue.detail}")

# Only use valid records
valid_dataset = SequenceDataset(labels, records)
```

Export operations raise exceptions on file I/O errors:

```python
try:
    csv_text = markov_csv_text(columns, rows)
    with open("results.csv", "w") as f:
        f.write(csv_text)
except IOError as e:
    print(f"Failed to write results: {e}")
```

## See Also

- [User Guide: Input Formats](../user-guide/input-formats.md)
- [API: Records & Parsing](records-parsing.md)
- [API: Stimulus Search](stimulus-search.md)
- [Examples: Stimulus Search](../examples/stimulus-search.md)
