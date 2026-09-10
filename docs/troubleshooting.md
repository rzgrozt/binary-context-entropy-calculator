# Troubleshooting Guide

Common issues, error messages, and solutions for **bspe**.

## Table of Contents

- [Parsing Errors](#parsing-errors)
- [Configuration Errors](#configuration-errors)
- [Analysis Errors](#analysis-errors)
- [Stimulus Search Errors](#stimulus-search-errors)
- [Streamlit Issues](#streamlit-issues)
- [Performance Issues](#performance-issues)

---

## Parsing Errors

### `BatchParseError: invalid_utf8`

**Error Message:**
```
batch parse failed (invalid_utf8): 'utf-8' codec can't decode byte 0xff in position 0
```

**Cause:** File is not valid UTF-8 (or has incorrect encoding).

**Solution:**
1. Ensure file is saved as UTF-8 (not UTF-16, Latin-1, etc.)
2. If using a BOM (Byte Order Mark), ensure it's UTF-8 BOM (`EF BB BF`)
3. Check for non-ASCII characters that may have been corrupted

**Example:**
```python
# ✓ Correct: UTF-8 with optional BOM
with open("sequences.txt", "rb") as f:
    data = f.read()
dataset = parse_txt_batch(data, labels)

# ✗ Wrong: Reading as text then encoding
with open("sequences.txt", "r", encoding="latin-1") as f:
    text = f.read()
dataset = parse_manual_batch(text, labels)  # May fail
```

---

### `BatchRecordError: invalid_sequence`

**Error Message:**
```
batch parse failed with 1 record issue:
- row 2, record "seq-002", token 3 (invalid_sequence): invalid sequence token 'C' at position 3
```

**Cause:** Sequence contains a symbol that doesn't match either observable label.

**Solution:**
1. Check observable labels match your data
2. Verify symbols are separated by commas or whitespace
3. Ensure no extra characters (spaces, punctuation) in symbols

**Example:**
```python
# ✓ Correct
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, A, B", labels)

# ✗ Wrong: Symbol 'C' not in observables
dataset = parse_manual_batch("A, B, C, B", labels)
# → BatchRecordError: invalid_sequence

# ✗ Wrong: Extra spaces in symbol
labels = BinaryLabels(states=("S1", "S2"), observables=("A", "B"))
dataset = parse_manual_batch("A , B, A, B", labels)
# → BatchRecordError: invalid_sequence (token is "A " not "A")
```

---

### `BatchRecordError: duplicate_record_id`

**Error Message:**
```
batch parse failed with 1 record issue:
- row 3, record "seq-001" (duplicate_record_id): record ID 'seq-001' duplicates an earlier row
```

**Cause:** CSV has duplicate values in the ID column.

**Solution:**
1. Ensure all record IDs are unique
2. Check for typos or copy-paste errors
3. Use auto-generated IDs with `parse_manual_batch()` if IDs aren't critical

**Example:**
```python
# ✓ Correct: Unique IDs
columns = CsvBatchColumns(id_column="id", sequence_column="seq")
dataset = parse_csv_batch(csv_bytes, labels, columns)

# ✗ Wrong: Duplicate ID
# CSV:
# id,seq
# rec-001,A,B,A
# rec-002,B,A,B
# rec-001,A,A,B  ← Duplicate!
```

---

### `BatchRecordError: malformed_row`

**Error Message:**
```
batch parse failed with 1 record issue:
- row 2, record "seq-001" (malformed_row): expected 3 cells; got 2
```

**Cause:** CSV row has wrong number of columns.

**Solution:**
1. Ensure all rows have the same number of columns
2. Check for missing commas or extra commas
3. Verify quoted fields don't contain unescaped quotes

**Example:**
```python
# ✗ Wrong: Row 2 missing a column
# CSV:
# id,seq,target
# rec-001,A B A,A
# rec-002,B A B        ← Missing target column!
# rec-003,A A B,B

# ✓ Correct: All rows have 3 columns
# CSV:
# id,seq,target
# rec-001,A B A,A
# rec-002,B A B,
# rec-003,A A B,B
```

---

## Configuration Errors

### `InvalidVMMConfigurationError: minimum_support must be finite and positive`

**Error Message:**
```
minimum_support must be finite and positive; got 0
```

**Cause:** `minimum_support` is ≤ 0 or non-finite.

**Solution:**
1. Set `minimum_support` to a positive integer (≥ 1)
2. Common values: 1 (accept all contexts), 2 (require at least 2 observations)

**Example:**
```python
# ✗ Wrong
config = VMMConfig(minimum_support=0)  # → Error

# ✓ Correct
config = VMMConfig(minimum_support=2)
```

---

### `InvalidVMMConfigurationError: alpha must be finite and positive`

**Error Message:**
```
alpha must be finite and positive; got -0.5
```

**Cause:** Smoothing alpha is negative or non-finite.

**Solution:**
1. Use `KTSmoothing()` for default (α=0.5)
2. Use `MLESmoothing()` for no smoothing (α=0)
3. Use `AdditiveSmoothing(alpha=X)` for custom positive α

**Example:**
```python
# ✗ Wrong
config = VMMConfig(smoothing=AdditiveSmoothing(alpha=-0.1))  # → Error

# ✓ Correct
config = VMMConfig(smoothing=AdditiveSmoothing(alpha=0.1))
```

---

### `InvalidStimulusSearchConfigurationError: invalid stimulus search sequence_length: 0`

**Error Message:**
```
invalid stimulus search sequence_length: 0
```

**Cause:** Sequence length is ≤ 0 or > 32.

**Solution:**
1. Set `sequence_length` to 1–32
2. Longer sequences require exponentially more candidates

**Example:**
```python
# ✗ Wrong
config = StimulusSearchConfig(sequence_length=0, ...)  # → Error
config = StimulusSearchConfig(sequence_length=64, ...)  # → Error (> 32)

# ✓ Correct
config = StimulusSearchConfig(sequence_length=12, ...)
```

---

### `InvalidStimulusSearchConfigurationError: invalid stimulus search desired_stimuli: 100`

**Error Message:**
```
invalid stimulus search desired_stimuli: 100
```

**Cause:** `desired_stimuli` > `candidate_limit`.

**Solution:**
1. Increase `candidate_limit` or decrease `desired_stimuli`
2. Ensure `desired_stimuli` ≤ `candidate_limit`

**Example:**
```python
# ✗ Wrong
config = StimulusSearchConfig(
    desired_stimuli=100,
    candidate_limit=50,  # Too small!
    ...
)

# ✓ Correct
config = StimulusSearchConfig(
    desired_stimuli=50,
    candidate_limit=100,
    ...
)
```

---

## Analysis Errors

### `ProbabilityRangeError: probability must be finite and in [0, 1]`

**Error Message:**
```
probability[0] must be finite and in [0, 1]; got 1.5
```

**Cause:** HMM probability is outside [0, 1] or non-finite.

**Solution:**
1. Ensure all HMM probabilities are in [0, 1]
2. Check for typos (e.g., `0.15` vs `1.5`)

**Example:**
```python
# ✗ Wrong
hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.7, 0.3], [0.2, 0.8]],
    emission=[[1.5, -0.5], [0.2, 0.8]],  # Invalid!
)

# ✓ Correct
hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.7, 0.3], [0.2, 0.8]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)
```

---

### `ProbabilitySumError: initial must sum to 1`

**Error Message:**
```
initial must sum to 1; got 0.9
```

**Cause:** HMM distribution doesn't sum to 1.0 (within tolerance).

**Solution:**
1. Ensure probabilities sum to exactly 1.0
2. Use `math.fsum()` for accurate summation

**Example:**
```python
# ✗ Wrong
hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.3],  # Sums to 0.9, not 1.0!
    ...
)

# ✓ Correct
hmm = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],  # Sums to 1.0
    ...
)
```

---

### `ZeroLikelihoodError: observable 1 at position 1 has zero likelihood`

**Error Message:**
```
observable 1 at position 1 has zero likelihood
```

**Cause:** HMM emission probability is 0 for an observed symbol.

**Solution:**
1. Adjust HMM emission probabilities to be > 0
2. Use small smoothing (e.g., 0.01) if needed

**Example:**
```python
# ✗ Wrong
hmm = BinaryHMM(
    labels=labels,
    initial=[1.0, 0.0],
    transition=[[1.0, 0.0], [0.0, 1.0]],
    emission=[[1.0, 0.0], [0.0, 1.0]],  # State 1 can't emit B!
)
# Analyzing sequence (1,) will fail

# ✓ Correct: Add small probability
hmm = BinaryHMM(
    labels=labels,
    initial=[1.0, 0.0],
    transition=[[1.0, 0.0], [0.0, 1.0]],
    emission=[[0.99, 0.01], [0.01, 0.99]],  # All probabilities > 0
)
```

---

## Stimulus Search Errors

### `InvalidStimulusSearchConfigurationError: invalid stimulus search seed: 9223372036854775808`

**Error Message:**
```
invalid stimulus search seed: 9223372036854775808
```

**Cause:** Seed is outside [0, 2^63 - 1].

**Solution:**
1. Use seed in range [0, 9223372036854775807]
2. Common practice: use small seeds (0–10000) or current timestamp

**Example:**
```python
# ✗ Wrong
config = StimulusSearchConfig(seed=2**63, ...)  # Too large!

# ✓ Correct
config = StimulusSearchConfig(seed=2026, ...)
```

---

### `InvalidStimulusSearchConfigurationError: invalid stimulus search symbol_mapping: ('A', 'A')`

**Error Message:**
```
invalid stimulus search symbol_mapping: ('A', 'A')
```

**Cause:** Symbol mapping has duplicate labels or unsafe characters.

**Solution:**
1. Ensure both symbols are distinct
2. Avoid commas, pipes, newlines in symbols
3. Trim whitespace

**Example:**
```python
# ✗ Wrong
config = StimulusSearchConfig(
    symbol_mapping=("A", "A"),  # Duplicate!
    ...
)

config = StimulusSearchConfig(
    symbol_mapping=("A,B", "C"),  # Contains comma!
    ...
)

# ✓ Correct
config = StimulusSearchConfig(
    symbol_mapping=("Heads", "Tails"),
    ...
)
```

---

### Search returns `PARTIAL` status

**Cause:** Desired stimuli not reached before hitting candidate limit or universe exhaustion.

**Solution:**
1. Increase `candidate_limit` (up to 5000)
2. Relax hard constraints
3. Accept partial results if acceptable for your use case

**Example:**
```python
result = search_stimuli(config)
if result.status == SearchStatus.PARTIAL:
    print(f"Reason: {result.partial_reason}")
    print(f"Selected: {result.selected_count} / {result.config.desired_stimuli}")
    # Inspect violations to see which constraints are too strict
    for violation in result.violations:
        print(f"  {violation.constraint}: {violation.frequency:.1%}")
```

---

## Streamlit Issues

### `StreamlitAPIException: Cannot write to widget with key "X" outside of a form`

**Cause:** Widget created outside of a form context.

**Solution:**
1. Ensure widgets are created within `with st.form():`
2. Check for nested form issues

**Example:**
```python
# ✗ Wrong
value = st.slider("Select", 0, 10)  # Outside form

# ✓ Correct
with st.form("my_form"):
    value = st.slider("Select", 0, 10)
    st.form_submit_button("Submit")
```

---

### Results not updating after calculation

**Cause:** Session state not cleared when inputs change.

**Solution:**
1. Use `st.session_state` to track form fingerprint
2. Clear stale results when inputs change
3. Use `@st.cache_data` with appropriate dependencies

**Example:**
```python
# In sidebar.py
form = render_workbench_form(methods, labels)
if st.button("Calculate"):
    store_workbench_calculation(calculate_workbench(form), form)

# Results are only shown if fingerprint matches
calculation = workbench_calculation_record()
if calculation and calculation.success.fingerprint == form.fingerprint():
    render_results(calculation)
```

---

## Performance Issues

### Analysis is slow

**Cause:** Large dataset or high minimum support.

**Solution:**
1. Reduce dataset size (fewer records or shorter sequences)
2. Increase `minimum_support` to reduce context counts
3. Use per-sequence analysis instead of pooled (less data per model)

**Example:**
```python
# Slower: Pooled with low support
result = analyze_vmm(dataset, VMMConfig(minimum_support=1))

# Faster: Per-sequence with higher support
result = analyze_vmm_per_sequence(dataset, VMMConfig(minimum_support=3))
```

---

### Stimulus search is slow

**Cause:** Large candidate limit or expensive constraints.

**Solution:**
1. Reduce `candidate_limit` (default 1000, max 5000)
2. Reduce `sequence_length` (exponential growth: 2^length candidates)
3. Relax constraints to reduce filtering overhead

**Example:**
```python
# Slower: Large search space
config = StimulusSearchConfig(
    sequence_length=20,
    candidate_limit=5000,
    ...
)

# Faster: Smaller search space
config = StimulusSearchConfig(
    sequence_length=12,
    candidate_limit=1000,
    ...
)
```

---

### Memory usage is high

**Cause:** Large dataset or many candidates in memory.

**Solution:**
1. Stimulus search processes in 64-candidate batches (automatic)
2. Reduce dataset size
3. Use per-sequence analysis (smaller models)

---

## Getting Help

If you encounter an issue not listed here:

1. **Check the README**: Common patterns and examples
2. **Review API Reference**: `docs/api_reference.md`
3. **Check tests**: `tests/` contains working examples
4. **Open an issue**: [GitHub Issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
   - Include Python version, error message, and minimal reproduction
