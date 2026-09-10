# Troubleshooting Guide

Common issues, error messages, and solutions for **bspe**.

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

---

## Configuration Errors

### `InvalidVMMConfigurationError: minimum_support must be positive`

**Error Message:**
```
minimum_support must be finite and positive; got 0
```

**Cause:** `minimum_support` is ≤ 0 or non-finite.

**Solution:**
1. Set `minimum_support` to a positive integer (≥ 1)
2. Common values: 1 (accept all contexts), 2 (require at least 2 observations)

---

### `InvalidVMMConfigurationError: alpha must be positive`

**Error Message:**
```
alpha must be finite and positive; got -0.5
```

**Cause:** Smoothing alpha is negative or non-finite.

**Solution:**
1. Use `KTSmoothing()` for default (α=0.5)
2. Use `MLESmoothing()` for no smoothing (α=0)
3. Use `AdditiveSmoothing(alpha=X)` for custom positive α

---

### `InvalidStimulusSearchConfigurationError: sequence_length out of range`

**Error Message:**
```
invalid stimulus search sequence_length: 0
```

**Cause:** Sequence length is ≤ 0 or > 32.

**Solution:**
1. Set `sequence_length` to 1–32
2. Longer sequences require exponentially more candidates

---

### `InvalidStimulusSearchConfigurationError: desired_stimuli exceeds candidate_limit`

**Error Message:**
```
invalid stimulus search desired_stimuli: 100
```

**Cause:** `desired_stimuli` > `candidate_limit`.

**Solution:**
1. Increase `candidate_limit` or decrease `desired_stimuli`
2. Ensure `desired_stimuli` ≤ `candidate_limit`

---

## Analysis Errors

### `ProbabilityRangeError: probability outside [0, 1]`

**Error Message:**
```
probability[0] must be finite and in [0, 1]; got 1.5
```

**Cause:** HMM probability is outside [0, 1] or non-finite.

**Solution:**
1. Ensure all HMM probabilities are in [0, 1]
2. Check for typos (e.g., `0.15` vs `1.5`)

---

### `ProbabilitySumError: probabilities don't sum to 1`

**Error Message:**
```
initial must sum to 1; got 0.9
```

**Cause:** HMM distribution doesn't sum to 1.0 (within tolerance).

**Solution:**
1. Ensure probabilities sum to exactly 1.0
2. Use `math.fsum()` for accurate summation

---

### `ZeroLikelihoodError: zero likelihood observed`

**Error Message:**
```
observable 1 at position 1 has zero likelihood
```

**Cause:** HMM emission probability is 0 for an observed symbol.

**Solution:**
1. Adjust HMM emission probabilities to be > 0
2. Use small smoothing (e.g., 0.01) if needed

---

## Stimulus Search Errors

### `InvalidStimulusSearchConfigurationError: seed out of range`

**Error Message:**
```
invalid stimulus search seed: 9223372036854775808
```

**Cause:** Seed is outside [0, 2^63 - 1].

**Solution:**
1. Use seed in range [0, 9223372036854775807]
2. Common practice: use small seeds (0–10000) or current timestamp

---

### Search returns `PARTIAL` status

**Cause:** Desired stimuli not reached before hitting candidate limit or universe exhaustion.

**Solution:**
1. Increase `candidate_limit` (up to 5000)
2. Relax hard constraints
3. Accept partial results if acceptable for your use case

---

## Streamlit Issues

### `StreamlitAPIException: Cannot write to widget outside form`

**Cause:** Widget created outside of a form context.

**Solution:**
1. Ensure widgets are created within `with st.form():`
2. Check for nested form issues

---

### Results not updating after calculation

**Cause:** Session state not cleared when inputs change.

**Solution:**
1. Use `st.session_state` to track form fingerprint
2. Clear stale results when inputs change
3. Use `@st.cache_data` with appropriate dependencies

---

## Performance Issues

### Analysis is slow

**Cause:** Large dataset or high minimum support.

**Solution:**
1. Reduce dataset size (fewer records or shorter sequences)
2. Increase `minimum_support` to reduce context counts
3. Use per-sequence analysis instead of pooled (less data per model)

---

### Stimulus search is slow

**Cause:** Large candidate limit or expensive constraints.

**Solution:**
1. Reduce `candidate_limit` (default 1000, max 5000)
2. Reduce `sequence_length` (exponential growth: 2^length candidates)
3. Relax constraints to reduce filtering overhead

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
2. **Review API Reference**: Complete API documentation
3. **Check tests**: `tests/` contains working examples
4. **Open an issue**: [GitHub Issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
   - Include Python version, error message, and minimal reproduction
