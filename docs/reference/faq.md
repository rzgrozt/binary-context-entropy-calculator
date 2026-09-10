# Frequently Asked Questions (FAQ)

Common questions about **bspe** and binary-sequence analysis.

## General Questions

### What is bspe?

**bspe** (Binary Sequence Prediction & Entropy) is a typed Python framework for analyzing binary sequences using:
- **Variable-order Markov (VMM)**: Context-aware prediction with automatic suffix backoff
- **First-order Markov**: Transition-matrix baseline
- **Hidden Markov Model (HMM)**: Configured two-state model
- **Observed Shannon Entropy**: Empirical symbol frequency analysis
- **Stimulus Search**: Bounded generation and ranking of candidate sequences

It includes a Streamlit workbench for interactive analysis and reproducible exports.

### Who should use bspe?

**bspe** is designed for:
- Researchers studying binary-sequence prediction and entropy
- Cognitive scientists analyzing behavioral sequences
- Anyone needing reproducible, boundary-preserving sequence analysis
- Users who want explicit control over smoothing, scopes, and constraints

### Is bspe suitable for production use?

**bspe** is research-grade software (v0.1.0). It prioritizes:
- Correctness and reproducibility over performance
- Explicit configuration over convenience
- Scientific validity over ease of use

For production use, consider:
- Thorough testing on your specific data
- Pinning dependencies to known-good versions
- Running the test suite in your environment

---

## Analysis Questions

### What's the difference between VMM and first-order Markov?

| Aspect | VMM | First-order Markov |
|--------|-----|-------------------|
| **Context** | Variable-length suffix (0 to N) | Fixed: previous symbol only |
| **Backoff** | Automatic to shorter suffix | N/A (always uses previous) |
| **Complexity** | O(N²) contexts | O(1) contexts |
| **Prediction** | More accurate with patterns | Simpler, more stable |
| **Use case** | Detect recurrent patterns | Baseline comparison |

### When should I use per-sequence vs. pooled analysis?

| Mode | Use When | Pros | Cons |
|------|----------|------|------|
| **Pooled** | Sequences are similar | More data per model | Assumes homogeneity |
| **Per-sequence** | Sequences are different | Captures individual patterns | Less data per model |

### What smoothing method should I use?

| Method | Alpha | Use When | Pros | Cons |
|--------|-------|----------|------|------|
| **KT** | 0.5 | Default choice | Principled, balanced | Arbitrary constant |
| **MLE** | 0 | Unseen contexts OK | No pseudocount | Fails on unseen contexts |
| **Additive** | Custom | Fine-tuning needed | Flexible | Requires justification |

**Recommendation:** Start with KT (default), then try MLE or custom alpha if needed.

---

## Stimulus Search Questions

### What is stimulus search?

**Stimulus search** generates and ranks binary sequences matching your criteria:
1. Generate bounded sample of candidates
2. Analyze each with VMM
3. Filter by hard constraints
4. Rank by soft preferences
5. Optionally match complements and assign targets

### What are hard constraints vs. soft preferences?

| Type | Behavior | Example |
|------|----------|---------|
| **Hard constraint** | Must satisfy or reject | Predicted symbol = A |
| **Soft preference** | Ranked by distance | Entropy close to 0.5 bits |

### Why does my search return PARTIAL status?

**Partial status** means desired stimuli weren't reached. Reasons:
- **CANDIDATE_LIMIT**: Hit the candidate limit before finding enough
- **UNIVERSE_EXHAUSTED**: Searched entire binary universe (2^length)

**Solution:**
1. Increase `candidate_limit` (up to 5000)
2. Relax hard constraints
3. Accept partial results if acceptable

---

## Data Format Questions

### What sequence format does bspe accept?

**bspe** accepts binary sequences as:
- **Symbols**: Observable labels (e.g., "A", "B", "Heads", "Tails")
- **Separated by**: Commas, spaces, tabs, or newlines
- **Parsed as**: Complete labels (not character-by-character)

**Examples:**
```python
# ✓ All valid
"A, B, A, B"
"A B A B"
"A\nB\nA\nB"
"Heads, Tails, Heads"
```

### How do I parse CSV files?

```python
from bspe import parse_csv_batch, CsvBatchColumns

columns = CsvBatchColumns(
    id_column="record_id",
    sequence_column="binary_sequence",
    actual_target_column="observed_next",  # Optional
)

with open("data.csv", "rb") as f:
    dataset = parse_csv_batch(f.read(), labels, columns)
```

---

## Export Questions

### What formats can I export?

**bspe** exports:
- **CSV**: Tabular data (prefix results, candidates, evidence)
- **JSON**: Configuration, results, reproducibility info

### How precise are exports?

| Format | Precision | Use Case |
|--------|-----------|----------|
| **UI display** | 3 decimal places | Visual inspection |
| **CSV** | 12+ fractional places | Data analysis |
| **JSON** | Full round-trip float64 | Reproducibility |

---

## Troubleshooting Questions

### My analysis gives unexpected results

**Checklist:**
1. Verify sequence parsing: Print first few records
2. Check configuration: Smoothing, minimum support, scope
3. Inspect depth evidence: See which contexts were used
4. Compare methods: VMM vs. Markov vs. HMM
5. Review hand-calculated example: Verify against manual calculation

### Where can I find more help?

- **README**: Overview and quick start
- **User Guide**: Detailed method descriptions
- **API Reference**: Complete API documentation
- **Examples**: Runnable code examples
- **Tests**: `tests/` (working code examples)
- **Issues**: [GitHub Issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
