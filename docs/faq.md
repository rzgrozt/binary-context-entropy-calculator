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

**Example:**
```python
# VMM: Detects that "AA" predicts B better than just "A"
sequence = "A, A, B, A, A, B, A, A, B"
# Context "AA" → B (100%)
# Context "A" → B (66%)

# First-order: Only sees previous symbol
# Previous "A" → B (66%)
```

### When should I use per-sequence vs. pooled analysis?

| Mode | Use When | Pros | Cons |
|------|----------|------|------|
| **Pooled** | Sequences are similar | More data per model | Assumes homogeneity |
| **Per-sequence** | Sequences are different | Captures individual patterns | Less data per model |

**Example:**
```python
# Pooled: Good for repeated trials from same subject
result = analyze_vmm(dataset, VMMConfig(), VMMResultScope.POOLED)

# Per-sequence: Good for different subjects or conditions
result = analyze_vmm_per_sequence(dataset, VMMConfig(), VMMResultScope.PER_SEQUENCE)
```

### What smoothing method should I use?

| Method | Alpha | Use When | Pros | Cons |
|--------|-------|----------|------|------|
| **KT** | 0.5 | Default choice | Principled, balanced | Arbitrary constant |
| **MLE** | 0 | Unseen contexts OK | No pseudocount | Fails on unseen contexts |
| **Additive** | Custom | Fine-tuning needed | Flexible | Requires justification |

**Recommendation:** Start with KT (default), then try MLE or custom alpha if needed.

### What is "minimum support"?

**Minimum support** is the threshold for accepting a context:
- Contexts with fewer observations are rejected
- Analysis backs off to shorter suffix
- Higher values → more stable, fewer contexts
- Lower values → more specific, more contexts

**Example:**
```python
# minimum_support=1: Accept all contexts
# Context "AAA" with 1 observation → accepted

# minimum_support=2: Require at least 2 observations
# Context "AAA" with 1 observation → rejected, back off to "AA"
```

### How do I interpret predictive entropy?

**Predictive entropy** (in bits) measures prediction uncertainty:
- **0 bits**: Certain prediction (P=1.0 or P=0.0)
- **0.5 bits**: Moderate uncertainty (P≈0.71 or P≈0.29)
- **1 bit**: Maximum uncertainty (P=0.5)

**Example:**
```python
# P(A)=0.9, P(B)=0.1 → entropy ≈ 0.47 bits (confident)
# P(A)=0.5, P(B)=0.5 → entropy = 1.0 bits (uncertain)
```

### What does "target assessment" mean?

**Target assessment** evaluates an optional observed next symbol:
- **Probability**: P(observed symbol)
- **Surprisal**: -log₂(probability) in bits
- **Classification**: MODAL (predicted), LOWER_PROBABILITY, or TIED

**Example:**
```python
# Prediction: P(A)=0.8, P(B)=0.2
# Observed: B
# Classification: LOWER_PROBABILITY (B was less likely)
# Surprisal: -log₂(0.2) ≈ 2.32 bits
```

---

## Stimulus Search Questions

### What is stimulus search?

**Stimulus search** generates and ranks binary sequences matching your criteria:
1. Generate bounded sample of candidates
2. Analyze each with VMM
3. Filter by hard constraints
4. Rank by soft preferences
5. Optionally match complements and assign targets

**Use case:** Create experimental stimuli with specific properties.

### What are hard constraints vs. soft preferences?

| Type | Behavior | Example |
|------|----------|---------|
| **Hard constraint** | Must satisfy or reject | Predicted symbol = A |
| **Soft preference** | Ranked by distance | Entropy close to 0.5 bits |

**Example:**
```python
constraints = StimulusConstraints(
    predicted_symbol=PredictedSymbol.A,  # Hard: must predict A
    predicted_probability=InclusiveRange(0.6, 0.9),  # Hard: P(A) in [0.6, 0.9]
)

preferences = (
    SoftPreference(PreferenceMetric.PREDICTIVE_ENTROPY, 0.5, weight=2.0),  # Soft: prefer entropy ≈ 0.5
)
```

### Why does my search return PARTIAL status?

**Partial status** means desired stimuli weren't reached. Reasons:
- **CANDIDATE_LIMIT**: Hit the candidate limit before finding enough
- **UNIVERSE_EXHAUSTED**: Searched entire binary universe (2^length)

**Solution:**
1. Increase `candidate_limit` (up to 5000)
2. Relax hard constraints
3. Accept partial results if acceptable

### How do I interpret constraint violations?

**Violations** show which constraints rejected the most candidates:

```python
for violation in result.violations:
    print(f"{violation.constraint}: {violation.frequency:.1%} rejected")
    # e.g., "predicted_probability: 45.2% rejected"
```

**Interpretation:**
- High frequency → constraint is too strict
- Low frequency → constraint is reasonable

### What is complement matching?

**Complement matching** pairs sequences with opposite predictions:
- Generate bitwise complements (0↔1)
- Match within tolerances (entropy, probability, runs, etc.)
- Useful for balanced experimental designs

**Example:**
```python
candidates = result.accepted
complements = create_complement_candidates(candidates, config)
pairs = match_stimuli(candidates + complements, tolerances)
# Each pair has opposite predictions but similar structure
```

### How do I assign targets?

**Target assignment** adds observed next symbols:

```python
# Balanced: Equal expected/unexpected
assigned = assign_targets(candidates, seed=2026)

# Or use explicit policy
assigned = assign_outcomes(candidates, TargetAssignmentMode.BALANCED, seed=2026)
```

**Policies:**
- **BALANCED**: Equal expected/unexpected
- **EXPECTED**: Always match prediction
- **UNEXPECTED**: Always opposite prediction
- **A** / **B**: Always assign A or B

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

### Can I include optional targets?

**Yes.** Targets are optional observed next symbols:

```python
# Single sequence with target
record = SequenceRecord(
    sequence_id="seq-001",
    sequence=(0, 1, 1, 0),
    actual_target_index=1,  # Observed next was B (index 1)
)

# CSV with target column
columns = CsvBatchColumns(
    id_column="id",
    sequence_column="seq",
    actual_target_column="target",
)
```

**Note:** Targets never influence fitting or prediction; they're assessed after analysis.

---

## Export Questions

### What formats can I export?

**bspe** exports:
- **CSV**: Tabular data (prefix results, candidates, evidence)
- **JSON**: Configuration, results, reproducibility info

**Method-specific exports:**
- **VMM**: Context model JSON, context evidence CSV, evaluation CSV
- **Markov**: Model JSON, prefix CSV, batch-summary CSV
- **HMM**: Preset JSON, prefix CSV, summary CSV
- **Stimulus Search**: Candidate CSV, scientific CSV, config JSON, experiment CSV

### How precise are exports?

| Format | Precision | Use Case |
|--------|-----------|----------|
| **UI display** | 3 decimal places | Visual inspection |
| **CSV** | 12+ fractional places | Data analysis |
| **JSON** | Full round-trip float64 | Reproducibility |

### Can I reproduce an analysis from exports?

**Yes.** Exports include:
- Full configuration
- Timestamp (UTC)
- Package version
- Sorted results

Use `stimulus_generator_config_json()` to capture reproducibility info.

---

## Performance Questions

### How fast is bspe?

**bspe** prioritizes correctness over speed. Typical times:
- **VMM analysis**: ~10ms per record (12-symbol sequence)
- **Markov analysis**: ~1ms per record
- **HMM analysis**: ~5ms per record
- **Stimulus search**: ~1s for 1000 candidates (12-symbol)

**Factors:**
- Sequence length (exponential for stimulus search)
- Dataset size
- Minimum support (higher = faster)
- Constraint complexity

### How much memory does bspe use?

**Typical usage:**
- **VMM model**: ~1KB per context (hundreds to thousands)
- **Markov model**: ~1KB (fixed size)
- **HMM model**: ~1KB (fixed size)
- **Stimulus search**: ~64 candidates in memory (batched)

**Scaling:**
- Longer sequences → more contexts → more memory
- Larger datasets → more records → more memory
- Stimulus search batches automatically (64 candidates)

### Can I speed up stimulus search?

**Yes:**
1. Reduce `sequence_length` (exponential effect)
2. Reduce `candidate_limit`
3. Increase `minimum_support` (fewer contexts to analyze)
4. Relax constraints (less filtering)

---

## Troubleshooting Questions

### My analysis gives unexpected results

**Checklist:**
1. Verify sequence parsing: Print first few records
2. Check configuration: Smoothing, minimum support, scope
3. Inspect depth evidence: See which contexts were used
4. Compare methods: VMM vs. Markov vs. HMM
5. Review hand-calculated example: Verify against manual calculation

### How do I debug a Streamlit issue?

**Steps:**
1. Run with `streamlit run streamlit_app.py --logger.level=debug`
2. Check browser console (F12) for JavaScript errors
3. Inspect session state: `st.write(st.session_state)`
4. Add `st.write()` statements to trace execution
5. Check terminal output for Python errors

### Where can I find more help?

- **README**: Overview and quick start
- **API Reference**: `docs/api_reference.md`
- **Examples**: `docs/examples/`
- **Tests**: `tests/` (working code examples)
- **Issues**: [GitHub Issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
