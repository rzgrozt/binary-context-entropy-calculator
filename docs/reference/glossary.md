# Glossary

Quick reference for **bspe** terminology.

## Analysis Concepts

### Binary Sequence
An ordered list of two distinct symbols (e.g., A, B or Heads, Tails).

**Example:** `(0, 1, 1, 0, 0, 1)` or `"A, B, B, A, A, B"`

### Context
A suffix (recent history) used to predict the next symbol.

**Example:** In sequence `A, B, B, A`, the context for predicting after position 3 is `(B, A)`.

### Entropy (Predictive)
Measure of prediction uncertainty in bits.
- **0 bits:** Certain prediction (P=1.0 or P=0.0)
- **1 bit:** Maximum uncertainty (P=0.5)

**Formula:** `H(p) = -p log₂(p) - (1-p) log₂(1-p)`

### Surprisal
Information content of an observed outcome in bits.

**Formula:** `I(x) = -log₂(P(x))`

**Example:** If P(A)=0.8, surprisal of A is -log₂(0.8) ≈ 0.32 bits.

### Support
Number of times a context was observed in the training data.

**Example:** Context "AA" appears 5 times → support = 5.

## Method-Specific Terms

### Variable-Order Markov (VMM)

**Automatic Suffix Backoff:** When a context has insufficient support, VMM backs off to a shorter suffix.

**Example:**
- Context "AAA" has support 1 (too low)
- Back off to "AA" with support 3 (acceptable)

**Minimum Support:** Threshold for accepting a context (default: 1).

**Depth:** Length of the context suffix (0 = no context, 1 = previous symbol, etc.).

### First-Order Markov

**Transition Matrix:** 2×2 matrix of probabilities P(next | current).

**Stationary Distribution:** Long-run probability of each state (if unique).

**Entropy Rate:** Long-run average entropy per symbol.

### Hidden Markov Model (HMM)

**Hidden State:** Unobserved state that generates observations.

**Emission Probability:** P(observable | hidden state).

**Filtering:** Updating belief about hidden state after observing a symbol.

**Posterior:** Probability distribution over hidden states after observation.

### Shannon Entropy

**Observed Entropy:** Empirical entropy of symbol frequencies (no prediction).

**Example:** If dataset has 60% A and 40% B, entropy ≈ 0.97 bits.

## Smoothing Terms

### Krichevsky-Trofimov (KT)
Default smoothing with fixed α = 0.5.

**Formula:** `P(x|c) = (count_x + 0.5) / (support + 1)`

### Maximum Likelihood Estimation (MLE)
No smoothing; α = 0.

**Formula:** `P(x|c) = count_x / support`

**Limitation:** Fails on unseen contexts.

### Additive Smoothing
Custom smoothing with positive α.

**Formula:** `P(x|c) = (count_x + α) / (support + 2α)`

## Scope Terms

### Pooled Analysis
One model shared across all records.

**Pros:** More data per model, stable estimates
**Cons:** Assumes homogeneity

### Per-Sequence Analysis
Independent model for each record.

**Pros:** Captures individual patterns
**Cons:** Less data per model, less stable

## Stimulus Search Terms

### Candidate
A generated binary sequence analyzed independently.

### Hard Constraint
A filter that rejects candidates (never relaxed).

**Example:** "Predicted symbol must be A"

### Soft Preference
A ranking criterion using weighted absolute distance.

**Example:** "Prefer entropy close to 0.5 bits (weight=2.0)"

### Complement
Bitwise inverse of a sequence (0↔1).

**Example:** Complement of `A, B, A` is `B, A, B`.

### Matching
Pairing sequences with opposite predictions within tolerances.

### Target Assignment
Adding observed next symbols to candidates.

**Policies:**
- **BALANCED:** Equal expected/unexpected
- **EXPECTED:** Always match prediction
- **UNEXPECTED:** Always opposite prediction
- **A/B:** Always assign A or B

### Congruency
Relationship between assigned target and prediction.

- **EXPECTED:** Target matches prediction
- **UNEXPECTED:** Target opposes prediction
- **TIE:** Prediction was tied

## Data Format Terms

### Record
One independent binary sequence with optional ID and target.

### Boundary Preservation
Never concatenating or merging records.

**Why:** Maintains scientific validity and interpretability.

### Observable Label
Symbol name (e.g., "A", "B", "Heads", "Tails").

### State Label
Hidden state name (HMM only).

## Export Terms

### CSV Export
Tabular data with 12+ decimal places precision.

**Use:** Data analysis, spreadsheets.

### JSON Export
Structured configuration and results with full float64 precision.

**Use:** Reproducibility, archival.

### Context Model JSON
Experimental VMM artifact with all fitted distributions.

### Context Evidence CSV
Experimental VMM artifact with every examined suffix.

### Evaluation CSV
Experimental VMM artifact with final predictions and targets.

## Statistical Terms

### Probability Distribution
Vector of probabilities summing to 1.0.

**Example:** `[0.6, 0.4]` for P(A)=0.6, P(B)=0.4.

### Modal Prediction
Most likely outcome (highest probability).

**Example:** If P(A)=0.8, P(B)=0.2, modal prediction is A.

### Tie
When two outcomes have equal probability.

**Example:** P(A)=0.5, P(B)=0.5 → tie (no modal prediction).

### Likelihood
Probability of observed data given a model.

### Empirical
Based on observed data (not theoretical).

## Computational Terms

### Determinism
Same input always produces same output.

**In bspe:** Seeded randomness, sorted output, no floating-point ambiguity.

### Batching
Processing data in fixed-size chunks to bound memory.

**In bspe:** Stimulus search processes 64 candidates at a time.

### Backoff
Falling back to a simpler model when primary model fails.

**In VMM:** Back off to shorter suffix when context has low support.

## Abbreviations

| Abbreviation | Full Name |
|--------------|-----------|
| VMM | Variable-Order Markov |
| HMM | Hidden Markov Model |
| KT | Krichevsky-Trofimov |
| MLE | Maximum Likelihood Estimation |
| CSV | Comma-Separated Values |
| JSON | JavaScript Object Notation |
| UTF-8 | 8-bit Unicode Transformation Format |
| BOM | Byte Order Mark |
| P(x) | Probability of x |
| H(p) | Entropy of probability p |
| I(x) | Surprisal of x |

## See Also

- **[Concepts](../getting-started/concepts.md)**: Detailed explanations
- **[FAQ](../faq.md)**: Common questions
- **[API Reference](../api/overview.md)**: Complete API
