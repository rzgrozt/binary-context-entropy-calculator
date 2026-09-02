# Binary Sequence Probability, Prediction & Entropy Workbench

A local Streamlit workbench for fitting, comparing, and inspecting binary
sequence methods. It keeps independently submitted records separate and makes
the distinction between a next-symbol prediction and a description of symbols
already observed explicit.

## Install and run

Python 3.13 or newer is required.

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Choose methods and provide data

The setup screen selects **Markov Chain** by default. Choose any subset of:

- **Markov Chain**, with Variable-order Markov selected by default and First-order Markov available as a baseline.
- **Hidden Markov Model**, a configured two-hidden-state, two-observable
  predictive model.
- **Observed Shannon Entropy**, a descriptive analysis with no prediction.

Only controls for selected methods appear. Results run only after
**Calculate selected methods** is pressed, and changing a relevant input hides
the stale result until recalculation.

The two observable labels are shared by all selected methods. Labels may
contain spaces, so sequences are parsed as complete labels rather than by
splitting every space. Commas and whitespace, including tabs and newlines,
separate symbols within a sequence.

### Input modes and record boundaries

- **Single sequence** accepts one sequence. Newlines remain part of that one
  sequence. A sequence ID and optional observed next target apply to it.
- **Batch paste** treats each nonblank physical line as an independent
  sequence. IDs are assigned in submitted order as `sequence-001`,
  `sequence-002`, and so on. An optional selected target applies to every
  record.
- **TXT upload** accepts one `.txt` file, decoded as strict UTF-8 with an
  optional UTF-8 BOM. Each nonblank physical line is one independent sequence.
  An optional selected target applies to every record.
- **CSV upload** accepts one `.csv` file, decoded as strict UTF-8 with an
  optional UTF-8 BOM. Map the ID and sequence columns explicitly, then
  optionally map a target column. Each CSV row is one record, and a mapped
  target must contain at most one configured symbol.

No method concatenates records or counts a transition from the end of one
record to the start of another. Empty sequences are allowed where the chosen
input supplies a valid record ID, although some quantities are unavailable
without observations or transitions.

## Methods and equations

All logarithms are base 2. For a binary distribution `(p, 1-p)`, entropy is

```text
H(p) = -p log2(p) - (1-p) log2(1-p)
```

with `0 log2(0) = 0`. Thus binary entropy is in `[0, 1]` bits.

### Variable-order Markov

Variable-order Markov (VMM) is the default predictive workflow. The model
detects and predicts recurrent finite-context statistical dependencies in
binary sequences. It does not claim to discover every possible pattern.

For each order `k` from 0 through the usable suffix depth, the model counts a
context `c` and the symbols that followed it within each independent record:

```text
N(c, x) = number of within-record occurrences of context c followed by x
N(c) = N(c, A) + N(c, B)
```

Order 0 uses observed symbol counts and has no suffix. For the current record,
the model examines suffixes from deepest to shortest and uses the deepest one
whose support `N(c)` meets the configured minimum. Unseen or under-supported
suffixes back off to the next shorter supported suffix; backoff is never
silent. Records are never concatenated.

VMM offers three explicit estimation choices:

- **Krichevsky-Trofimov (KT)** is the default and fixes `alpha = 0.5`:

  ```text
  P(next = x | c) = (N(c, x) + 0.5) / (N(c) + 1)
  ```
- **Maximum likelihood estimation (MLE)** fixes `alpha = 0`:

  ```text
  P(next = x | c) = N(c, x) / N(c)
  ```

  MLE unavailable: unseen context has no occurrences in the training dataset.
- **Custom additive smoothing** accepts only a positive `alpha`:

  ```text
  P(next = x | c) = (N(c, x) + alpha) / (N(c) + 2 alpha)
  ```

Choose a pooled model to sum within-record counts across independent sequences
and analyze each sequence against the shared fit, or choose per-sequence
analysis to fit each record separately. Every record reports the effective
predictive context depth, context used, support, next-symbol probabilities,
prediction or tie, predictive Shannon entropy, A/B surprisal, and a table of
all examined depths. The context-depth evidence table preserves workflow,
scope, support and sparse status, the automatic suffix-backoff outcome and
reason, and per-depth target values when a target is supplied. An optional
actual target is assessed only after prediction, is labeled `In-sample
evaluation, not held out`, and never contributes to fitting or context
selection.

For `A,A,B,A,A,B,A,A` with minimum support 2 and KT smoothing, suffix `AA`
has support 2 and gives `P(next B | AA) = 2.5 / 3`. The first-order suffix `A`
has continuation counts `(A=3, B=2)` and gives `P(next B | A) = 2.5 / 6`.

### First-order Markov baseline

The first-order baseline predicts after a nonempty prefix
uses the transition row for the current, final observed symbol:

```text
T[i, j] = P(X[t+1] = j | X[t] = i)
q_t = T[X[t], :]
```

This baseline does not condition directly on a longer history. Longer sequences
can affect a fitted transition estimate, but they do not create a higher-order
model. At depth 0, no current state exists, so a Markov prediction is
unavailable.

For transition counts `n[i, j]`, choose maximum likelihood estimation or
additive smoothing with `alpha >= 0`:

```text
T[i, j] = (n[i, j] + alpha) / (sum_j n[i, j] + 2 alpha)
```

Maximum likelihood uses `alpha = 0`. If no outgoing transition has been seen
for a state, its maximum-likelihood row and predictions from it are
unavailable. Laplace/add-one smoothing uses `alpha = 1`; a custom nonnegative
alpha is also available.

Choose one prefix mode:

- **Fixed fitted transition matrix** fits one model to the selected scope and
  uses its row for each prefix's current state.
- **Re-estimate from each prefix** fits from the prefix of that record at each
  depth. This updates estimates with evidence; it does not add higher-order
  memory.

Choose a result scope:

- **Pooled model** counts transitions across all records while preserving every
  record boundary. Its final fitted matrix is shared by record results.
- **Per-sequence analysis** fits a separate full-sequence model for each
  record.

The Markov results also report fitted transition counts, starting-symbol
frequencies, empirical conditional entropy, and, when identifiable, the unique
stationary distribution and its entropy rate.

### Configured Hidden Markov Model

The HMM remains a configured model with exactly two hidden states and two
observable symbols. It is not fitted or trained from the entered sequences.
Each record is filtered independently under the same submitted model.

Rows are distributions. `T[i, j]` is the transition probability from hidden
state `i` to `j`, `E[i, x]` is the probability of observable `x` from hidden
state `i`, and `pi` is the initial hidden distribution. At depth 0, no
observation or transition has been consumed:

```text
next_hidden_0 = pi
q_0 = normalize(pi @ E)
```

After observing `x_t`, filtering follows the existing convention:

```text
posterior_t = normalize(prior_t * E[:, x_t])
next_hidden_t = normalize(posterior_t @ T)
q_t = normalize(next_hidden_t @ E)
```

`q_t` is the distribution for the next observable after the consumed prefix.
The HMM controls preserve schema-v1 model preset compatibility. A preset holds
the preset name, hidden-state and observable labels, `initial`, `transition`,
and `emission`; it does not hold sequence data or a target.

### Observed-symbol Shannon entropy

Observed Shannon entropy summarizes the empirical symbol frequencies already
present in a record or pooled dataset:

```text
p_A = count(A) / n
H_observed = H(p_A)
```

It does not produce a next-symbol distribution, prediction, or target score.
The workbench shows pooled and per-sequence summaries, plus nonempty prefix
summaries for each record.

### Predictive entropy and target surprisal

VMM, first-order Markov, and HMM predictive entropy apply `H` to their
next-symbol distribution `q_t`. Observed-symbol Shannon entropy instead
describes data that has already been supplied, so the quantities are not
interchangeable.

An optional observed next target evaluates an existing final Markov or HMM
prediction without changing fitting or prediction:

```text
I(x) = -log2(q_t[x])
```

`I(x)` is `infinity` when `q_t[x] = 0`. When predicted probabilities tie, the
first configured observable is the reported modal symbol; target assessment
still identifies the probabilities as tied.

## Results, precision, and downloads

The UI displays finite scientific values to exactly three decimal places.
Calculations retain float64 precision. Display rounding is separate from raw
exports: HMM CSV exports use 12 fixed decimal places, Markov CSV exports
preserve round-trip float64 values with at least 12 fractional decimal places,
and JSON exports retain serialized numeric values. Tables, charts, and exports
use deterministic record and context-depth order.

Selected methods appear in a comparison table. Predictive fields are marked
not applicable for Observed Shannon Entropy. VMM shows per-sequence final
values, context-depth evidence, and a static predictive-entropy chart with a
fixed vertical axis from 0 to 1 bits. HMM and first-order Markov retain their
existing prefix results and charts.

Available downloads are method-specific:

- **Markov model JSON** contains the fitted model and analysis settings,
  including counts, matrix availability, stationary information, scope, and
  prefix mode. It does not contain source sequences.
- **Markov prefix CSV** contains one row for every depth of every submitted
  record, including context, fitted-transition count, prediction when
  available, and final target assessment when supplied.
- **Markov batch-summary CSV** contains one deterministic summary row per
  record with sequence, counts, fitted transition values, final prediction,
  observed Shannon entropy, optional target assessment, and method settings.
- **Context model JSON** is an experimental VMM artifact containing configured
  selection, training-data provenance, record stimuli, and every fitted context
  distribution. It does not claim a separate held-out evaluation dataset.
- **Context evidence CSV** is an experimental VMM artifact with every examined
  suffix in deterministic record and requested-depth order, including support,
  sparse, backoff, probability, entropy, and per-depth target fields.
- **Evaluation CSV** is an experimental VMM artifact with final predictions and
  optional targets. Supplied targets are reported as `In-sample evaluation, not
  held out`, never as held-out evaluation.
- **HMM preset JSON** imports and exports the schema-v1 configured model.
- **HMM prefix CSV** and **HMM candidate-summary CSV** are available for each
  record. The prefix file covers depths 0 through the complete prefix; the
  summary file records the configured model, observed entropy, final
  prediction, and an optional target assessment.

Observed Shannon Entropy currently has no download export.

## Reusable Python API

The package exposes immutable records, parsers, method requests, and analysis
functions. This example parses a batch and runs the default VMM analysis:

```python
from binary_entropy import (
    BinaryLabels,
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
    parse_manual_batch,
)

labels = BinaryLabels(states=("State 1", "State 2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B\nB, A", labels)
result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    ),
)
```

For the first-order baseline, use `MarkovAnalysisRequest`. For a configured
HMM, use `BinaryHMM`, `HMMAnalysisRequest`, and `analyze_dataset`.
`parse_csv_batch`, `parse_txt_batch`, `SequenceRecord`, `SequenceDataset`,
`ShannonAnalysisRequest`, `compare_methods`, `KTSmoothing`, `MLESmoothing`,
`AdditiveSmoothing`, and the VMM and first-order fit functions are also public
exports.

## Architecture

- `streamlit_app.py` provides setup, selected-method controls, explicit
  submission, stale-result handling, and rendering.
- `records.py` and `batch_parsing.py` define independent records and single,
  multiline, TXT, and CSV intake boundaries.
- `workbench.py` routes typed VMM, first-order Markov, HMM, and Shannon requests.
- `methods/vmm.py` and `vmm_types.py` implement boundary-preserving context
  counts, KT, MLE, and custom additive smoothing, automatic deepest-supported
  suffix selection, and per-record VMM results.
- `vmm_serialization.py` produces the experimental Context model JSON, Context
  evidence CSV, and Evaluation CSV artifacts.
- `methods/markov.py`, `methods/hmm.py`, and `methods/shannon.py` implement
  the three analyses.
- `markov_types.py`, `markov_information.py`, and the Markov serialization
  modules hold fitted-model values, information measures, and exports.
- The existing HMM domain, filtering, analysis, presentation, and serialization
  modules preserve the configured HMM calculation and schema-v1 preset path.
- `ui/` contains input controls, result tables, charts, downloads, and session
  state.

## Hand-worked HMM reference

The included HMM example uses:

```text
pi = [0.6, 0.4]
T = [[0.7, 0.3], [0.2, 0.8]]
E = [[0.9, 0.1], [0.2, 0.8]]
```

At depth 0, `q_0 = pi @ E = [0.62, 0.38]`. After observing `A`, the
unnormalized posterior is `[0.54, 0.08]`, its likelihood is `0.62`, and the
posterior is `[27/31, 4/31]`. The next hidden distribution is
`[0.635483870967742, 0.364516129032258]`; the following prediction is
`[0.644838709677419, 0.355161290322581]`; its predictive entropy is
`0.938593249062606` bits.

For `A,B,B,A,A,A,B`, the observed-symbol Shannon entropy is
`0.985228136034251` bits. `tests/fixtures/hand_sequence.json` is the
authoritative machine-readable source for this model and its complete HMM
prefix profile. `tests/unit/test_filtering_analysis.py` reproduces the
first-observation derivation and every filtering value; `tests/ui/test_results.py`
checks the visible HMM table.

## Tests and quality gates

Run the focused HMM reference checks or the full suite and static gates:

```bash
uv run pytest tests/unit/test_filtering_analysis.py tests/ui/test_results.py
uv run pytest
uv run ruff check .
uv run basedpyright
```

The suite covers parsers and record boundaries, Markov fitting and scope,
Shannon prefixes, serialization, and Streamlit `AppTest` workflows for method
selection, uploads, presets, targets, results, and stale state.

## Limitations

- The workbench is binary only. The HMM has two hidden states and two
  observables. VMM models recurrent finite suffix contexts, not arbitrary or
  causal patterns.
- VMM and first-order Markov fitting are count-based. The HMM is configured,
  not trained, and the workbench does not perform statistical inference.
- An unseen VMM context under MLE is unavailable rather than implicitly
  smoothed.
- A maximum-likelihood Markov row with no outgoing evidence is unavailable.
- Stationary distributions and entropy rates appear only when the fitted
  transition matrix identifies a unique stationary distribution.
- Results are conditional calculations, not causal claims or validation of a
  model's suitability for a dataset.
- Numerical work uses float64 arithmetic.

## Citation

If you use this software in research, cite:

> Ozturk, R. (2026). *Binary Sequence Probability, Prediction & Entropy
> Workbench* (Version 0.1.0) [Computer software]. GitHub.
> https://github.com/rzgrozt/binary-context-entropy-calculator

```bibtex
@software{ozturk2026binaryworkbench,
  author  = {Ozturk, Ruzgar},
  title   = {Binary Sequence Probability, Prediction \& Entropy Workbench},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rzgrozt/binary-context-entropy-calculator}
}
```
