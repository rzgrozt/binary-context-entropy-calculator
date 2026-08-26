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

- **Markov Chain**, a fitted first-order predictive model.
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

### First-order Markov Chain

The Markov method is first-order only. Its prediction after a nonempty prefix
uses the transition row for the current, final observed symbol:

```text
T[i, j] = P(X[t+1] = j | X[t] = i)
q_t = T[X[t], :]
```

The method does not condition directly on a longer history. Longer sequences
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

Markov and HMM predictive entropy apply `H` to their next-symbol distribution
`q_t`. Observed-symbol Shannon entropy instead describes data that has already
been supplied, so the two are not interchangeable.

An optional observed next target evaluates an existing final Markov or HMM
prediction without changing fitting or prediction:

```text
I(x) = -log2(q_t[x])
```

`I(x)` is `infinity` when `q_t[x] = 0`. When predicted probabilities tie, the
first configured observable is the reported modal symbol; target assessment
still identifies the probabilities as tied.

## Results, precision, and downloads

The UI displays scientific values to exactly three decimal places. Calculations
retain float64 precision. HMM CSV exports use 12 fixed decimal places; Markov
CSV exports preserve round-trip float64 values with at least 12 fractional
decimal places. JSON exports retain their serialized numeric values.

Selected methods appear in a comparison table. Predictive fields are marked
not applicable for Observed Shannon Entropy. Each predictive method also shows
per-sequence final values and prefix evidence. HMM results include an entropy
chart; Markov results include probability and entropy charts when predictions
are available.

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
- **HMM preset JSON** imports and exports the schema-v1 configured model.
- **HMM prefix CSV** and **HMM candidate-summary CSV** are available for each
  record. The prefix file covers depths 0 through the complete prefix; the
  summary file records the configured model, observed entropy, final
  prediction, and an optional target assessment.

Observed Shannon Entropy currently has no download export.

## Reusable Python API

The package exposes immutable records, parsers, method requests, and analysis
functions. This example parses a batch, runs a first-order Markov analysis,
and serializes its prefix rows:

```python
from binary_entropy import (
    BinaryLabels,
    MarkovAnalysisRequest,
    analyze_dataset,
    markov_sequence_csv,
    parse_manual_batch,
)

labels = BinaryLabels(states=("State 1", "State 2"), observables=("A", "B"))
dataset = parse_manual_batch("A, B, B\nB, A", labels)
result = analyze_dataset(dataset, MarkovAnalysisRequest(smoothing_alpha=1.0))
csv_text = markov_sequence_csv(result)
```

For a configured HMM, use `BinaryHMM`, `HMMAnalysisRequest`, and
`analyze_dataset`. `parse_csv_batch`, `parse_txt_batch`, `SequenceRecord`,
`SequenceDataset`, `ShannonAnalysisRequest`, `compare_methods`, and the Markov
fit and prediction functions are also public exports.

## Architecture

- `streamlit_app.py` provides setup, selected-method controls, explicit
  submission, stale-result handling, and rendering.
- `records.py` and `batch_parsing.py` define independent records and single,
  multiline, TXT, and CSV intake boundaries.
- `workbench.py` routes typed Markov, HMM, and Shannon requests.
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
  observables; Markov order is fixed at one.
- Markov fitting is count-based MLE or additive smoothing. There is no
  higher-order fitting, HMM training, or statistical inference.
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
