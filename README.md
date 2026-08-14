# Binary Sequence Predictive Entropy Calculator

A local Streamlit scientific calculator for one binary observed sequence under
one configured two-state hidden Markov model (HMM). It reports the model's
next-observable prediction at depth 0 and after every consumed prefix.

The entered sequence alone does not uniquely determine predictions. Results
are conditional on the selected HMM parameters.

## Install and run

Python 3.13 or newer is required.

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Implemented architecture

- `streamlit_app.py` is the single Streamlit entry point.
- `src/binary_entropy/domain.py` owns immutable, validated HMM and result
  values.
- `src/binary_entropy/parsing.py` parses sequence text into observable
  indices.
- `src/binary_entropy/filtering.py`, `analysis.py`, and `information.py`
  perform filtering, every-prefix analysis, entropy, and surprisal.
- `src/binary_entropy/presentation.py` and `serialization.py` provide stable
  tables and exports.
- `src/binary_entropy/ui/` collects form state, handles explicit submission
  and stale results, and renders summaries, tables, the entropy chart, and
  downloads.

The reusable package exports `BinaryHMM`, `BinaryLabels`,
`SequenceAnalysis`, `parse_sequence`, and `analyze_sequence`.

## Inputs and validation

The UI has exactly two observable labels and exactly two hidden-state labels.
Labels are trimmed, must be distinct within their category, and may be any
nonempty length except that commas and line breaks are not allowed. Observable
labels may contain spaces.

The sequence may be any length, including empty. It accepts configured whole
observable labels separated by commas, spaces, tabs, line breaks, or mixed
separators. For example, with labels `light red` and `deep blue`, this is
valid:

```text
light red, deep blue
light red deep blue
```

For the initial distribution `pi`, each row of the transition matrix `T`, and
each row of the emission matrix `E`, every value must be finite and in
`[0, 1]`; each two-value row must sum to 1 within an absolute tolerance of
`1e-12`. Invalid input is rejected, never silently normalized, clamped, or
redistributed.

## Model convention and equations

Rows are state distributions, `T[i, j]` is the transition from hidden state
`i` to `j`, and `E[i, x]` is the emission probability of observable `x` from
hidden state `i`. The initial distribution `pi` applies at the first
observation.

At context depth 0, no observation has been consumed and no transition is
applied:

```text
next_hidden_0 = pi
q_1 = normalize(pi @ E)
```

For observation `x_t` at a later depth, with the prior hidden distribution
`prior_t`, the calculator computes:

```text
unnormalized_posterior_t = prior_t * E[:, x_t]
posterior_t = normalize(unnormalized_posterior_t)
next_hidden_t = normalize(posterior_t @ T)
q_t = normalize(next_hidden_t @ E)
```

Thus the first-observation posterior is proportional to `pi * emission`; each
later prior is the preceding `posterior @ T`; and the next-hidden distribution
is also `posterior @ T`. Calculated vectors are normalized at every step.

`q_t` is the next-observable distribution after the prefix of depth `t`.
Depth is the number of observations already consumed, so a sequence of length
`n` has rows 0 through `n`.

## Information quantities

These are intentionally separate quantities:

- **HMM predictive entropy** is the binary Shannon entropy of `q_t` and
  measures next-observable uncertainty under the selected HMM:
  `H(q) = -sum_x q(x) log2 q(x)`.
- **Observed-symbol Shannon entropy** is descriptive entropy of empirical
  `A`/`B` frequencies in the entered sequence. It is not an HMM prediction.
- **Target surprisal** is the self-information of a specified candidate or an
  optional user-selected actual next target: `I(x) = -log2 q(x)`.

The convention is `0 log2 0 = 0` for entropy and `surprisal(0) = infinity`.
For a binary distribution, `H` is in `[0, 1]` bits. Deterministic modal ties
choose observable variable 1, the first configured observable (index 0).

The optional actual-target assessment is separate from the internal
observed-next value attached to each nonfinal prefix for presentation/export.
It evaluates only the user-selected target against the final prediction.

## Results, graph, and exports

After the explicit **Calculate entropy** action, the application renders a
final summary and an every-prefix table. The table is the exact-value fallback
for the Plotly line-and-marker graph of predictive entropy by context depth.
The graph's y-axis is fixed to 0--1 bits. Display values, chart hover values,
and CSV floats use 12 decimal places.

Available downloads are:

- **Model preset JSON**: strict UTF-8 schema version 1 containing
  `schema_version`, `preset_name`, state and observable labels, `initial`,
  `transition`, and `emission`. It contains model parameters only, not the
  sequence or optional target.
- **Prefix CSV**: one deterministic row for every depth from 0 through the
  full prefix. It includes predictive probabilities, entropy, candidate
  surprisals, observed/next target fields, posterior, and next-hidden values.
- **Candidate-summary CSV**: one row with reproducibility fields (sequence
  ID, preset name, labels, sequence, all model probabilities, sequence length,
  and observed entropy) plus final prediction fields. Its actual-target
  symbol, probability, surprisal, and classification fields are optional and
  are empty when no actual target was selected.

For example, a preset has this shape:

```json
{
  "schema_version": 1,
  "preset_name": "hand-calculated model",
  "state_labels": ["State 1", "State 2"],
  "observable_labels": ["A", "B"],
  "initial": [0.6, 0.4],
  "transition": [[0.7, 0.3], [0.2, 0.8]],
  "emission": [[0.9, 0.1], [0.2, 0.8]]
}
```

## Reusable Python core

Use only the package-level public API for a programmatic calculation:

```python
from binary_entropy import BinaryHMM, BinaryLabels, analyze_sequence, parse_sequence

labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B"),
)
model = BinaryHMM(
    labels=labels,
    initial=[0.6, 0.4],
    transition=[[0.7, 0.3], [0.2, 0.8]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)
sequence = parse_sequence("A, B, B, A, A, A, B", labels)
analysis = analyze_sequence(model, sequence)
final_prediction = analysis.rows[-1].predictive
```

## Hand-worked reference model

The included example uses:

```text
pi = [0.6, 0.4]
T = [[0.7, 0.3], [0.2, 0.8]]
E = [[0.9, 0.1], [0.2, 0.8]]
```

For the hand-worked initial row, `q_0 = pi @ E = [0.62, 0.38]`. After observing `A`, the
unnormalized posterior is `[0.54, 0.08]`, its likelihood is `0.62`, and the
posterior is `[27/31, 4/31]`. The next hidden distribution is
`[0.635483870967742, 0.364516129032258]`; the following prediction is
`[0.644838709677419, 0.355161290322581]`; its entropy is
`0.938593249062606` bits.

For `A,B,B,A,A,A,B`, the observed-symbol Shannon entropy for four `A` values
and three `B` values is `0.985228136034251` bits.

The following is the complete program output profile at canonical display
precision. Context is the consumed prefix; surprisals are hypothetical values
for the next observable at that depth.

| Depth | Context | P(A) | P(B) | Predicted target | Entropy (bits) | Surprisal A (bits) | Surprisal B (bits) |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | `(empty prefix)` | 0.620000000000 | 0.380000000000 | A | 0.958042022226 | 0.689659879388 | 1.395928676331 |
| 1 | `A` | 0.644838709677 | 0.355161290323 | A | 0.938593249063 | 0.632989743417 | 1.493453746597 |
| 2 | `A, B` | 0.402624886467 | 0.597375113533 | B | 0.972465360818 | 1.312491746094 | 0.743290958227 |
| 3 | `A, B, B` | 0.356959602256 | 0.643040397744 | B | 0.940130453574 | 1.486167283721 | 0.637018720019 |
| 4 | `A, B, B, A` | 0.537870628970 | 0.462129371030 | A | 0.995857852439 | 0.894668883793 | 1.113631310744 |
| 5 | `A, B, B, A, A` | 0.622673518217 | 0.377326481783 | A | 0.956131895281 | 0.683452170979 | 1.406114738984 |
| 6 | `A, B, B, A, A, A` | 0.645461975872 | 0.354538024128 | A | 0.938055727085 | 0.631595985934 | 1.495987730386 |
| 7 | `A, B, B, A, A, A, B` | 0.402822877316 | 0.597177122684 | B | 0.972577939805 | 1.311782474967 | 0.743769196699 |

`tests/fixtures/hand_sequence.json` is the authoritative machine-readable
source for this model and full-depth numeric profile. The fixture-backed tests
in `tests/unit/test_filtering_analysis.py` reproduce the first-observation
derivation and every filtering value; `tests/ui/test_results.py` checks the
all-depth visible numeric table. Tests do not assert this README's prose.

## Tests and strict gates

Run the focused numerical checks or the full release gates:

```bash
uv run pytest tests/unit/test_filtering_analysis.py tests/ui/test_results.py
uv run pytest
uv run ruff check .
uv run basedpyright
```

The suite also contains Streamlit `AppTest` integration coverage for explicit
calculation, validation, stale results, JSON preset import, and optional
actual-target assessment.

## Limitations

- The first release UI is fixed at two hidden states and two observables.
- Model parameters are supplied by the user; the application does not learn
  or train an HMM.
- The first release analyzes one sequence at a time and has no batch mode.
- Results are conditional model calculations and make no causal or
  statistical-validity claim beyond that calculation.
- Numerical calculations use float64 arithmetic.
