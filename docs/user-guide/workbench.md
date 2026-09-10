# Streamlit Workbench

The **Streamlit Workbench** is an interactive local interface for analyzing binary sequences and running stimulus searches.

## Overview

The workbench has two main modes:

1. **Analyzer**: Analyze sequences with different methods
2. **Stimulus Search**: Generate and rank candidate sequences

## Running the Workbench

### Start

```bash
uv run streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

### Stop

Press `Ctrl+C` in terminal or close browser tab

## Analyzer Mode

### Overview

The Analyzer lets you:
- Upload or paste binary sequences
- Select analysis methods
- Compare results
- Export findings

### Workflow

1. **Configure** sidebar settings
2. **Input data**: Paste, upload file, or single sequence
3. **Select methods**: VMM, Markov, HMM, Shannon
4. **Calculate**: Click "Calculate"
5. **Inspect**: View results, charts, tables
6. **Export**: Download CSV or JSON

### Sidebar Configuration

#### Observable Labels

Set the names for your two symbols:

```
Observable A: [text input]
Observable B: [text input]
```

Default: "A", "B"

#### State Labels (for HMM)

Set names for hidden states:

```
Hidden State A: [text input]
Hidden State B: [text input]
```

Default: "State 1", "State 2"

#### VMM Configuration

```
Smoothing Method: [KT / MLE / Additive]
Minimum Support: [slider 1-5]
Analysis Scope: [Pooled / Per-Sequence]
```

#### Markov Configuration

```
Smoothing Alpha: [slider 0.0-1.0]
Prediction Mode: [Fixed Model / Cumulative Prefix]
Analysis Scope: [Pooled / Per-Sequence]
```

#### HMM Configuration

```
Model Preset: [dropdown of available presets]
```

### Data Input

Three input methods:

#### 1. Single Sequence

```
Paste single sequence: [text input]
Format: A, B, B, A, A or A B B A A
```

#### 2. Batch (Multiple Lines)

```
Paste multiple sequences (one per line):
A, B, B, A, A
B, A, A, B
A, A, B, B, A
```

#### 3. File Upload

```
Upload TXT or CSV file
```

For CSV, specify column mappings:

```
ID Column: [dropdown]
Sequence Column: [dropdown]
Target Column (optional): [dropdown]
```

### Results View

After calculation, results appear in tabs:

#### Summary Tab

- Overall statistics
- Method comparison table
- Key findings

#### Detailed Results Tab

- Per-record predictions
- Probability distributions
- Entropy measurements
- Depth evidence (VMM)
- Transition matrix (Markov)

#### Charts Tab

- Prediction distributions
- Entropy curves
- Method comparison plots

#### Export Tab

```
Download Results:
- [CSV] Export as CSV
- [JSON] Export as JSON
```

## Stimulus Search Mode

### Overview

The Stimulus Search mode lets you:
- Define search criteria
- Generate candidate sequences
- Inspect and rank results
- Optionally match complements
- Assign targets and export

### Workflow

1. **Configure search**: Sequence length, candidate limit, seed
2. **Define constraints**: Hard filters (optional)
3. **Define preferences**: Soft ranking (optional)
4. **Run search**: Click "Search"
5. **Inspect results**: View top candidates
6. **Optionally match**: Create complement pairs
7. **Optionally assign**: Add observed targets
8. **Export**: Download final stimuli

### Configuration

#### Search Parameters

```
Sequence Length: [slider 1-32]
Desired Stimuli: [number input]
Candidate Limit: [slider 1-5000]
Seed: [number input, default=current timestamp]
Minimum Support (VMM): [slider 1-5]
```

#### Hard Constraints (Expand Section)

```
Predicted Symbol: [A / B / Either]
Predicted Probability: [min/max range]
Predictive Entropy: [min/max range]
Effective Depth: [min/max range]
Context Support: [min/max range]
A Count: [min/max range]
A Proportion: [min/max range]
Switches: [min/max range]
Switch Rate: [min/max range]
Longest A Run: [min/max range]
Longest B Run: [min/max range]
Longest Run: [min/max range]
```

Each constraint has:
```
☐ Enable [min input] [max input]
```

#### Soft Preferences (Expand Section)

```
Preference 1: [metric dropdown] [target input] [weight slider]
+ Add Preference
```

Available metrics:
- Predicted Probability
- Predictive Entropy
- Effective Depth
- Context Support
- A Proportion
- Switch Rate
- Longest Run

### Results View

#### Status

```
Status: [COMPLETE / PARTIAL]
Reason: [if partial, why]
```

#### Statistics

```
Evaluated Candidates: 1000
Accepted Candidates: 45
Selected Candidates: 20
```

#### Constraint Violations (if PARTIAL)

```
Constraint | Rejected | Frequency
predicted_probability | 350 | 35.0%
predictive_entropy | 250 | 25.0%
...
```

**Interpretation:** Most violated constraints are too strict

#### Top Candidates

Table showing:

```
Rank | Stimulus ID | Sequence | A | B | P(A) | Entropy | Depth | Score
1    | stim-0001   | A B A... | 6 | 6 | 0.67 | 0.998   | 2     | 0.142
2    | stim-0002   | B A B... | 6 | 6 | 0.71 | 0.891   | 1     | 0.156
...
```

Click row to expand and see full details

#### Candidate Details

For each candidate:

```
Sequence: [A B A B A B A B A B A B]
Predicted Target: [A / B]
P(A): 0.667
P(B): 0.333
Predictive Entropy: 0.918 bits
Effective Depth: 2
Context Used: [A A]
Context Support: 5
A Count: 6
B Count: 6
Switches: 5
Longest Run: 2
Rank Score: 0.156
```

### Advanced Features

#### Create Complements

```
☐ Generate Bitwise Complements
  Match within tolerances:
  - Entropy Tolerance: [slider]
  - Proportion Tolerance: [slider]
```

Results:

```
Matched Pairs: 15
Unmatched: 5 originals, 10 complements
```

#### Assign Targets

```
Target Assignment:
○ Balanced (Expected/Unexpected)
○ Expected (Match Prediction)
○ Unexpected (Oppose Prediction)
○ Always A
○ Always B

Seed: [number input]
```

Click "Assign Targets" to add observed next symbols

#### Review Assignments

Table showing:

```
Stimulus ID | Sequence | Predicted | Target | Congruency
stim-0001   | A B A... | A         | B      | Unexpected
stim-0002   | B A B... | B         | B      | Expected
...
```

### Export

#### Download Options

```
[CSV] Candidates
[CSV] Scientific Evidence
[CSV] Experiment-Ready
[JSON] Configuration
```

**Files:**
- **Candidates.csv**: All candidate stimuli with properties
- **Scientific.csv**: VMM depth-evidence for all candidates
- **Experiment-Ready.csv**: Final assigned stimuli
- **Config.json**: Full reproducible configuration

## Tips and Tricks

### Analyzer Mode

1. **Start simple**: Use default settings, then customize
2. **Compare methods**: Run with multiple methods to see differences
3. **Inspect depth evidence**: Check how VMM backs off
4. **Use targets**: Assess with optional observed outcomes
5. **Export for analysis**: Use CSV for statistical analysis

### Stimulus Search Mode

1. **Start with no constraints**: See overall acceptance rate
2. **Add constraints incrementally**: See impact of each
3. **Check violations**: Relax the most-violated constraints
4. **Use preferences sparingly**: Few preferences with high weights
5. **Verify reproducibility**: Same seed produces same results
6. **Test small first**: Try sequence_length=8 before 12

### General

1. **Use sidebar**: Pin/unpin for easier access
2. **Clear cache**: Use Streamlit menu if getting stale results
3. **Check browser console**: Debug JavaScript issues
4. **Export everything**: Save CSV/JSON for reproducibility
5. **Iterate**: Run multiple times with different configs

## Troubleshooting

### "Stale results detected"

**Cause:** Configuration changed but results weren't recalculated

**Solution:** Click "Calculate" or "Search" to rerun

### "No results"

**Cause:** Constraints too strict or error in configuration

**Solution:** Check error message, relax constraints, rerun

### "Upload failed"

**Cause:** File format or encoding issue

**Solution:** Ensure UTF-8 encoding, CSV has headers, TXT has one sequence per line

### "Search is slow"

**Cause:** Large sequence_length or high candidate_limit

**Solution:** Reduce sequence_length, reduce candidate_limit, or increase minimum_support

### Session state issues

**Solution:** Press F5 to reload page and reset state

## See Also

- **[Analyzer Mode](user-guide/overview.md)**: Detailed method descriptions
- **[Stimulus Search](stimulus-search.md)**: Search configuration and results
- **[Input Formats](input-formats.md)**: Data format requirements
- **[Examples](../examples/vmm-analysis.md)**: Runnable examples
