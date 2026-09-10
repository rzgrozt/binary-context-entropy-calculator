# User Guide Overview

Welcome to the **bspe** User Guide! This section covers everything you need to know about analyzing binary sequences and using the Streamlit workbench.

## What You'll Learn

- **[Input Formats](input-formats.md)**: How to prepare and parse your data
- **[Methods](methods/vmm.md)**: Detailed explanation of each analysis method
- **[Stimulus Search](stimulus-search.md)**: Generating and ranking candidate sequences
- **[Workbench](workbench.md)**: Using the interactive Streamlit interface

## Quick Navigation

### By Task

**I want to...**

- **Analyze a sequence**: See [Quick Start](../getting-started/quick-start.md)
- **Understand VMM**: See [Variable-order Markov](methods/vmm.md)
- **Compare methods**: See [Methods Overview](methods/vmm.md)
- **Search for stimuli**: See [Stimulus Search](stimulus-search.md)
- **Use the workbench**: See [Workbench](workbench.md)

### By Method

- **[Variable-order Markov (VMM)](methods/vmm.md)**: Context-aware prediction with automatic suffix backoff
- **[First-order Markov](methods/markov.md)**: Transition-matrix baseline
- **[Hidden Markov Model (HMM)](methods/hmm.md)**: Configured two-state model
- **[Observed Shannon Entropy](methods/shannon.md)**: Empirical symbol frequency analysis

## Key Concepts

### Boundary Preservation
Records are never concatenated or merged. Each sequence maintains its own context counts, transition counts, and filtering state. This ensures **scientific validity** and **reproducibility**.

### Explicit Separation of Concerns
- **Prediction**: VMM, Markov, HMM fit and predict
- **Description**: Shannon entropy analyzes empirical frequencies
- **Search**: Stimulus generation generates candidates independently
- **Target Assignment**: Separate from fitting (never influences model)

### Determinism
All operations produce deterministic, reproducible results:
- Seeded randomness (same seed → same results)
- Sorted output (consistent ordering)
- Reproducible exports (full configuration saved)

## Workflow

### Typical Analysis Workflow

```
1. Prepare Data
   ↓ (Input Formats)
   
2. Parse Sequences
   ↓ (parse_manual_batch, parse_csv_batch, etc.)
   
3. Configure Analysis
   ↓ (VMMConfig, MarkovAnalysisRequest, etc.)
   
4. Analyze Dataset
   ↓ (analyze_dataset)
   
5. Inspect Results
   ↓ (probability, entropy, depth evidence, etc.)
   
6. Export Results
   (CSV, JSON)
```

### Typical Stimulus Search Workflow

```
1. Define Constraints
   ↓ (StimulusConstraints)
   
2. Define Preferences
   ↓ (SoftPreference)
   
3. Configure Search
   ↓ (StimulusSearchConfig)
   
4. Run Search
   ↓ (search_stimuli)
   
5. Inspect Results
   ↓ (violations, candidates, rank scores)
   
6. Optional: Match & Assign
   ↓ (create_complement_candidates, match_stimuli, assign_targets)
   
7. Export Results
   (CSV, JSON)
```

## Method Comparison

| Method | Type | Use Case | Speed | Data Needed |
|--------|------|----------|-------|------------|
| **VMM** | Prediction | Detect patterns | Medium | Moderate |
| **Markov** | Prediction | Baseline comparison | Fast | Low |
| **HMM** | Prediction | Configured model | Fast | None (model provided) |
| **Shannon** | Description | Empirical summary | Fastest | Low |

## Common Configurations

### Exploration (Fast, Less Stable)
```python
VMMConfig(minimum_support=1)
```
- Accepts all contexts
- Many contexts, noisy predictions
- Good for exploratory analysis

### Production (Balanced)
```python
VMMConfig(minimum_support=2)
```
- Requires at least 2 observations per context
- Fewer contexts, more stable
- Recommended default

### Conservative (Slow, Most Stable)
```python
VMMConfig(minimum_support=3)
```
- Requires at least 3 observations per context
- Fewest contexts, most stable
- Good for small datasets

## Data Sizes

### For Interactive Use
- **Sequences**: 1–100 records
- **Sequence length**: 8–16 symbols
- **Expected time**: < 1 second

### For Batch Processing
- **Sequences**: 100–1000 records
- **Sequence length**: 12–32 symbols
- **Expected time**: 1–60 seconds

### For Research
- **Sequences**: 100–10000 records
- **Sequence length**: 12–32 symbols
- **Expected time**: 10–600 seconds

## File Structure

```
docs/
├── input-formats.md           # How to prepare data
├── methods/
│   ├── vmm.md                 # Variable-order Markov
│   ├── markov.md              # First-order Markov
│   ├── hmm.md                 # Hidden Markov Model
│   └── shannon.md             # Observed Shannon Entropy
├── stimulus-search.md         # Generating candidates
└── workbench.md               # Interactive interface
```

## Next Steps

- **[Input Formats](input-formats.md)**: Learn how to prepare your data
- **[Methods](methods/vmm.md)**: Deep dive into each analysis method
- **[Stimulus Search](stimulus-search.md)**: Generate and rank candidate sequences
- **[Workbench](workbench.md)**: Use the interactive interface
