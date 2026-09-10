# Stimulus Search

Stimulus Search generates and ranks binary sequences matching specific prediction and structural criteria using bounded deterministic search.

## Overview

### What Stimulus Search Does

1. **Generates** a seeded sample of candidate binary sequences
2. **Analyzes** each candidate using VMM
3. **Filters** by hard constraints (never relaxed)
4. **Ranks** by soft preferences (weighted distance)
5. **Optionally matches** complements and assigns targets

### When to Use Stimulus Search

- **Experimental design**: Create sequences with known properties
- **Exploration**: Find sequences matching specific criteria
- **Reproducibility**: Seed ensures exact repeatability
- **Quality control**: Verify stimulus properties

## Configuration

### StimulusSearchConfig

```python
from bspe import (
    StimulusSearchConfig,
    VMMConfig,
    StimulusConstraints,
    SoftPreference,
    PreferenceMetric,
)

config = StimulusSearchConfig(
    sequence_length=12,           # 12-symbol sequences
    desired_stimuli=20,           # Want 20 candidates
    seed=2026,                    # Reproducible randomness
    candidate_limit=1000,         # Search up to 1000 candidates
    vmm_config=VMMConfig(minimum_support=2),
    constraints=StimulusConstraints(),
    preferences=(),
)
```

### Parameters

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `sequence_length` | int | 1–32 | Required | Symbol count per sequence |
| `desired_stimuli` | int | 1–5000 | Required | Target number of accepted candidates |
| `seed` | int | 0–2^63-1 | Required | Random seed for reproducibility |
| `candidate_limit` | int | 1–5000 | 1000 | Max candidates to search |
| `vmm_config` | VMMConfig | N/A | Default | VMM analysis config |
| `constraints` | StimulusConstraints | N/A | Empty | Hard filters |
| `preferences` | tuple[SoftPreference] | N/A | Empty | Soft ranking criteria |

## Hard Constraints

Hard constraints are **never relaxed** — a candidate either passes or fails.

### StimulusConstraints

```python
from bspe import StimulusConstraints, InclusiveRange, PredictedSymbol

constraints = StimulusConstraints(
    # Predicted symbol
    predicted_symbol=PredictedSymbol.A,  # Must predict A
    
    # Probability range
    predicted_probability=InclusiveRange(min=0.6, max=0.9),
    
    # Entropy range
    predictive_entropy=InclusiveRange(min=0.3, max=0.8),
    
    # Context depth
    effective_depth=InclusiveRange(min=1, max=None),
    
    # Support count
    context_support=InclusiveRange(min=3, max=None),
    
    # Symbol counts
    a_count=InclusiveRange(min=4, max=8),
    b_count=InclusiveRange(min=4, max=8),
    
    # Proportions
    a_proportion=InclusiveRange(min=0.4, max=0.6),
    b_proportion=InclusiveRange(min=0.4, max=0.6),
    
    # Transitions (switches)
    switches=InclusiveRange(min=2, max=5),
    switch_rate=InclusiveRange(min=0.15, max=0.4),
    
    # Runs (longest same symbol)
    longest_a_run=InclusiveRange(min=None, max=4),
    longest_b_run=InclusiveRange(min=None, max=4),
    longest_run=InclusiveRange(min=None, max=4),
)
```

### InclusiveRange

```python
from bspe import InclusiveRange

# Both bounds
range1 = InclusiveRange(min=0.6, max=0.9)

# Only lower
range2 = InclusiveRange(min=0.6, max=None)

# Only upper
range3 = InclusiveRange(min=None, max=0.9)

# Always pass (rarely used)
range4 = InclusiveRange(min=None, max=None)
```

## Soft Preferences

Soft preferences **rank** accepted candidates by distance to target.

### SoftPreference

```python
from bspe import SoftPreference, PreferenceMetric

preferences = (
    # Prefer entropy close to 0.5 bits (weight=2.0 means important)
    SoftPreference(
        metric=PreferenceMetric.PREDICTIVE_ENTROPY,
        target=0.5,
        weight=2.0,
    ),
    
    # Prefer A proportion close to 0.5 (balanced)
    SoftPreference(
        metric=PreferenceMetric.A_PROPORTION,
        target=0.5,
        weight=1.0,
    ),
    
    # Prefer switch rate close to 0.3
    SoftPreference(
        metric=PreferenceMetric.SWITCH_RATE,
        target=0.3,
        weight=1.0,
    ),
)
```

### Ranking

Preferences are combined into single **rank score**:

```
rank_score = Σ weight * |metric - target|
```

Lower score = better ranking

**Example:**
```
Preference 1: weight=2.0, target=0.5
  Candidate A: entropy=0.4 → distance=0.1 → contribution=0.2
  Candidate B: entropy=0.7 → distance=0.2 → contribution=0.4
  
Preference 2: weight=1.0, target=0.5
  Candidate A: proportion=0.48 → distance=0.02 → contribution=0.02
  Candidate B: proportion=0.52 → distance=0.02 → contribution=0.02

Rank Scores:
  Candidate A: 0.2 + 0.02 = 0.22 (better)
  Candidate B: 0.4 + 0.02 = 0.42 (worse)
```

## Running a Search

### Basic Search

```python
from bspe import search_stimuli, StimulusSearchConfig, VMMConfig

config = StimulusSearchConfig(
    sequence_length=12,
    desired_stimuli=20,
    seed=2026,
    candidate_limit=1000,
    vmm_config=VMMConfig(minimum_support=2),
)

result = search_stimuli(config)

print(f"Status: {result.status}")
print(f"Evaluated: {result.evaluated_count}")
print(f"Accepted: {result.accepted_count}")
print(f"Selected: {result.selected_count}")
```

### With Constraints

```python
from bspe import (
    StimulusConstraints,
    InclusiveRange,
    PredictedSymbol,
)

constraints = StimulusConstraints(
    predicted_symbol=PredictedSymbol.A,
    predicted_probability=InclusiveRange(0.6, 0.9),
    predictive_entropy=InclusiveRange(0.3, 0.8),
)

config = StimulusSearchConfig(
    sequence_length=12,
    desired_stimuli=20,
    seed=2026,
    candidate_limit=1000,
    vmm_config=VMMConfig(minimum_support=2),
    constraints=constraints,
)

result = search_stimuli(config)
```

### With Preferences

```python
from bspe import SoftPreference, PreferenceMetric

preferences = (
    SoftPreference(PreferenceMetric.PREDICTIVE_ENTROPY, 0.5, weight=2.0),
    SoftPreference(PreferenceMetric.A_PROPORTION, 0.5, weight=1.0),
)

config = StimulusSearchConfig(
    sequence_length=12,
    desired_stimuli=20,
    seed=2026,
    candidate_limit=1000,
    vmm_config=VMMConfig(minimum_support=2),
    constraints=constraints,
    preferences=preferences,
)

result = search_stimuli(config)
```

## Results

### SearchResult

```python
result = search_stimuli(config)

# Status
print(result.status)  # SearchStatus.COMPLETE or SearchStatus.PARTIAL
print(result.partial_reason)  # Why partial (if applicable)

# Counts
print(result.evaluated_count)  # Candidates analyzed
print(result.accepted_count)   # Passed hard constraints
print(result.selected_count)   # Top candidates returned

# Candidates
print(len(result.accepted))   # All accepted candidates
print(len(result.selected))   # Top selected candidates
print(result.selected[:5])    # Top 5

# Violations
for violation in result.violations:
    print(f"{violation.name}: {violation.frequency:.1%}")
```

### Candidate Properties

```python
candidate = result.selected[0]

print(candidate.stimulus_id)           # Unique ID
print(candidate.sequence_str)          # Human-readable
print(candidate.predicted_target_index)  # 0, 1, or None
print(candidate.probability_a)
print(candidate.probability_b)
print(candidate.predictive_entropy_bits)
print(candidate.effective_context_depth)
print(candidate.rank_score)            # Score (lower is better)
```

## Constraint Violations

### Understanding Violations

```python
for violation in result.violations:
    print(f"Constraint: {violation.name}")
    print(f"  Rejected: {violation.count} candidates")
    print(f"  Frequency: {violation.frequency:.1%}")
```

**Interpretation:**
- **High frequency (>20%)**: Constraint is very strict
- **Medium frequency (5–20%)**: Constraint is selective
- **Low frequency (<5%)**: Constraint is mild

### Relaxing Constraints

If search returns PARTIAL:

```python
# Try relaxing the strictest constraint
print("Most rejecting constraint:")
print(max(result.violations, key=lambda v: v.frequency))

# Relax it and retry
constraints = StimulusConstraints(
    predicted_probability=InclusiveRange(0.5, 1.0),  # Wider range
    predictive_entropy=InclusiveRange(0.3, 0.8),
)

config = replace(config, constraints=constraints)
result = search_stimuli(config)
```

## Advanced Features

### Complement Generation

Generate bitwise complements (0↔1):

```python
from bspe import create_complement_candidates

candidates = result.accepted
complements = create_complement_candidates(candidates, config)

print(f"Original: {len(candidates)} candidates")
print(f"Complements: {len(complements)} candidates")
```

**Use:** Create paired stimuli with opposite predictions

### Complement Matching

Match candidates with opposite predictions:

```python
from bspe import match_stimuli, MatchTolerances

tolerances = MatchTolerances(
    predictive_entropy_tolerance=0.1,
    a_proportion_tolerance=0.1,
)

all_candidates = result.accepted + complements
pairs = match_stimuli(all_candidates, tolerances)

print(f"Matched pairs: {len(pairs.matched)}")
```

### Target Assignment

Assign balanced observed next symbols:

```python
from bspe import assign_targets

# Balanced: Equal expected/unexpected
assigned = assign_targets(result.accepted, seed=2026)

for candidate in assigned:
    print(f"Target: {candidate.actual_target_index}")
    print(f"Congruency: {candidate.congruency}")
```

## Exports

### CSV Export

```python
from bspe import stimulus_candidate_csv

csv = stimulus_candidate_csv(result)
with open("candidates.csv", "w") as f:
    f.write(csv)
```

### JSON Export

```python
from bspe import stimulus_generator_config_json

json_str = stimulus_generator_config_json(
    result,
    match_tolerances=tolerances,
    assignment_seed=2026,
    final_candidates=assigned,
)
with open("config.json", "w") as f:
    f.write(json_str)
```

## Performance

### Typical Times

| Config | Time |
|--------|------|
| sequence_length=8, limit=256 | ~50ms |
| sequence_length=12, limit=1000 | ~1s |
| sequence_length=16, limit=5000 | ~10s |

### Optimization

```python
# Faster: Shorter sequences, higher min_support
config = StimulusSearchConfig(
    sequence_length=10,
    candidate_limit=500,
    vmm_config=VMMConfig(minimum_support=3),
)

# Slower: Longer sequences, lower min_support
config = StimulusSearchConfig(
    sequence_length=18,
    candidate_limit=5000,
    vmm_config=VMMConfig(minimum_support=1),
)
```

## See Also

- **[Variable-order Markov](methods/vmm.md)**: VMM analysis used for search
- **[Workbench](workbench.md)**: Interactive interface for search
- **[Examples](../examples/stimulus-search.md)**: Runnable examples
- **[Glossary](../reference/glossary.md)**: Constraint definitions
