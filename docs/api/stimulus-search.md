# Stimulus Search API

Synthetic sequence generation and optimization.

## Main Entry Point

### search_stimulus

Generate synthetic sequences satisfying constraints and optimized for preferences:

```python
from bspe import search_stimulus, StimulusConstraints, StimulusPreferences

result = search_stimulus(
    labels=labels,
    constraints=StimulusConstraints(
        minimum_length=20,
        maximum_length=30,
        entropy_bits_minimum=0.3,
        entropy_bits_maximum=0.7,
    ),
    preferences=StimulusPreferences(
        target_entropy_bits=0.5,
        require_transition_cross=True,
    ),
    max_candidates=100,
)
```

**Parameters:**
- `labels`: BinaryLabels for observable names
- `constraints`: StimulusConstraints (hard requirements)
- `preferences`: StimulusPreferences (optimization goals)
- `max_candidates`: Maximum candidates to generate (default: 1000)

**Returns:** StimulusSearchResult with ranked candidates

## Configuration Classes

### StimulusConstraints

Hard constraints that all generated sequences must satisfy:

```python
from bspe import StimulusConstraints

constraints = StimulusConstraints(
    minimum_length=20,
    maximum_length=30,
    entropy_bits_minimum=0.0,
    entropy_bits_maximum=1.0,
    observable_a_min=0.3,
    observable_a_max=0.7,
    transition_same_min=0.0,
    transition_same_max=1.0,
    first_observable_index=None,  # Optional: 0 or 1
    last_observable_index=None,   # Optional: 0 or 1
)
```

**Fields:**

- `minimum_length` (int): Minimum sequence length
- `maximum_length` (int): Maximum sequence length
- `entropy_bits_minimum` (float): Minimum Shannon entropy (0–1 bits)
- `entropy_bits_maximum` (float): Maximum Shannon entropy (0–1 bits)
- `observable_a_min` (float): Minimum frequency of observable A (0–1)
- `observable_a_max` (float): Maximum frequency of observable A (0–1)
- `transition_same_min` (float): Minimum frequency of same-value transitions (0–1)
- `transition_same_max` (float): Maximum frequency of same-value transitions (0–1)
- `first_observable_index` (0|1|None): Require first observable to be specific value
- `last_observable_index` (0|1|None): Require last observable to be specific value

### StimulusPreferences

Soft constraints (optimization goals) used for ranking:

```python
from bspe import StimulusPreferences

preferences = StimulusPreferences(
    target_entropy_bits=0.5,  # Prefer entropy near this value
    require_transition_a=True,  # Prefer sequences with A→A transitions
    require_transition_b=False,  # Prefer sequences with B→B transitions
    require_transition_cross=True,  # Prefer cross transitions (A→B or B→A)
    target_observable_index=0,  # Optional: prefer sequences with observable 0
)
```

**All fields optional; only specified preferences affect ranking.**

## Results

### search_stimulus Result

Returns a `StimulusSearchResult`:

```python
result = search_stimulus(labels, constraints, preferences)

candidates = result.candidates  # List[StimulusCandidate]
total_generated = result.total_candidates_generated
valid_count = result.valid_candidates_count
```

### StimulusCandidate

Individual candidate from search results:

```python
candidate = result.candidates[0]

sequence = candidate.sequence  # String of observables
entropy_bits = candidate.entropy_bits  # 0–1 bits
ranking_score = candidate.ranking_score  # Higher is better
```

## Constraint Validation

### Constraint Validation Errors

**InvalidStimulusSearchConfigurationError**

Raised when constraints are contradictory or infeasible:

```python
from bspe import search_stimulus, InvalidStimulusSearchConfigurationError

try:
    result = search_stimulus(
        labels,
        StimulusConstraints(
            entropy_bits_minimum=0.95,
            entropy_bits_maximum=0.1,  # Contradictory!
        ),
        preferences,
    )
except InvalidStimulusSearchConfigurationError as e:
    print(f"Invalid constraint: {e}")
```

## Usage Patterns

### Basic Search

```python
from bspe import search_stimulus, StimulusConstraints, StimulusPreferences

result = search_stimulus(
    labels=labels,
    constraints=StimulusConstraints(minimum_length=20, maximum_length=30),
    preferences=StimulusPreferences(target_entropy_bits=0.5),
    max_candidates=100,
)

# Access top candidates
for candidate in result.candidates[:10]:
    print(f"{candidate.sequence} (entropy: {candidate.entropy_bits:.3f})")
```

### Boundary Constraints

```python
# Must start with 0, end with 1
constraints = StimulusConstraints(
    minimum_length=15,
    first_observable_index=0,
    last_observable_index=1,
)

result = search_stimulus(labels, constraints, StimulusPreferences())
```

### Transition Preferences

```python
# Prefer alternating pattern
constraints = StimulusConstraints(
    minimum_length=20,
    transition_same_min=0.0,
    transition_same_max=0.2,  # Few same-value transitions
)

preferences = StimulusPreferences(
    require_transition_cross=True,  # Must have cross transitions
)

result = search_stimulus(labels, constraints, preferences)
```

### Observable Distribution

```python
# Generate sequences with 70% A, 30% B
constraints = StimulusConstraints(
    minimum_length=25,
    observable_a_min=0.65,
    observable_a_max=0.75,
)

result = search_stimulus(labels, constraints, StimulusPreferences())
```

## Ranking and Selection

Candidates are ranked by:

1. **Constraint satisfaction** (binary: satisfied or not)
2. **Preference score** (soft optimization)
   - Distance to `target_entropy_bits`
   - Satisfaction of transition/observable preferences

Select candidates by:

```python
# Top N by ranking score
top_10 = result.candidates[:10]

# Filter by specific entropy
high_entropy = [c for c in result.candidates if c.entropy_bits > 0.9]

# Random selection for comparison
import random
sample = random.sample(result.candidates, k=5)
```

## Export Formats

### JSON Export

```python
import json

export_data = [
    {
        "sequence": c.sequence,
        "entropy_bits": c.entropy_bits,
        "ranking_score": c.ranking_score,
        "constraints_satisfied": True,
    }
    for c in result.candidates[:50]
]

with open("stimuli.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

### CSV Export

```python
import csv

with open("stimuli.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sequence", "entropy_bits", "ranking_score"])
    for candidate in result.candidates[:50]:
        writer.writerow([
            candidate.sequence,
            candidate.entropy_bits,
            candidate.ranking_score,
        ])
```

## Performance Considerations

### Scaling with Constraints

- **No constraints**: Fast (100K+ candidates possible)
- **Length only**: Fast (50K+ candidates)
- **Entropy bounds**: Moderate (5K–20K candidates typical)
- **Transition constraints**: Slower (1K–5K candidates typical)
- **Multiple constraints**: Slowest (100–1K candidates typical)

### Optimization Tips

1. **Reduce max_candidates** if search is slow
2. **Simplify constraints** (fewer hard constraints = faster)
3. **Use only needed preferences** (fewer soft constraints = faster)
4. **Batch searches** if generating many stimulus sets

## See Also

- [Stimulus Search User Guide](../user-guide/stimulus-search.md)
- [Examples: Stimulus Search](../examples/stimulus-search.md)
- [API: Analysis Methods](analysis-methods.md)
- [Glossary](../reference/glossary.md)
