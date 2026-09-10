# Stimulus Search Examples

This page demonstrates bspe's **stimulus search** capability, which generates synthetic sequences that satisfy constraints and optimize preferences.

## Example 1: Simple Entropy Bounds

Generate sequences with target entropy ranges:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))

# Generate sequences with low entropy (predictable)
constraints_low = StimulusConstraints(
    minimum_length=20,
    maximum_length=30,
    entropy_bits_minimum=0.0,
    entropy_bits_maximum=0.3,
)

preferences_low = StimulusPreferences(
    target_entropy_bits=0.1,
)

result = search_stimulus(
    labels=labels,
    constraints=constraints_low,
    preferences=preferences_low,
    max_candidates=100,
)

print(f"Generated {len(result.candidates)} low-entropy sequences:")
for i, candidate in enumerate(result.candidates[:3]):
    print(f"  {i+1}. {candidate.sequence} "
          f"(entropy: {candidate.entropy_bits:.3f} bits)")

# Generate sequences with high entropy (unpredictable)
constraints_high = StimulusConstraints(
    minimum_length=20,
    maximum_length=30,
    entropy_bits_minimum=0.9,
    entropy_bits_maximum=1.0,
)

preferences_high = StimulusPreferences(
    target_entropy_bits=1.0,
)

result = search_stimulus(
    labels=labels,
    constraints=constraints_high,
    preferences=preferences_high,
    max_candidates=100,
)

print(f"\nGenerated {len(result.candidates)} high-entropy sequences:")
for i, candidate in enumerate(result.candidates[:3]):
    print(f"  {i+1}. {candidate.sequence} "
          f"(entropy: {candidate.entropy_bits:.3f} bits)")
```

**Output** (illustrative):
```
Generated 47 low-entropy sequences:
  1. 00000000000000000000 (entropy: 0.000 bits)
  2. 11111111111111111111 (entropy: 0.000 bits)
  3. 00000001000000000000 (entropy: 0.209 bits)

Generated 52 high-entropy sequences:
  1. 01010101010101010101 (entropy: 1.000 bits)
  2. 01101001101001101001 (entropy: 1.000 bits)
  3. 10011010110110101011 (entropy: 0.954 bits)
```

## Example 2: Transition Frequency Constraints

Generate sequences with specific transition patterns:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)

labels = BinaryLabels(states=("S", "D"), observables=("Same", "Diff"))

# Sequences with many transitions (alternating preference)
constraints_alt = StimulusConstraints(
    minimum_length=30,
    maximum_length=40,
    transition_same_min=0.0,
    transition_same_max=0.3,  # Few same-value transitions
)

result_alt = search_stimulus(
    labels=labels,
    constraints=constraints_alt,
    preferences=StimulusPreferences(target_entropy_bits=0.95),
    max_candidates=50,
)

print("Alternating sequences:")
for candidate in result_alt.candidates[:2]:
    seq_str = candidate.sequence
    same_count = sum(1 for i in range(len(seq_str)-1) if seq_str[i] == seq_str[i+1])
    total_transitions = len(seq_str) - 1
    print(f"  {seq_str} "
          f"({same_count}/{total_transitions} same transitions)")

# Sequences with few transitions (run-like preference)
constraints_runs = StimulusConstraints(
    minimum_length=30,
    maximum_length=40,
    transition_same_min=0.8,  # Many same-value transitions
    transition_same_max=1.0,
)

result_runs = search_stimulus(
    labels=labels,
    constraints=constraints_runs,
    preferences=StimulusPreferences(target_entropy_bits=0.3),
    max_candidates=50,
)

print("\nRun-like sequences:")
for candidate in result_runs.candidates[:2]:
    seq_str = candidate.sequence
    same_count = sum(1 for i in range(len(seq_str)-1) if seq_str[i] == seq_str[i+1])
    total_transitions = len(seq_str) - 1
    print(f"  {seq_str} "
          f"({same_count}/{total_transitions} same transitions)")
```

## Example 3: Boundary Constraints

Generate sequences with specific first and last values:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)

labels = BinaryLabels(states=("A", "B"), observables=("H", "T"))

# Sequences starting with H, ending with T
constraints = StimulusConstraints(
    minimum_length=15,
    maximum_length=25,
    first_observable_index=0,  # Must start with H (index 0)
    last_observable_index=1,   # Must end with T (index 1)
)

result = search_stimulus(
    labels=labels,
    constraints=constraints,
    preferences=StimulusPreferences(target_entropy_bits=0.5),
    max_candidates=100,
)

print("Sequences starting with H, ending with T:")
for candidate in result.candidates[:5]:
    seq_str = candidate.sequence
    first, last = seq_str[0], seq_str[-1]
    print(f"  {seq_str} (first={first}, last={last})")
```

## Example 4: Multiple Constraints and Preferences

Combine hard constraints with soft optimization goals:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)

labels = BinaryLabels(states=("X", "Y"), observables=("A", "B"))

# Must be 20-30 chars, mostly B's, but some A's
constraints = StimulusConstraints(
    minimum_length=20,
    maximum_length=30,
    observable_a_min=0.2,  # At least 20% A
    observable_a_max=0.4,  # At most 40% A
)

preferences = StimulusPreferences(
    target_entropy_bits=0.6,  # Moderate entropy
)

result = search_stimulus(
    labels=labels,
    constraints=constraints,
    preferences=preferences,
    max_candidates=200,
)

print(f"Generated {len(result.candidates)} sequences:")
print("Top 5 candidates by score:")
for i, candidate in enumerate(result.candidates[:5]):
    seq_str = candidate.sequence
    a_count = seq_str.count("A")
    a_freq = a_count / len(seq_str)
    print(f"  {i+1}. {seq_str}")
    print(f"     Entropy: {candidate.entropy_bits:.3f} bits, "
          f"A frequency: {a_freq:.1%}, "
          f"Score: {candidate.ranking_score:.3f}")
```

## Example 5: Stimulus Complement Pairs

Generate paired sequences with opposite properties for experimental design:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
    match_complement,
)

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))

# High-entropy stimulus
high_entropy_constraints = StimulusConstraints(
    minimum_length=25,
    maximum_length=35,
    entropy_bits_minimum=0.95,
    entropy_bits_maximum=1.0,
)

high_entropy_result = search_stimulus(
    labels=labels,
    constraints=high_entropy_constraints,
    preferences=StimulusPreferences(target_entropy_bits=1.0),
    max_candidates=100,
)

# Low-entropy stimulus
low_entropy_constraints = StimulusConstraints(
    minimum_length=25,
    maximum_length=35,
    entropy_bits_minimum=0.0,
    entropy_bits_maximum=0.2,
)

low_entropy_result = search_stimulus(
    labels=labels,
    constraints=low_entropy_constraints,
    preferences=StimulusPreferences(target_entropy_bits=0.0),
    max_candidates=100,
)

# Match pairs by bitwise complement
if high_entropy_result.candidates and low_entropy_result.candidates:
    high_seq = high_entropy_result.candidates[0].sequence
    
    # Find complement in low-entropy candidates
    for low_cand in low_entropy_result.candidates:
        low_seq = low_cand.sequence
        if len(high_seq) == len(low_seq):
            # Check if they are bitwise complements
            is_complement = all(
                h != l for h, l in zip(high_seq, low_seq)
            )
            if is_complement:
                print(f"High entropy: {high_seq} ({high_seq.count('1')}/len is 1's)")
                print(f"Low entropy:  {low_seq} ({low_seq.count('1')}/len is 1's)")
                print(f"Bitwise complement: {is_complement}")
                break
```

## Example 6: Observable Frequency Targeting

Generate sequences with specific observable distributions:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)

labels = BinaryLabels(states=("A", "B"), observables=("X", "Y"))

# 80% X, 20% Y
constraints_skewed = StimulusConstraints(
    minimum_length=25,
    maximum_length=35,
    observable_a_min=0.75,
    observable_a_max=0.85,
)

result_skewed = search_stimulus(
    labels=labels,
    constraints=constraints_skewed,
    preferences=StimulusPreferences(target_entropy_bits=0.4),
    max_candidates=100,
)

print("Skewed distribution (80% X, 20% Y):")
for candidate in result_skewed.candidates[:3]:
    seq = candidate.sequence
    x_freq = seq.count("X") / len(seq)
    print(f"  {seq} ({x_freq:.1%} X)")

# Balanced distribution (50% each)
constraints_balanced = StimulusConstraints(
    minimum_length=25,
    maximum_length=35,
    observable_a_min=0.45,
    observable_a_max=0.55,
)

result_balanced = search_stimulus(
    labels=labels,
    constraints=constraints_balanced,
    preferences=StimulusPreferences(target_entropy_bits=1.0),
    max_candidates=100,
)

print("\nBalanced distribution (50% X, 50% Y):")
for candidate in result_balanced.candidates[:3]:
    seq = candidate.sequence
    x_freq = seq.count("X") / len(seq)
    print(f"  {seq} ({x_freq:.1%} X)")
```

## Example 7: Batched Generation with Export

Generate multiple stimulus sets and export results:

```python
from bspe import (
    BinaryLabels,
    search_stimulus,
    StimulusConstraints,
    StimulusPreferences,
)
import json

labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))

configs = [
    {
        "name": "Low Entropy",
        "constraints": StimulusConstraints(
            minimum_length=20,
            entropy_bits_maximum=0.3,
        ),
        "preferences": StimulusPreferences(target_entropy_bits=0.1),
    },
    {
        "name": "Mid Entropy",
        "constraints": StimulusConstraints(
            minimum_length=20,
            entropy_bits_minimum=0.4,
            entropy_bits_maximum=0.6,
        ),
        "preferences": StimulusPreferences(target_entropy_bits=0.5),
    },
    {
        "name": "High Entropy",
        "constraints": StimulusConstraints(
            minimum_length=20,
            entropy_bits_minimum=0.9,
        ),
        "preferences": StimulusPreferences(target_entropy_bits=0.95),
    },
]

results_by_config = {}
for config in configs:
    result = search_stimulus(
        labels=labels,
        constraints=config["constraints"],
        preferences=config["preferences"],
        max_candidates=50,
    )
    results_by_config[config["name"]] = [
        {
            "sequence": c.sequence,
            "entropy_bits": c.entropy_bits,
            "score": c.ranking_score,
        }
        for c in result.candidates[:10]
    ]

# Export to JSON
with open("stimulus_results.json", "w") as f:
    json.dump(results_by_config, f, indent=2)

print("Exported stimulus results:")
for config_name, candidates in results_by_config.items():
    print(f"  {config_name}: {len(candidates)} candidates")
```

## Best Practices

1. **Constraint Feasibility**: Ensure constraints aren't contradictory (e.g., entropy_max < entropy_min). The search will fail gracefully if infeasible.
2. **Max Candidates**: Balance thoroughness and speed. 50–200 is typical; higher for complex constraints.
3. **Seed Control**: For reproducible searches, consider setting random seed before calling search_stimulus.
4. **Result Inspection**: Always inspect top-ranked candidates; scores may not reflect all desired properties.
5. **Complement Validation**: When pairing stimuli, verify bitwise complements explicitly.
6. **Export Format**: Use JSON for machine-readable exports; CSV for spreadsheet analysis.

## See Also

- [Stimulus Search User Guide](../user-guide/stimulus-search.md)
- [Concepts: Stimulus](../getting-started/concepts.md#stimulus)
- [API Reference: Stimulus Search](../api/stimulus-search.md)
