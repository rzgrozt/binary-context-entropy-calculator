# bspe — Binary Sequence Prediction & Entropy

**bspe** is a typed Python framework for binary-sequence predictive entropy analysis. It bundles a reusable public API (variable-order Markov, first-order Markov, configured HMM, and observed Shannon entropy, plus reproducible bounded stimulus search) with a local Streamlit workbench for fitting, comparing, and inspecting binary-sequence methods.

## Key Features

- **Variable-order Markov (VMM)**: Context-aware prediction with automatic suffix backoff
  - Krichevsky-Trofimov (KT), MLE, and custom additive smoothing
  - Pooled and per-sequence analysis modes
  - Configurable minimum context support
  - Depth-evidence tables with backoff tracking

- **First-order Markov Baseline**: Transition-matrix prediction
  - Fixed fitted matrix or re-estimated prefix modes
  - Stationary distribution and entropy rate calculation
  - Maximum likelihood and additive smoothing

- **Configured Hidden Markov Model**: Two hidden states, two observables
  - Schema-v1 preset compatibility
  - Independent per-record filtering
  - Posterior and predictive distributions

- **Observed Shannon Entropy**: Empirical symbol frequency analysis
  - Pooled and per-sequence summaries
  - Prefix-level analysis

- **Stimulus Search**: Bounded deterministic candidate generation and ranking
  - Seeded sample without replacement
  - Hard constraints (11 categories)
  - Soft preferences with weighted absolute-distance ranking
  - Complement generation and tolerance-based matching
  - Balanced target assignment
  - Batch processing for bounded memory

- **Streamlit Workbench**: Interactive local analysis and search interface
  - Analyzer mode: method selection, data input, result comparison
  - Stimulus Search mode: configuration, candidate inspection, matching, assignment
  - Multiple input modes: single sequence, batch paste, TXT upload, CSV upload
  - CSV and JSON exports with full precision

## Quick Start

### Installation

```bash
pip install bspe
```

### Basic Usage

```python
from bspe import (
    BinaryLabels,
    VMMAnalysisRequest,
    VMMConfig,
    VMMResultScope,
    analyze_dataset,
    parse_manual_batch,
)

# Define labels
labels = BinaryLabels(
    states=("State 1", "State 2"),
    observables=("A", "B")
)

# Parse sequences
dataset = parse_manual_batch("A, B, B\nB, A", labels)

# Analyze with VMM
result = analyze_dataset(
    dataset,
    VMMAnalysisRequest(
        config=VMMConfig(minimum_support=2),
        result_scope=VMMResultScope.POOLED,
    )
)

# Inspect results
for record in result.records:
    print(f"Sequence: {record.sequence}")
    print(f"Entropy: {record.predictive_entropy_bits:.4f} bits")
```

### Run the Workbench

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation, quick start, and core concepts
- **[User Guide](user-guide/overview.md)**: Detailed method descriptions and workflows
- **[Examples](examples/vmm-analysis.md)**: Runnable code examples
- **[API Reference](api/overview.md)**: Complete API documentation
- **[Developer Guide](developer/contributing.md)**: Contributing, architecture, testing
- **[Reference](reference/faq.md)**: FAQ, troubleshooting, performance, glossary

## Key Concepts

### Boundary Preservation

Records are never concatenated or merged. Each sequence maintains its own:
- Context counts (VMM)
- Transition counts (Markov)
- Filtering state (HMM)

### Explicit Separation of Concerns

The framework separates:
- **Prediction**: VMM, Markov, HMM (fit and predict)
- **Description**: Shannon entropy (empirical analysis)
- **Search**: Stimulus generation and ranking
- **Target Assignment**: Separate from fitting (never influences model)

### Determinism

All operations are deterministic:
- Seeded randomness in stimulus search
- Sorted output (by context, stimulus ID, etc.)
- Reproducible JSON/CSV exports
- No floating-point comparison ambiguity

## Requirements

- Python 3.13 or newer
- numpy ≥2.1
- pandas ≥2.2
- plotly ≥6.9.0
- pydantic ≥2.10
- streamlit ≥1.61.1

## Citation

If you use this software in research, cite:

```bibtex
@software{ozturk2026bspe,
  author  = {Ozturk, Ruzgar},
  title   = {bspe: Binary Sequence Prediction \& Entropy},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rzgrozt/binary-context-entropy-calculator}
}
```

## License

MIT License — see [LICENSE](https://github.com/rzgrozt/binary-context-entropy-calculator/blob/main/LICENSE)

## Links

- **GitHub**: [rzgrozt/binary-context-entropy-calculator](https://github.com/rzgrozt/binary-context-entropy-calculator)
- **PyPI**: [bspe](https://pypi.org/project/bspe/)
- **Issues**: [GitHub Issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
