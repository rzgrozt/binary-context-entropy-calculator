# Changelog

All notable changes to **bspe** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-10

### Initial Release

**bspe** v0.1.0 is the first public release of the Binary Sequence Prediction & Entropy framework.

#### Added

**Core Analysis Methods:**
- Variable-Order Markov (VMM) with automatic suffix backoff
  - Krichevsky-Trofimov (KT) smoothing (default)
  - Maximum Likelihood Estimation (MLE)
  - Custom additive smoothing
  - Pooled and per-sequence analysis modes
  - Configurable minimum context support
  - Depth-evidence tables with backoff tracking

- First-Order Markov baseline
  - Fixed fitted matrix or re-estimated prefix modes
  - Stationary distribution calculation
  - Entropy rate computation
  - Maximum likelihood and additive smoothing

- Configured Hidden Markov Model
  - Two hidden states, two observables
  - Schema-v1 preset compatibility
  - Independent per-record filtering
  - Posterior and predictive distributions

- Observed Shannon Entropy
  - Pooled and per-sequence summaries
  - Prefix-level analysis

**Stimulus Search:**
- Bounded deterministic candidate generation
- Seeded sample without replacement
- Hard constraints (11 categories)
- Soft preferences with weighted absolute-distance ranking
- Complement generation and tolerance-based matching
- Balanced target assignment
- Batch processing for bounded memory

**Data Parsing:**
- Manual batch parsing (one sequence per line)
- TXT file parsing (UTF-8 with optional BOM)
- CSV parsing with explicit column mapping
- Comprehensive error reporting with row/token positions

**Serialization:**
- VMM context model JSON (experimental)
- VMM context evidence CSV (experimental)
- VMM evaluation CSV (experimental)
- Markov model JSON
- Markov prefix CSV
- Markov batch-summary CSV
- HMM preset JSON (schema-v1)
- HMM prefix CSV
- HMM candidate-summary CSV
- Stimulus candidate CSV
- Stimulus scientific CSV
- Stimulus experiment-ready CSV
- Stimulus reproducibility configuration JSON

**Streamlit Workbench:**
- Analyzer mode: method selection, data input, result comparison
- Stimulus Search mode: configuration, candidate inspection, matching, assignment
- Multiple input modes: single sequence, batch paste, TXT upload, CSV upload
- Interactive result tables and charts
- CSV and JSON exports with full precision
- Persistent sidebar configuration
- Stale-result handling

**Documentation:**
- Getting Started guide (installation, quick start, concepts)
- User Guide (methods, input formats, workflows)
- Examples (VMM, HMM, Markov, stimulus search)
- API Reference (complete public API)
- Developer Guide (contributing, architecture, testing)
- Reference (FAQ, troubleshooting, performance, glossary)
- Hand-worked HMM reference with fixture

**Testing:**
- Unit tests for parsing, fitting, analysis
- Integration tests for Streamlit workflows
- Hand-calculated HMM reference verification
- 95%+ code coverage

**Quality Assurance:**
- Type hints (Python 3.13+)
- Static analysis (basedpyright, ruff)
- Deterministic operations (seeded randomness, sorted output)
- Reproducible exports (full configuration, timestamps)

#### Design Principles

- **Boundary Preservation:** Records never concatenated or merged
- **Explicit Separation:** Prediction, description, search, and target assignment are distinct
- **Determinism:** Same input always produces same output
- **Reproducibility:** Full configuration and precision in exports
- **Scientific Validity:** No implicit assumptions or hidden transformations

#### Known Limitations

- Binary sequences only (not n-ary)
- HMM is configured, not trained
- VMM detects recurrent patterns, not arbitrary or causal patterns
- Stimulus search is bounded (may return partial results)
- Numerical work uses float64 arithmetic
- No held-out validation or statistical inference

#### Requirements

- Python 3.13 or newer
- numpy ≥ 2.1
- pandas ≥ 2.2
- plotly ≥ 6.9.0
- pydantic ≥ 2.10
- streamlit ≥ 1.61.1

#### Contributors

- Ruzgar Ozturk (author)

---

## Versioning

**bspe** uses Semantic Versioning:
- **MAJOR:** Breaking API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

## Roadmap

Potential future directions (not committed):

- [ ] N-ary sequences (not just binary)
- [ ] HMM training/inference
- [ ] Held-out validation
- [ ] Statistical significance testing
- [ ] Performance optimizations
- [ ] GPU acceleration
- [ ] Additional smoothing methods
- [ ] Bayesian inference
- [ ] Interactive visualization improvements

## Support

For issues, questions, or contributions:

- **GitHub Issues:** [rzgrozt/binary-context-entropy-calculator/issues](https://github.com/rzgrozt/binary-context-entropy-calculator/issues)
- **Discussions:** [GitHub Discussions](https://github.com/rzgrozt/binary-context-entropy-calculator/discussions)
- **Email:** 32547215+rzgrozt@users.noreply.github.com

## License

MIT License — see [LICENSE](https://github.com/rzgrozt/binary-context-entropy-calculator/blob/main/LICENSE)

## Citation

If you use **bspe** in research, please cite:

```bibtex
@software{ozturk2026bspe,
  author  = {Ozturk, Ruzgar},
  title   = {bspe: Binary Sequence Prediction \& Entropy},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rzgrozt/binary-context-entropy-calculator}
}
```

Or in text:

> Ozturk, R. (2026). *bspe: Binary Sequence Prediction & Entropy* (Version 0.1.0) [Computer software]. GitHub. https://github.com/rzgrozt/binary-context-entropy-calculator
