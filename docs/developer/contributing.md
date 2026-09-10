# Contributing Guide

Contributions to bspe are welcome! This guide covers development setup, testing, and submission guidelines.

## Development Setup

### Prerequisites

- Python 3.13+
- `uv` (fast Python package installer): https://docs.astral.sh/uv/

### Clone and Install

```bash
git clone https://github.com/rzgrozt/binary-context-entropy-calculator.git
cd binary-context-entropy-calculator

# Install with development dependencies
uv sync --all-groups
```

### Development Environment

Verify setup:

```bash
uv run python --version  # Should be 3.13+
uv run pytest --version
uv run ruff --version
uv run basedpyright --version
```

## Code Style

### Formatting and Linting

bspe uses **Ruff** (all rules) at 88 columns with Google-style docstrings.

Check and fix:

```bash
# Check
uv run ruff check src/ tests/

# Fix (auto-fixes most issues)
uv run ruff check src/ tests/ --fix

# Format
uv run ruff format src/ tests/
```

### Type Checking

Use **basedpyright** in all-strict mode:

```bash
uv run basedpyright src/
```

All public APIs must have full type hints.

### Docstring Style

Google format with complete Args, Returns, Raises sections:

```python
def analyze_sequence(
    sequence: BinarySequence,
    model: BinaryHMM,
) -> SequenceResult:
    """Analyze a single sequence with a Hidden Markov Model.
    
    Applies the forward algorithm to compute predictive distributions
    and entropy estimates at each position.
    
    Args:
        sequence: Binary sequence to analyze.
        model: Two-state, two-observable HMM configuration.
    
    Returns:
        Per-position predictions and aggregate entropy.
    
    Raises:
        ZeroLikelihoodError: If observable has zero probability in context.
        NumericalInvariantError: If entropy calculation fails.
    """
```

## Testing

### Test Organization

```
tests/
├── unit/           # Scientific/core logic, no side effects
├── ui/             # State management and helpers
└── integration/    # Streamlit workflows (AppTest)
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/unit/test_markov.py

# Specific test
uv run pytest tests/unit/test_markov.py::test_markov_entropy

# With verbose output
uv run pytest -v

# With coverage
uv run pytest --cov=src/bspe
```

### Test Configuration

Pytest is configured with:
- **Strict mode**: Warnings as errors
- **Coverage threshold**: 95%
- **Markers**: For categorizing tests

Config in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
strict_markers = true
filterwarnings = ["error"]
```

### Writing Tests

**Unit tests** (scientific correctness):

```python
def test_markov_entropy_alternating() -> None:
    """Test Markov entropy on perfectly alternating sequence."""
    labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))
    sequence = (0, 1, 0, 1, 0, 1)  # Perfect alternation
    
    result = analyze_markov(sequence, labels)
    
    # Perfectly predictable → entropy = 0
    assert result.entropy_bits == pytest.approx(0.0, abs=1e-6)
```

**UI tests** (state and helper functions):

```python
def test_form_validation_rejects_invalid_probability() -> None:
    """Test form validation catches probabilities outside [0, 1]."""
    form = CalculatorForm(
        observable_labels=("H", "T"),
        initial_probability_a=1.5,  # Invalid
    )
    
    # Validation should fail
    with pytest.raises(ProbabilityRangeError):
        form.to_model()
```

**Integration tests** (Streamlit workflows):

```python
def test_streamlit_markov_workflow() -> None:
    """Test end-to-end Markov analysis in Streamlit app."""
    app = AppTest(script_path("streamlit_app.py"))
    app.run()
    
    # Upload data
    app.session_state.uploaded_file = ...
    app.run()
    
    # Select method
    app.session_state.method = "Markov"
    app.run()
    
    # Check results displayed
    assert "Entropy" in app.text
```

## Version Control

### Commit Messages

Use clear, descriptive messages:

```
feat: add VMM depth evidence visualization in workbench

- Implement per-position depth evidence metrics
- Add depth evidence column to result export
- Update example notebook with depth interpretation

Fixes #42
```

Format:
- **Type**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- **Description**: Present tense, ~50 chars
- **Body**: Detailed explanation (optional)
- **Footer**: Reference issues/PRs

### Branch Naming

```
feature/vmm-depth-evidence
fix/markov-entropy-edge-case
docs/update-readme
```

## Pull Request Process

1. **Fork** the repository
2. **Branch** off main with descriptive name
3. **Commit** with clear messages
4. **Push** to your fork
5. **Create PR** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
6. **Pass CI**:
   - Tests (pytest)
   - Linting (ruff)
   - Type checking (basedpyright)
   - Coverage (95%+)
7. **Review** process
   - Address feedback
   - Force-push to update branch

## Adding a New Analysis Method

### Steps

1. **Create method module** in `src/bspe/methods/newmethod.py`
2. **Define request** class (inherits `AnalysisRequest`)
3. **Implement analysis** function
4. **Define result** classes
5. **Add tests** in `tests/unit/test_newmethod.py`
6. **Export** from `src/bspe/__init__.py`
7. **Update documentation** (user guide, examples, API)
8. **Update workbench** if needed

### Template

```python
# src/bspe/methods/newmethod.py
from dataclasses import dataclass
from bspe.domain import BinaryLabels, BinarySequence
from bspe.records import SequenceDataset

@dataclass(frozen=True, slots=True, init=False)
class NewMethodAnalysisRequest:
    """Configuration for new method analysis."""
    
    def __init__(self) -> None:
        pass

@dataclass(frozen=True, slots=True)
class NewMethodResult:
    """Result container for new method."""
    records: tuple[NewMethodRecordAnalysis, ...]

def analyze_new_method(
    dataset: SequenceDataset,
    request: NewMethodAnalysisRequest,
) -> NewMethodResult:
    """Analyze sequences with new method."""
    records = []
    for record in dataset.records:
        # Implementation
        pass
    return NewMethodResult(records=tuple(records))
```

## Documentation Updates

When making code changes:

1. **Update docstrings** (Google format)
2. **Add type hints** (all public APIs)
3. **Update user guide** if behavior changes
4. **Add examples** for new functionality
5. **Update API reference** (auto-generated from docstrings)
6. **Update changelog** in docs/reference/changelog.md

## Performance Considerations

- **Time complexity**: Target O(n) for sequence length n
- **Space complexity**: Keep reasonable (< 100MB for typical datasets)
- **Vectorization**: Use NumPy where beneficial
- **Profiling**: Use `cProfile` for hot paths

Benchmark before submitting large changes:

```bash
uv run python -m cProfile -s cumtime your_script.py
```

## Release Process

1. **Increment version** in `pyproject.toml`
2. **Update changelog** with new features/fixes
3. **Tag commit** with version (`v0.2.0`)
4. **Build package**: `uv build`
5. **Upload to PyPI**: (automated via CI)
6. **Create GitHub release** with changelog

## Common Issues

### Import Errors

If pytest can't find modules:

```bash
# Reinstall in editable mode
uv sync --all-groups
```

### Type Checking Failures

Use `basedpyright` for full diagnostic:

```bash
uv run basedpyright src/ --verbose
```

### Coverage Below 95%

Add missing test cases:

```bash
uv run pytest --cov=src/bspe --cov-report=html
# Open htmlcov/index.html to see gaps
```

### Ruff Errors

Most can be auto-fixed:

```bash
uv run ruff check src/ tests/ --fix
```

For unsupported warnings, document as exception in pyproject.toml.

## Architecture

See [Architecture Guide](architecture.md) for component overview and design patterns.

## Questions?

- **Issues**: Open a GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact via GitHub profile

## Code of Conduct

Be respectful and inclusive. Harassment, discrimination, or disruptive behavior is not tolerated.

## License

By contributing, you agree that your code will be licensed under MIT (same as the project).
