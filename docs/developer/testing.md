# Testing Guide

Comprehensive testing strategy and best practices for bspe.

## Test Organization

```
tests/
├── unit/                    # Scientific/core logic tests
│   ├── test_markov.py
│   ├── test_vmm.py
│   ├── test_hmm.py
│   ├── test_shannon.py
│   ├── test_parsing.py
│   ├── test_batch_parsing.py
│   ├── test_stimulus_search.py
│   └── helpers.py           # Shared fixtures
│
├── ui/                      # UI state and helpers
│   ├── test_state.py
│   ├── test_results.py
│   ├── test_components.py
│   └── ...
│
└── integration/             # End-to-end workflows
    ├── test_streamlit_calculation.py
    ├── test_streamlit_workbench.py
    ├── test_streamlit_workspace.py
    └── ...
```

## Running Tests

### All Tests

```bash
uv run pytest
```

### Specific Category

```bash
uv run pytest tests/unit/          # Only unit tests
uv run pytest tests/ui/            # Only UI tests
uv run pytest tests/integration/   # Only integration tests
```

### Specific Test File

```bash
uv run pytest tests/unit/test_markov.py
```

### Specific Test Function

```bash
uv run pytest tests/unit/test_markov.py::test_markov_entropy_alternating
```

### With Verbose Output

```bash
uv run pytest -v
```

### With Coverage Report

```bash
uv run pytest --cov=src/bspe --cov-report=html
# Open htmlcov/index.html in browser
```

### Watch Mode (on file change)

```bash
uv run pytest-watch
```

## Test Configuration

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
strict_markers = true
filterwarnings = ["error"]  # Warnings fail tests
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src/bspe"]
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
min_percentage = 95
```

Minimum 95% line coverage required.

## Unit Testing

### Purpose

Test scientific correctness and isolated components:

- Entropy calculations
- Probability distributions
- Parsing logic
- Data validation
- Algorithm correctness

### Structure

```python
import pytest
from bspe import analyze_markov, BinaryLabels, SequenceRecord

def test_markov_entropy_alternating() -> None:
    """Test Markov entropy on perfectly alternating sequence.
    
    For a perfectly alternating sequence (0101...), the next observable
    is fully determined by the previous one. Entropy should be 0.
    """
    labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))
    sequence = (0, 1, 0, 1, 0, 1)
    
    record = SequenceRecord("seq1", sequence, labels=labels)
    result = analyze_markov(record, labels)
    
    assert result.entropy_bits == pytest.approx(0.0, abs=1e-6)
```

### Naming Conventions

- `test_<component>_<scenario>`: Clearly describe what is tested
- Examples:
  - `test_markov_entropy_alternating`
  - `test_vmm_backoff_minimum_support`
  - `test_parse_sequence_invalid_token`

### Assertions

Use clear assertions with helpful messages:

```python
# Good
assert result.entropy_bits == pytest.approx(expected, abs=1e-6)
assert record.sequence == (0, 1, 0, 1)
assert len(results) == 3

# Less good
assert result.entropy_bits > 0  # Too vague
```

### Fixtures

Shared test data in `tests/unit/helpers.py`:

```python
import pytest
from bspe import BinaryLabels

@pytest.fixture
def labels() -> BinaryLabels:
    """Standard binary labels for testing."""
    return BinaryLabels(states=("A", "B"), observables=("0", "1"))

@pytest.fixture
def alternating_sequence() -> tuple[int, ...]:
    """Perfect alternating sequence."""
    return (0, 1, 0, 1, 0, 1, 0, 1)
```

Usage in tests:

```python
def test_something(labels: BinaryLabels, alternating_sequence: tuple[int, ...]) -> None:
    """Test using fixtures."""
    # Use labels and alternating_sequence
```

### Test Data

Organize in `tests/fixtures/`:

```
tests/fixtures/
├── hand_sequence.json        # Pre-computed test sequence
├── sample_data.csv           # CSV test data
└── README.md                 # Fixture documentation
```

Load in tests:

```python
import json

def load_hand_fixture() -> dict:
    """Load pre-computed hand sequence fixture."""
    path = Path(__file__).parent / "fixtures" / "hand_sequence.json"
    with open(path) as f:
        return json.load(f)
```

### Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("sequence,expected_entropy", [
    ((0, 0, 0, 0), 0.0),           # All same
    ((0, 1, 0, 1), 0.0),           # Alternating
    ((0, 0, 1, 1), 0.0),           # Two runs
    ((0, 1, 0, 1, 1, 0), 0.918),  # Mixed
])
def test_markov_entropy(
    sequence: tuple[int, ...],
    expected_entropy: float,
) -> None:
    """Test Markov entropy on various sequences."""
    result = analyze_markov(sequence, labels)
    assert result.entropy_bits == pytest.approx(expected_entropy, abs=1e-3)
```

## UI Testing

### Purpose

Test state management and helper functions:

- Form validation
- State transitions
- Data transformation
- Error handling

### Structure

```python
def test_form_validation_rejects_invalid_probability() -> None:
    """Test form validation catches probability errors."""
    with pytest.raises(ProbabilityRangeError):
        ModelForm(
            observable_labels=("H", "T"),
            initial_probability_a=1.5,  # Invalid: > 1.0
        )

def test_form_converts_to_model() -> None:
    """Test form correctly creates BinaryHMM."""
    form = ModelForm(
        observable_labels=("H", "T"),
        initial_probability_a=0.6,
        transition_p_a=[[0.7, 0.3], [0.4, 0.6]],
        emission_p_a=[[0.9, 0.1], [0.2, 0.8]],
    )
    
    model = form.to_model()
    assert model.initial[0] == pytest.approx(0.6)
```

### No Network/Streamlit Interaction

UI tests are isolated (no actual Streamlit session):

```python
# Good: Test state transformation
def test_state_update() -> None:
    state = CalculatorState(...)
    updated = state.with_method("Markov")
    assert updated.method == "Markov"

# Not in unit tests: Actual Streamlit calls
# (Those go in integration tests)
```

## Integration Testing

### Purpose

Test end-to-end workflows in Streamlit:

- User interactions
- Data upload and parsing
- Method selection and execution
- Results display and export

### Tools

Use `streamlit.testing.v1.AppTest`:

```python
from streamlit.testing.v1 import AppTest

def test_streamlit_markov_workflow() -> None:
    """Test end-to-end Markov analysis workflow."""
    app = AppTest(script_path="streamlit_app.py"))
    app.run()
    
    # Verify initial state
    assert "Binary Sequence Analysis" in app.title[0].value
    
    # Upload data
    app.file_uploader[0].set_file_content(b"0 1 0 1")
    app.run()
    
    # Select Markov method
    app.selectbox[0].select("Markov")
    app.run()
    
    # Verify results
    assert "Entropy" in app.text
```

### Test Structure

```python
def test_streamlit_<workflow>() -> None:
    """Test <workflow> in Streamlit app.
    
    Steps:
    1. Launch app
    2. Interact with widgets (upload, select, click)
    3. Verify output (text, dataframes, charts)
    """
    app = AppTest(...)
    
    # Setup
    app.run()
    
    # Action
    app.selectbox[0].select(...)
    app.run()
    
    # Assertion
    assert "Expected output" in app.text
```

### Widget Interaction

Common patterns:

```python
# Text input
app.text_input[0].set_value("new text")

# Number input
app.number_input[0].set_value(42)

# Select/Radio
app.selectbox[0].select("Option")

# Checkbox
app.checkbox[0].check()

# File upload
app.file_uploader[0].set_file_content(b"file content")

# Buttons (no return value, just trigger)
app.button[0].click()

# Sliders
app.slider[0].set_value(0.5)
```

After any interaction, call `app.run()` to re-execute.

### Testing with Fixtures

```python
@pytest.fixture
def sample_csv_content() -> bytes:
    """Sample CSV for testing."""
    return b"""id,sequence,target
seq1,0 1 0 1 0 1,0
seq2,1 1 0 0 1 1,1
"""

def test_upload_and_analyze(sample_csv_content: bytes) -> None:
    """Test uploading CSV and running analysis."""
    app = AppTest(...)
    app.run()
    
    app.file_uploader[0].set_file_content(sample_csv_content)
    app.run()
    
    # Verify parsing succeeded
    assert "2 sequences parsed" in app.text
```

## Test Coverage

### Viewing Coverage

```bash
# Terminal report
uv run pytest --cov=src/bspe

# HTML report
uv run pytest --cov=src/bspe --cov-report=html
# Open htmlcov/index.html
```

### Identifying Gaps

Lines not covered appear in `htmlcov/index.html`:

1. Navigate to file
2. Red lines = untested
3. Add test cases to cover

### Coverage by Component

Target at least 95% per component:

```
src/bspe/
├── domain.py              95%+
├── records.py             95%+
├── parsing.py             95%+
├── methods/markov.py      95%+
├── methods/vmm.py         95%+
├── methods/hmm.py         95%+
└── methods/shannon.py     95%+
```

## Performance Testing

### Profiling

Identify slow operations:

```bash
uv run python -m cProfile -s cumtime your_script.py
```

### Benchmarking

Test execution time doesn't regress:

```python
def test_markov_performance_on_10k_sequences() -> None:
    """Markov analysis should complete in < 5 seconds on 10k sequences."""
    import time
    
    # Create 10,000 test sequences
    sequences = [...]
    dataset = SequenceDataset(labels, [
        SequenceRecord(f"seq_{i}", seq, labels=labels)
        for i, seq in enumerate(sequences)
    ])
    
    start = time.perf_counter()
    result = analyze_dataset(dataset, MarkovAnalysisRequest())
    elapsed = time.perf_counter() - start
    
    assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected < 5s"
```

## Error Testing

### Testing Exceptions

```python
def test_parse_sequence_invalid_token() -> None:
    """Parsing invalid token raises InvalidSequenceTokenError."""
    labels = BinaryLabels(states=("A", "B"), observables=("0", "1"))
    
    with pytest.raises(InvalidSequenceTokenError) as exc_info:
        parse_sequence("0 1 X", labels, separator=" ")
    
    assert exc_info.value.token == "X"

def test_invalid_probability_raises_error() -> None:
    """Invalid probability raises ProbabilityRangeError."""
    with pytest.raises(ProbabilityRangeError):
        BinaryHMM(
            labels=labels,
            initial=[1.5, -0.5],  # Invalid
            transition=[[0.7, 0.3], [0.4, 0.6]],
            emission=[[0.5, 0.5], [0.5, 0.5]],
        )
```

### Testing Recoverable Errors

Batch parsing returns issues separately:

```python
def test_batch_parsing_recovers_from_invalid_rows() -> None:
    """Batch parse returns valid records and skips invalid ones."""
    csv_content = b"""id,sequence
seq1,0 1 0
seq2,INVALID
seq3,1 0 1
"""
    
    records, issues = parse_batch_from_csv(
        ...,
        csv_content,
        ...,
    )
    
    # Should have 2 valid records, 1 issue
    assert len(records) == 2
    assert len(issues) == 1
    assert issues[0].record_index == 1  # Row 2 (0-indexed)
```

## Edge Cases

### Always Test

1. **Boundary conditions**:
   - Empty sequences: `()`
   - Single-element: `(0,)` or `(1,)`
   - Minimum/maximum lengths

2. **Degenerate distributions**:
   - All same: `(0, 0, 0, 0)`
   - Perfect alternation: `(0, 1, 0, 1)`
   - Skewed: `(0, 0, 0, 1)`

3. **Probability extremes**:
   - All probability on one outcome: `[0.0, 1.0]`
   - Uniform distribution: `[0.5, 0.5]`

4. **Numerical edge cases**:
   - Very small probabilities: `[1e-10, 1.0 - 1e-10]`
   - Log of near-zero: Handled gracefully

### Example

```python
@pytest.mark.parametrize("sequence", [
    (),              # Empty
    (0,),            # Single element
    (0, 0, 0),       # All same
    (0, 1, 0, 1),    # Perfect alternation
])
def test_entropy_edge_cases(sequence: tuple[int, ...]) -> None:
    """Test entropy calculation on edge cases."""
    result = analyze_markov(sequence, labels)
    # Entropy should be finite, in [0, 1]
    assert 0.0 <= result.entropy_bits <= 1.0
    assert not math.isnan(result.entropy_bits)
```

## Documentation in Tests

### Clear Test Names

Test names are documentation:

```python
# Clear
def test_vmm_backoff_when_minimum_support_not_met() -> None:
    """VMM backs off to shorter context when support threshold unmet."""

# Unclear
def test_vmm_1() -> None:
    """Test VMM."""
```

### Docstring Comments

```python
def test_markov_entropy_is_zero_for_deterministic_sequence() -> None:
    """Test that Markov entropy is 0 for perfectly predictable sequences.
    
    When each observable is fully determined by the previous observable
    (e.g., alternating patterns), entropy should be 0 bits.
    
    Rationale: If the next observable is always determined, there's
    no uncertainty; entropy quantifies uncertainty.
    """
```

## Continuous Integration

Tests run automatically on:

- **Push to main**: Full test suite
- **Pull requests**: Full test suite + coverage check
- **Scheduled**: Nightly full run on all Python 3.13 versions

See `.github/workflows/tests.yml` for CI configuration.

## Common Issues

### Import Errors in Tests

```bash
# Fix: Reinstall in editable mode
uv sync --all-groups
```

### Coverage Below 95%

```bash
# Identify gaps
uv run pytest --cov=src/bspe --cov-report=term-missing
# Add tests for missing lines
```

### Slow Tests

```bash
# Identify slow tests
uv run pytest --durations=10

# Mark slow test to skip
@pytest.mark.slow
def test_slow_operation() -> None:
    ...

# Run fast tests only
uv run pytest -m "not slow"
```

### Flaky Tests

Avoid time-dependent or random tests:

```python
# Bad: Time-dependent
def test_analysis_completes_fast() -> None:
    start = time.time()
    analyze_dataset(...)
    elapsed = time.time() - start
    assert elapsed < 1.0  # May fail on slow machine

# Good: Deterministic
def test_analysis_correctness() -> None:
    result = analyze_dataset(...)
    assert result.entropy_bits == pytest.approx(expected)
```

## See Also

- [Contributing Guide](contributing.md)
- [Architecture](architecture.md)
- [pytest documentation](https://docs.pytest.org/)
- [Streamlit testing](https://docs.streamlit.io/develop/api-reference/app-testing)
