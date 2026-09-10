# Installation

## Requirements

- **Python 3.13 or newer** (required for type hints and pattern matching)
- pip or uv package manager

## Install from PyPI

The easiest way to install **bspe** is from PyPI:

```bash
pip install bspe
```

Or with uv:

```bash
uv pip install bspe
```

## Verify Installation

```python
import bspe
print(bspe.__version__)  # Should print: 0.1.0
```

## Development Installation

To contribute or run the Streamlit workbench from source:

```bash
git clone https://github.com/rzgrozt/binary-context-entropy-calculator.git
cd binary-context-entropy-calculator
uv sync
```

This installs all dependencies and development tools in an isolated environment.

## Run the Streamlit Workbench

```bash
uv run streamlit run streamlit_app.py
```

The workbench opens at `http://localhost:8501`.

## Troubleshooting

### Python Version Error

If you see `Python 3.13 or newer is required`:

```bash
python --version  # Check your Python version
```

Install Python 3.13+ from [python.org](https://www.python.org/downloads/) or use a version manager:

```bash
# Using pyenv
pyenv install 3.13.0
pyenv local 3.13.0

# Using conda
conda create -n bspe python=3.13
conda activate bspe
```

### Import Error

If `import bspe` fails:

```bash
# Verify installation
pip show bspe

# Reinstall
pip install --upgrade --force-reinstall bspe
```

### Streamlit Not Found

If `streamlit run` fails:

```bash
# Install Streamlit explicitly
pip install streamlit>=1.61.1

# Or reinstall bspe with all dependencies
pip install --upgrade bspe[all]
```

## Next Steps

- **[Quick Start](quick-start.md)**: Run your first analysis
- **[Concepts](concepts.md)**: Understand core ideas
- **[User Guide](../user-guide/overview.md)**: Detailed method descriptions
