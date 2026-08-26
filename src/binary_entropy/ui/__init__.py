"""Streamlit-facing adapters for the binary entropy calculator."""

from binary_entropy.ui.state import default_form
from binary_entropy.ui.workbench_state import default_workbench_form

__all__ = [
    "default_form",
    "default_workbench_form",
]
