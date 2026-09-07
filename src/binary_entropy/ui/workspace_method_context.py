"""Immutable context shared by active method renderers."""

from dataclasses import dataclass

from binary_entropy.records import SequenceDataset
from binary_entropy.ui.workbench_state import WorkbenchForm
from binary_entropy.ui.workspace_selection import SequenceScope
from binary_entropy.ui.workspace_session import WorkspaceSubview
from binary_entropy.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class MethodRenderContext:
    """Inputs shared by every active method renderer."""

    result: WorkbenchResult
    form: WorkbenchForm
    dataset: SequenceDataset
    scope: SequenceScope
    subview: WorkspaceSubview
