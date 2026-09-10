"""Immutable context shared by active method renderers."""

from dataclasses import dataclass

from bspe.records import SequenceDataset
from bspe.ui.workbench_state import WorkbenchForm
from bspe.ui.workspace_selection import SequenceScope
from bspe.ui.workspace_session import WorkspaceSubview
from bspe.workbench import WorkbenchResult


@dataclass(frozen=True, slots=True)
class MethodRenderContext:
    """Inputs shared by every active method renderer."""

    result: WorkbenchResult
    form: WorkbenchForm
    dataset: SequenceDataset
    scope: SequenceScope
    subview: WorkspaceSubview
