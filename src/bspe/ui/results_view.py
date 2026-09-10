"""Tracked active-tab coordination for immutable workbench results."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, assert_never

import streamlit as st

from bspe.ui.session import (
    WorkbenchCalculationRecord,
    WorkbenchSubmissionFailure,
)
from bspe.ui.text import joined_text
from bspe.ui.workbench_state import MethodChoice, WorkbenchForm
from bspe.ui.workspace_current import (
    CurrentSubmission,
    current_submission,
    failure_for_method,
    result_for_method,
)
from bspe.ui.workspace_exports import (
    WorkspaceArtifact,
    selected_artifacts,
)
from bspe.ui.workspace_renderers import (
    ComparisonWorkspaceContext,
    MethodWorkspaceContext,
    render_comparison_workspace,
    render_method_workspace,
)
from bspe.ui.workspace_selection import SequenceScope, project_record
from bspe.ui.workspace_session import (
    WorkspaceSubview,
    render_subview_control,
    render_workspace_selection,
)


class WorkspaceTab(StrEnum):
    """Fixed tracked tabs in source and visual order."""

    COMPARISON = "Method Comparison"
    MARKOV = "Markov Chain"
    HMM = "Hidden Markov Model"
    SHANNON = "Observed Shannon Entropy"


TAB_LABELS: Final = tuple(tab.value for tab in WorkspaceTab)


@dataclass(frozen=True, slots=True)
class MethodTabContext:
    """Inputs required to render one active method tab."""

    method: MethodChoice
    current: CurrentSubmission
    projected: WorkbenchCalculationRecord
    form: WorkbenchForm
    scope: SequenceScope
    subview: WorkspaceSubview


def render_results(
    form: WorkbenchForm,
    record: WorkbenchCalculationRecord | None,
    failure: WorkbenchSubmissionFailure | None,
) -> None:
    """Render selectors and only the open tab from stored calculations."""
    _ = st.title("Result workspace")
    _ = st.caption("Wide comparison and result tables scroll horizontally.")
    tabs = st.tabs(TAB_LABELS, key="workspace-method-tabs", on_change="rerun")
    active_index = next(index for index, tab in enumerate(tabs) if tab.open)
    active_tab = WorkspaceTab(TAB_LABELS[active_index])
    if failure is not None:
        _ = st.error(failure.message)
    if record is None:
        with tabs[active_index]:
            _render_uncalculated(active_tab, form)
        return

    current = current_submission(form, record.success)
    if current.stale_methods:
        methods = ", ".join(method.value for method in current.stale_methods)
        _ = st.warning(f"Recalculation required: {methods}.")
    with tabs[active_index]:
        selection = render_workspace_selection(record.success.dataset)
        subview = render_subview_control()
        if not selection.selected_ids:
            _ = st.info(
                "Select at least one sequence identifier for Multiple scope."
            )
            return
        current_record = WorkbenchCalculationRecord(
            replace(record.success, results=current.results, failures=current.failures)
        )
        projected = project_record(current_record, selection)
        match active_tab:
            case WorkspaceTab.COMPARISON:
                render_comparison_workspace(
                    ComparisonWorkspaceContext(
                        projected.success.results,
                        form,
                        selection.scope,
                        subview,
                    )
                )
                return
            case WorkspaceTab.MARKOV:
                _render_method_tab(
                    MethodTabContext(
                        MethodChoice.MARKOV,
                        current,
                        projected,
                        form,
                        selection.scope,
                        subview,
                    )
                )
                return
            case WorkspaceTab.HMM:
                _render_method_tab(
                    MethodTabContext(
                        MethodChoice.HMM,
                        current,
                        projected,
                        form,
                        selection.scope,
                        subview,
                    )
                )
                return
            case WorkspaceTab.SHANNON:
                _render_method_tab(
                    MethodTabContext(
                        MethodChoice.SHANNON,
                        current,
                        projected,
                        form,
                        selection.scope,
                        subview,
                    )
                )
                return
        assert_never(active_tab)


def _render_uncalculated(active_tab: WorkspaceTab, form: WorkbenchForm) -> None:
    if active_tab is WorkspaceTab.HMM and MethodChoice.HMM not in form.methods:
        _ = st.info("HMM disabled for the current calculation.")
        return
    _ = st.info(
        f"Not calculated: {active_tab.value}. Use Calculate selected methods."
    )


def _render_method_tab(context: MethodTabContext) -> None:
    method = context.method
    if method not in context.form.methods:
        message = (
            "HMM disabled for the current calculation."
            if method is MethodChoice.HMM
            else f"{method.value} disabled for the current calculation."
        )
        _ = st.info(message)
        return
    if method in context.current.stale_methods:
        return
    result = result_for_method(context.projected.success.results, method)
    if result is None:
        failure = failure_for_method(context.current.failures, method)
        if failure is not None:
            _ = st.error(failure.message)
        else:
            _ = st.info("Not calculated.")
        return
    render_method_workspace(
        MethodWorkspaceContext(
            result,
            context.form,
            context.projected.success.dataset,
            context.scope,
            context.subview,
        )
    )
    _render_selected_exports(method, context.projected, context.form)
    with st.expander("Reproducibility details"):
        _ = st.markdown(
            "\n".join(
                (
                    f"- Method: {method.value}",
                    joined_text(
                        (
                            "- Selected records: ",
                            f"{len(context.projected.success.dataset.records)}",
                        )
                    ),
                    "- Dataset role: training",
                    "- Visible precision: exactly 3 decimal places",
                    "- Raw export precision: unrounded or at least 12 decimal places",
                    "- Ordering: deterministic submitted record order",
                    "- Evaluation targets do not affect fitting or prediction",
                )
            )
        )


def _render_selected_exports(
    method: MethodChoice,
    projected: WorkbenchCalculationRecord,
    form: WorkbenchForm,
) -> None:
    identifiers = tuple(
        str(item.sequence_id) for item in projected.success.dataset.records
    )
    for artifact in selected_artifacts(projected, form, identifiers):
        if _artifact_matches_method(artifact, method):
            _ = st.download_button(
                f"Download selected {artifact.name}",
                data=artifact.data,
                file_name=artifact.file_name,
                mime=artifact.mime,
                on_click="ignore",
            )


def _artifact_matches_method(
    artifact: WorkspaceArtifact,
    method: MethodChoice,
) -> bool:
    match method:
        case MethodChoice.MARKOV:
            return artifact.name.startswith("Markov") or artifact.name in {
                "Context model export",
                "Context evidence export",
                "Evaluation export",
            }
        case MethodChoice.HMM:
            return artifact.name.startswith("HMM")
        case MethodChoice.SHANNON:
            return False
    assert_never(method)
