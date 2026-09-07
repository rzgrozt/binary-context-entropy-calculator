"""Persistent grouped sidebar for configuration, status, and exports."""

from dataclasses import dataclass

import streamlit as st

from binary_entropy.ui.form import render_workbench_form
from binary_entropy.ui.inputs import render_observable_labels
from binary_entropy.ui.session import (
    WorkbenchCalculationRecord,
    store_workbench_calculation,
    workbench_calculation_record,
)
from binary_entropy.ui.setup import render_method_selection
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workbench_state import (
    MethodChoice,
    WorkbenchForm,
    calculate_workbench,
)
from binary_entropy.ui.workspace_current import current_calculation_record
from binary_entropy.ui.workspace_exports import complete_artifacts


@dataclass(frozen=True, slots=True)
class SidebarState:
    """Editable form and the latest immutable calculation snapshot."""

    form: WorkbenchForm
    calculation: WorkbenchCalculationRecord | None


def render_sidebar() -> SidebarState:
    """Render the compact scientific control rail in task order."""
    _ = st.title("Binary entropy workbench")
    _ = st.caption(
        "Configure methods and training data without leaving the results."
    )
    with st.expander("General settings", expanded=False):
        _ = st.subheader("Analysis methods")
        methods = render_method_selection()
        if methods:
            _ = st.caption(
                "Active methods: " + ", ".join(item.value for item in methods)
            )
        else:
            _ = st.error("Select at least one analysis method.")
        observable_labels = render_observable_labels()
        if MethodChoice.SHANNON in methods:
            _ = st.caption(
                joined_text(
                    (
                        "Observed Shannon entropy has no model controls; ",
                        "it summarizes submitted symbols.",
                    )
                )
            )

    form = render_workbench_form(methods, observable_labels)
    _ = st.caption(
        joined_text(
            (
                "No calculation runs while editing. ",
                "Submit the current configuration explicitly.",
            )
        )
    )
    if st.button(
        "Calculate selected methods",
        type="primary",
        disabled=not methods,
    ):
        store_workbench_calculation(calculate_workbench(form), form)

    calculation = workbench_calculation_record()
    current = current_calculation_record(form, calculation)
    with st.expander("Calculation status", expanded=False):
        if calculation is None:
            _ = st.info("Not calculated.")
        elif current is not None and current.success.failures:
            _ = st.warning("Current calculation has unavailable method results.")
        elif current is not None and current.success.results:
            _ = st.success("Current results are available.")
        else:
            _ = st.warning("Recalculation required for the selected methods.")
    with st.expander("Complete exports"):
        if current is None or not current.success.results:
            _ = st.caption("Complete exports require a current valid result.")
        else:
            _render_complete_exports(current, form)
    return SidebarState(form, calculation)


def _render_complete_exports(
    calculation: WorkbenchCalculationRecord,
    form: WorkbenchForm,
) -> None:
    for artifact in complete_artifacts(calculation, form):
        _ = st.download_button(
            f"Download {artifact.name}",
            data=artifact.data,
            file_name=artifact.file_name,
            mime=artifact.mime,
            on_click="ignore",
        )
