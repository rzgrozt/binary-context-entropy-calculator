from dataclasses import replace

from binary_entropy.ui.session import WorkbenchCalculationRecord
from binary_entropy.ui.workbench_state import (
    InputMode,
    MarkovWorkflow,
    MethodChoice,
    WorkbenchCalculationSuccess,
    WorkbenchForm,
    calculate_workbench,
    default_workbench_form,
)


def calculated_workspace(
    workflow: MarkovWorkflow = MarkovWorkflow.VMM,
) -> tuple[WorkbenchForm, WorkbenchCalculationRecord]:
    default = default_workbench_form()
    form = replace(
        default,
        methods=(MethodChoice.MARKOV, MethodChoice.HMM, MethodChoice.SHANNON),
        intake=replace(
            default.intake,
            mode=InputMode.BATCH,
            text="A,A,B,A\nB,A,B,B\nA,B,A,A",
        ),
        markov=replace(default.markov, workflow=workflow),
    )
    outcome = calculate_workbench(form)
    assert isinstance(outcome, WorkbenchCalculationSuccess)
    return form, WorkbenchCalculationRecord(outcome)
