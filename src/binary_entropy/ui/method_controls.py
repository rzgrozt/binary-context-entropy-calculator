"""Selected-method configuration controls."""

from typing import Final

import streamlit as st

from binary_entropy.markov_types import MarkovPredictionMode, MarkovResultScope
from binary_entropy.ui.hmm_session import (
    PRESET_NAME_KEY,
    STATE_0_KEY,
    STATE_1_KEY,
    hmm_control_state,
    hydrate_hmm_widgets,
    store_hmm_controls,
    stored_hmm_model,
)
from binary_entropy.ui.model_inputs import render_probability_inputs
from binary_entropy.ui.preset_inputs import render_preset_import
from binary_entropy.ui.state import (
    ActualTargetChoice,
    CalculatorForm,
    ModelForm,
)
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.tokens import UI_NUMBER_FORMAT
from binary_entropy.ui.workbench_state import (
    ESTIMATION_OPTIONS,
    MarkovControls,
    MarkovEstimationChoice,
    MarkovWorkflow,
    VMMSmoothingChoice,
)

MARKOV_WORKFLOW_KEY: Final = "markov_workflow"
MARKOV_ESTIMATION_KEY: Final = "markov_estimation"
MARKOV_ALPHA_KEY: Final = "markov_custom_alpha"
MARKOV_PREFIX_KEY: Final = "markov_prefix_mode"
MARKOV_SCOPE_KEY: Final = "markov_result_scope"
VMM_SMOOTHING_KEY: Final = "vmm_smoothing"
VMM_ALPHA_KEY: Final = "vmm_custom_alpha"
VMM_SUPPORT_KEY: Final = "vmm_minimum_support"
FIXED_PREFIX_LABEL: Final = "Fixed fitted transition matrix"
CUMULATIVE_PREFIX_LABEL: Final = "Re-estimate from each prefix"
POOLED_SCOPE_LABEL: Final = "Pooled model"
PER_SEQUENCE_SCOPE_LABEL: Final = "Per-sequence analysis"


def render_markov_controls() -> MarkovControls:
    """Render one explicit Markov workflow and only its applicable controls."""
    _ = st.subheader("Markov Chain controls")
    workflow_value = st.selectbox(
        "Markov workflow",
        options=tuple(workflow.value for workflow in MarkovWorkflow),
        key=MARKOV_WORKFLOW_KEY,
    )
    workflow = MarkovWorkflow(workflow_value or MarkovWorkflow.VMM.value)

    estimation = MarkovEstimationChoice.MAXIMUM_LIKELIHOOD
    custom_alpha = 0.5
    prediction_mode = MarkovPredictionMode.FIXED_MODEL
    vmm_smoothing_choice = VMMSmoothingChoice.KT
    vmm_custom_alpha = 0.5
    minimum_support = 2

    match workflow:
        case MarkovWorkflow.VMM:
            _ = st.caption(
                joined_text(
                    (
                        "Fits finite suffix contexts and selects the deepest context ",
                        "with the configured minimum support.",
                    )
                )
            )
            smoothing_value = st.selectbox(
                "VMM smoothing",
                options=tuple(choice.value for choice in VMMSmoothingChoice),
                key=VMM_SMOOTHING_KEY,
            )
            vmm_smoothing_choice = VMMSmoothingChoice(
                smoothing_value or VMMSmoothingChoice.KT.value
            )
            if vmm_smoothing_choice is VMMSmoothingChoice.ADDITIVE:
                vmm_custom_alpha = st.number_input(
                    "Custom additive alpha",
                    min_value=0.001,
                    value=0.5,
                    step=0.001,
                    format=UI_NUMBER_FORMAT,
                    key=VMM_ALPHA_KEY,
                )
            minimum_support = st.number_input(
                "Minimum context support",
                min_value=1,
                value=2,
                step=1,
                key=VMM_SUPPORT_KEY,
            )
            _ = st.caption(
                joined_text(
                    (
                        "Unsupported suffixes back off to shorter supported contexts; ",
                        "records stay independent and are never concatenated.",
                    )
                )
            )
        case MarkovWorkflow.FIRST_ORDER:
            estimation_value = st.selectbox(
                "Estimation method",
                options=tuple(option.value for option in ESTIMATION_OPTIONS),
                key=MARKOV_ESTIMATION_KEY,
            )
            estimation = MarkovEstimationChoice(
                estimation_value
                or MarkovEstimationChoice.MAXIMUM_LIKELIHOOD.value
            )
            if estimation is MarkovEstimationChoice.CUSTOM:
                custom_alpha = st.number_input(
                    "Custom smoothing alpha",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                    format=UI_NUMBER_FORMAT,
                    key=MARKOV_ALPHA_KEY,
                )
            _ = st.text_input("Markov order", value="1", disabled=True)
            prefix_value = st.selectbox(
                "Prefix prediction mode",
                options=(FIXED_PREFIX_LABEL, CUMULATIVE_PREFIX_LABEL),
                key=MARKOV_PREFIX_KEY,
            )
            if prefix_value == CUMULATIVE_PREFIX_LABEL:
                prediction_mode = MarkovPredictionMode.CUMULATIVE_PREFIX
                _ = st.markdown(
                    joined_text(
                        (
                            "Cumulative mode: model estimates update with evidence, ",
                            "not higher-order memory.",
                        )
                    )
                )
            else:
                _ = st.markdown(
                    joined_text(
                        (
                            "Fixed mode: only the final prefix state changes row ",
                            "selection; the fitted transition matrix stays fixed.",
                        )
                    )
                )
    scope_value = st.selectbox(
        "Markov result scope",
        options=(POOLED_SCOPE_LABEL, PER_SEQUENCE_SCOPE_LABEL),
        key=MARKOV_SCOPE_KEY,
    )
    result_scope = (
        MarkovResultScope.PER_SEQUENCE
        if scope_value == PER_SEQUENCE_SCOPE_LABEL
        else MarkovResultScope.POOLED
    )
    return MarkovControls(
        estimation=estimation,
        custom_alpha=custom_alpha,
        prediction_mode=prediction_mode,
        result_scope=result_scope,
        workflow=workflow,
        vmm_smoothing_choice=vmm_smoothing_choice,
        vmm_custom_alpha=vmm_custom_alpha,
        minimum_support=minimum_support,
    )


def default_markov_controls() -> MarkovControls:
    """Return hidden Markov control defaults when Markov is not selected."""
    return MarkovControls(
        estimation=MarkovEstimationChoice.MAXIMUM_LIKELIHOOD,
        custom_alpha=0.5,
        prediction_mode=MarkovPredictionMode.FIXED_MODEL,
        result_scope=MarkovResultScope.POOLED,
        workflow=MarkovWorkflow.VMM,
        vmm_smoothing_choice=VMMSmoothingChoice.KT,
        vmm_custom_alpha=0.5,
        minimum_support=2,
    )


def render_hmm_controls(observable_labels: tuple[str, str]) -> tuple[ModelForm, str]:
    """Render state labels, complement rows, and schema-v1 preset controls."""
    hydrate_hmm_widgets()
    _ = st.subheader("Hidden Markov Model controls")
    state_columns = st.columns(2)
    state_0 = state_columns[0].text_input("Hidden state 1 label", key=STATE_0_KEY)
    state_1 = state_columns[1].text_input("Hidden state 2 label", key=STATE_1_KEY)
    state_labels = state_0 or "", state_1 or ""
    model = render_probability_inputs(state_labels, observable_labels)
    preset_name = st.text_input("Preset name", key=PRESET_NAME_KEY) or ""
    current = CalculatorForm(
        model=model,
        sequence_text="",
        actual_target=ActualTargetChoice.NONE,
        sequence_id="sequence-001",
        preset_name=preset_name,
    )
    store_hmm_controls(model, preset_name)
    render_preset_import(current)
    return model, preset_name


def hidden_hmm_controls(observable_labels: tuple[str, str]) -> tuple[ModelForm, str]:
    """Read preserved HMM values without creating unselected controls."""
    state = hmm_control_state()
    return stored_hmm_model(observable_labels), state.preset_name
