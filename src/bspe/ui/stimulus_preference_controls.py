"""Native soft-preference controls for Stimulus Search."""

from typing import Final, assert_never

import streamlit as st

from bspe import PreferenceMetric, SoftPreference
from bspe.ui.help_text import UI_HELP
from bspe.ui.stimulus_session import invalidate_stimulus_search

PREFERENCE_WIDGET_KEYS: Final = frozenset({"stimulus-search-preferences"}) | (
    frozenset(
        f"stimulus-search-preference-{metric.value}-{part}"
        for metric in PreferenceMetric
        for part in ("target", "weight")
    )
)


def render_preference_controls(
    sequence_length: int,
) -> tuple[SoftPreference, ...]:
    """Render selected preferences and return them in enum order."""
    selected_values = st.multiselect(
        "Soft preferences",
        options=tuple(metric.value for metric in PreferenceMetric),
        key="stimulus-search-preferences",
        on_change=invalidate_stimulus_search,
        help=UI_HELP["soft_preferences"],
    )
    selected = frozenset(selected_values)
    return tuple(
        _render_preference(metric, sequence_length)
        for metric in PreferenceMetric
        if metric.value in selected
    )


def _render_preference(
    metric: PreferenceMetric,
    sequence_length: int,
) -> SoftPreference:
    label = metric.value.replace("_", " ").capitalize()
    maximum = _target_maximum(metric, sequence_length)
    target = st.number_input(
        f"{label} target",
        min_value=0.0,
        max_value=maximum,
        value=min(0.5, maximum),
        step=0.001,
        format="%.3f",
        key=f"stimulus-search-preference-{metric.value}-target",
        on_change=invalidate_stimulus_search,
        help=UI_HELP["preference_target"],
    )
    weight = st.number_input(
        f"{label} weight",
        min_value=0.0,
        value=1.0,
        step=0.001,
        format="%.3f",
        key=f"stimulus-search-preference-{metric.value}-weight",
        on_change=invalidate_stimulus_search,
        help=UI_HELP["preference_weight"],
    )
    return SoftPreference(metric, target, weight)


def _target_maximum(metric: PreferenceMetric, sequence_length: int) -> float:
    match metric:
        case (
            PreferenceMetric.PREDICTED_PROBABILITY
            | PreferenceMetric.PREDICTIVE_ENTROPY
            | PreferenceMetric.A_PROPORTION
            | PreferenceMetric.SWITCH_RATE
        ):
            return 1.0
        case PreferenceMetric.EFFECTIVE_DEPTH | PreferenceMetric.CONTEXT_SUPPORT:
            return float(sequence_length)
        case remaining:
            if remaining is PreferenceMetric.LONGEST_RUN:
                return float(sequence_length)
            assert_never(remaining)
