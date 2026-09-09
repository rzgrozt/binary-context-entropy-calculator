"""Native VMM configuration controls for Stimulus Search."""

from dataclasses import dataclass
from typing import Final, assert_never

import streamlit as st

from binary_entropy import (
    AdditiveSmoothing,
    KTSmoothing,
    MLESmoothing,
    VMMConfig,
    VMMSmoothing,
)
from binary_entropy.ui.help_text import UI_HELP
from binary_entropy.ui.markov_state import VMMSmoothingChoice
from binary_entropy.ui.stimulus_session import invalidate_stimulus_search

VMM_WIDGET_KEYS: Final = frozenset(
    {
        "stimulus-search-vmm-smoothing",
        "stimulus-search-vmm-custom-alpha",
        "stimulus-search-minimum-support",
    }
)


@dataclass(frozen=True, slots=True)
class _VMMControlValues:
    smoothing_choice: VMMSmoothingChoice
    custom_alpha: float
    minimum_support: int

    def config(self) -> VMMConfig:
        return VMMConfig(
            smoothing=self._smoothing(),
            minimum_support=self.minimum_support,
        )

    def _smoothing(self) -> VMMSmoothing:
        match self.smoothing_choice:
            case VMMSmoothingChoice.KT:
                return KTSmoothing()
            case VMMSmoothingChoice.MLE:
                return MLESmoothing()
            case remaining:
                if remaining is VMMSmoothingChoice.ADDITIVE:
                    return AdditiveSmoothing(self.custom_alpha)
                assert_never(remaining)


def render_vmm_controls() -> _VMMControlValues:
    """Render every VMM setting supported by VMMConfig."""
    selected = st.selectbox(
        "VMM smoothing",
        options=tuple(choice.value for choice in VMMSmoothingChoice),
        key="stimulus-search-vmm-smoothing",
        on_change=invalidate_stimulus_search,
        help=UI_HELP["vmm_smoothing"],
    )
    choice = VMMSmoothingChoice(selected or VMMSmoothingChoice.KT.value)
    custom_alpha = 0.5
    if choice is VMMSmoothingChoice.ADDITIVE:
        custom_alpha = st.number_input(
            "Custom additive alpha",
            min_value=0.001,
            value=0.5,
            step=0.001,
            format="%.3f",
            key="stimulus-search-vmm-custom-alpha",
            on_change=invalidate_stimulus_search,
            help=UI_HELP["additive_smoothing"],
        )
    minimum_support = int(
        st.number_input(
            "Minimum context support",
            min_value=1,
            value=1,
            key="stimulus-search-minimum-support",
            on_change=invalidate_stimulus_search,
            help=UI_HELP["minimum_support"],
        )
    )
    return _VMMControlValues(choice, custom_alpha, minimum_support)
