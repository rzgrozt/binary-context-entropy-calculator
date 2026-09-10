"""Native hard-constraint controls for Stimulus Search."""

from dataclasses import dataclass
from typing import Final

import streamlit as st

from bspe import InclusiveRange, PredictedSymbol, StimulusConstraints
from bspe.ui.help_text import UI_HELP
from bspe.ui.stimulus_session import invalidate_stimulus_search

_CONSTRAINT_HELP_KEYS: Final = {
    "predicted probability": "prediction_strength",
    "predictive entropy": "predictive_entropy",
    "effective depth": "effective_depth",
    "context support": "context_support",
    "A count": "a_count",
    "A proportion": "a_proportion",
    "switches": "switches",
    "switch rate": "switch_rate",
    "longest A run": "longest_a_run",
    "longest B run": "longest_b_run",
    "longest run": "longest_run",
}

_RANGE_LABELS: Final = (
    "predicted probability",
    "predictive entropy",
    "effective depth",
    "context support",
    "A count",
    "A proportion",
    "switches",
    "switch rate",
    "longest A run",
    "longest B run",
    "longest run",
)
_INTEGER_RANGE_LABELS: Final = frozenset(
    {
        "effective depth",
        "context support",
        "A count",
        "switches",
        "longest A run",
        "longest B run",
        "longest run",
    }
)
_MAX_INTEGER_CONSTRAINT: Final = 32
CONSTRAINT_WIDGET_KEYS: Final = frozenset({"stimulus-search-predicted-symbol"}) | (
    frozenset(
        f"stimulus-search-{part}-{label.lower().replace(' ', '-')}"
        for label in _RANGE_LABELS
        for part in ("constrain", "minimum", "maximum")
    )
)


@dataclass(frozen=True, slots=True)
class _OptionalRangeControl:
    enabled: bool
    minimum: float
    maximum: float

    def inclusive_range(self) -> InclusiveRange | None:
        if not self.enabled:
            return None
        return InclusiveRange(self.minimum, self.maximum)


@dataclass(frozen=True, slots=True)
class _ConstraintControlValues:
    predicted_symbol: PredictedSymbol
    predicted_probability: _OptionalRangeControl
    predictive_entropy: _OptionalRangeControl
    effective_depth: _OptionalRangeControl
    context_support: _OptionalRangeControl
    a_count: _OptionalRangeControl
    a_proportion: _OptionalRangeControl
    switches: _OptionalRangeControl
    switch_rate: _OptionalRangeControl
    longest_a_run: _OptionalRangeControl
    longest_b_run: _OptionalRangeControl
    longest_run: _OptionalRangeControl

    def constraints(self) -> StimulusConstraints:
        return StimulusConstraints(
            predicted_symbol=self.predicted_symbol,
            predicted_probability=self.predicted_probability.inclusive_range(),
            predictive_entropy=self.predictive_entropy.inclusive_range(),
            effective_depth=self.effective_depth.inclusive_range(),
            context_support=self.context_support.inclusive_range(),
            a_count=self.a_count.inclusive_range(),
            a_proportion=self.a_proportion.inclusive_range(),
            switches=self.switches.inclusive_range(),
            switch_rate=self.switch_rate.inclusive_range(),
            longest_a_run=self.longest_a_run.inclusive_range(),
            longest_b_run=self.longest_b_run.inclusive_range(),
            longest_run=self.longest_run.inclusive_range(),
        )


def render_constraint_controls() -> _ConstraintControlValues:
    """Render every backend hard constraint without implicit defaults."""
    _ = st.caption(UI_HELP["hard_constraints"])
    selected = st.selectbox(
        "Predicted symbol",
        options=tuple(symbol.value for symbol in PredictedSymbol),
        index=2,
        key="stimulus-search-predicted-symbol",
        on_change=invalidate_stimulus_search,
        help=UI_HELP["predicted_symbol_constraint"],
    )
    predicted_symbol = PredictedSymbol(selected or PredictedSymbol.EITHER.value)
    return _ConstraintControlValues(
        predicted_symbol=predicted_symbol,
        predicted_probability=_optional_range("predicted probability", 1.0),
        predictive_entropy=_optional_range("predictive entropy", 1.0),
        effective_depth=_optional_range("effective depth", _MAX_INTEGER_CONSTRAINT),
        context_support=_optional_range("context support", _MAX_INTEGER_CONSTRAINT),
        a_count=_optional_range("A count", _MAX_INTEGER_CONSTRAINT),
        a_proportion=_optional_range("A proportion", 1.0),
        switches=_optional_range("switches", _MAX_INTEGER_CONSTRAINT),
        switch_rate=_optional_range("switch rate", 1.0),
        longest_a_run=_optional_range("longest A run", _MAX_INTEGER_CONSTRAINT),
        longest_b_run=_optional_range("longest B run", _MAX_INTEGER_CONSTRAINT),
        longest_run=_optional_range("longest run", _MAX_INTEGER_CONSTRAINT),
    )


def _optional_range(label: str, maximum: float) -> _OptionalRangeControl:
    key_label = label.lower().replace(" ", "-")
    enabled = st.checkbox(
        f"Constrain {label}",
        key=f"stimulus-search-constrain-{key_label}",
        on_change=invalidate_stimulus_search,
        help=UI_HELP[_CONSTRAINT_HELP_KEYS[label]],
    )
    if not enabled:
        return _OptionalRangeControl(enabled=False, minimum=0.0, maximum=maximum)
    number_label = f"{label} constraint" if label == "context support" else label
    whole = label in _INTEGER_RANGE_LABELS
    if whole:
        minimum = float(
            st.number_input(
                f"Minimum {number_label}",
                min_value=0,
                max_value=int(maximum),
                value=0,
                key=f"stimulus-search-minimum-{key_label}",
                on_change=invalidate_stimulus_search,
            )
        )
        upper = float(
            st.number_input(
                f"Maximum {number_label}",
                min_value=0,
                max_value=int(maximum),
                value=int(maximum),
                key=f"stimulus-search-maximum-{key_label}",
                on_change=invalidate_stimulus_search,
            )
        )
        return _OptionalRangeControl(enabled=True, minimum=minimum, maximum=upper)
    minimum = st.number_input(
        f"Minimum {number_label}",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        key=f"stimulus-search-minimum-{key_label}",
        on_change=invalidate_stimulus_search,
    )
    upper = st.number_input(
        f"Maximum {number_label}",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.001,
        format="%.3f",
        key=f"stimulus-search-maximum-{key_label}",
        on_change=invalidate_stimulus_search,
    )
    return _OptionalRangeControl(enabled=True, minimum=minimum, maximum=upper)
