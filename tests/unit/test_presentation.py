import pandas as pd  # noqa: PANDAS_OK

from binary_entropy.analysis import analyze_sequence
from binary_entropy.presentation import analysis_dataframe, format_decimal
from tests.unit.helpers import hand_model, hand_sequence


def test_format_decimal_when_value_varies_uses_twelve_decimal_places() -> None:
    assert format_decimal(0.62) == "0.620000000000"
    assert format_decimal(1.0 / 3.0) == "0.333333333333"
    assert format_decimal(float("inf")) == "inf"


def test_analysis_dataframe_when_using_hand_sequence_has_stable_columns() -> None:
    model = hand_model()
    analysis = analyze_sequence(model, hand_sequence())

    result = analysis_dataframe(analysis, model)

    assert result.columns.tolist() == [
        "depth",
        "observed_symbol",
        "next_target_symbol",
        "predictive_probability_A",
        "predictive_probability_B",
        "predictive_entropy_bits",
        "predicted_symbol",
        "actual_target_probability",
        "actual_target_surprisal_bits",
        "target_classification",
        "posterior_State 1",
        "posterior_State 2",
        "next_hidden_State 1",
        "next_hidden_State 2",
    ]
    assert result["depth"].tolist() == list(range(8))
    assert pd.isna(result.loc[0, "posterior_State 1"])
    assert result.loc[0, "next_hidden_State 1"] == 0.6
    assert pd.isna(result.loc[7, "next_target_symbol"])


def test_analysis_dataframe_when_rows_are_supplied_in_order_remains_depth_sorted() -> (
    None
):
    model = hand_model()
    analysis = analyze_sequence(model, hand_sequence())

    result = analysis_dataframe(analysis, model)

    assert result["depth"].is_monotonic_increasing
