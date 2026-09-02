from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
TITLE: Final = "Binary Sequence Probability, Prediction & Entropy Workbench"
METHOD_OPTIONS: Final = [
    "Markov Chain",
    "Hidden Markov Model",
    "Observed Shannon Entropy",
]


def _app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    return app


def _workspace(methods: list[str] | None = None) -> AppTest:
    app = _app()
    if methods is not None:
        _ = app.multiselect[0].set_value(methods)
        _ = app.run()
    _ = next(button for button in app.button if button.label == "Continue").click()
    _ = app.run()
    assert not app.exception
    workflow = next(
        item for item in app.selectbox if item.label == "Markov workflow"
    )
    _ = workflow.set_value("First-order Markov")
    _ = app.run()
    assert not app.exception
    return app


def _calculate(app: AppTest) -> AppTest:
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    assert not app.exception
    return app


def test_setup_when_launched_defaults_to_compact_markov_selection() -> None:
    # Given / When
    app = _app()

    # Then
    assert app.title[0].value == TITLE
    assert app.multiselect[0].options == METHOD_OPTIONS
    assert app.multiselect[0].value == ["Markov Chain"]
    assert "Continue" in [button.label for button in app.button]
    assert "Calculate selected methods" not in [button.label for button in app.button]
    assert not app.number_input


def test_workspace_when_markov_is_selected_hides_hmm_controls() -> None:
    # Given / When
    app = _workspace()

    # Then
    labels = [item.label for item in app.selectbox]
    assert "Estimation method" in labels
    assert "Prefix prediction mode" in labels
    assert "Markov result scope" in labels
    assert not any("Initial probability" in item.label for item in app.number_input)
    assert "Initial hidden-state distribution" not in [
        item.label for item in app.expander
    ]


def test_workspace_when_all_methods_are_selected_shows_hmm_complements() -> None:
    # Given / When
    app = _workspace(METHOD_OPTIONS)

    # Then
    expander_labels = [item.label for item in app.expander]
    assert "Initial hidden-state distribution" in expander_labels
    assert "Transition matrix" in expander_labels
    assert "Emission matrix" in expander_labels
    initial = {item.label: item for item in app.number_input}
    assert initial["Initial probability for State 1"].disabled is False
    assert initial["Derived probability for State 2 (1 - p)"].disabled is True
    assert initial["Derived probability for State 2 (1 - p)"].value == 0.4


def test_first_order_markov_when_calculated_renders_fixture_and_downloads() -> None:
    # Given
    app = _workspace()

    # When
    app = _calculate(app)

    # Then
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["P(next A)"] == "0.500"
    assert metrics["P(next B)"] == "0.500"
    assert metrics["Predictive entropy (bits)"] == "1.000"
    downloads = [item.label for item in app.download_button]
    assert "Download Markov model JSON" in downloads
    assert "Download Markov prefix CSV" in downloads
    assert "Download Markov batch-summary CSV" in downloads


def test_markov_when_batch_is_calculated_preserves_independent_boundaries() -> None:
    # Given
    app = _workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "Batch paste"
    )
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Batch sequences"
    ).set_value("A,A\nB,B")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    markdown = "\n".join(item.value for item in app.markdown)
    assert "2 independent sequences" in markdown
    assert "2 transitions" in markdown


def test_markov_when_per_sequence_scope_shows_each_independent_fit() -> None:
    # Given
    app = _workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "Batch paste"
    )
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Batch sequences"
    ).set_value("A,A,A,B,A\nA,B,B,B,A")
    _ = next(
        item for item in app.selectbox if item.label == "Markov result scope"
    ).set_value("Per-sequence analysis")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    model_frames = [
        item.value for item in app.dataframe if "Current state" in item.value.columns
    ]
    assert len(model_frames) == 2
    assert model_frames[0]["Count next A"].tolist() == [2, 1]
    assert model_frames[1]["Count next A"].tolist() == [0, 1]


def test_markov_when_mle_is_unavailable_smoothing_recovers_matrix() -> None:
    # Given
    app = _workspace()
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("A, A")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    assert any("MLE unavailable" in item.value for item in app.warning)

    # Given
    _ = next(
        item for item in app.selectbox if item.label == "Estimation method"
    ).set_value("Laplace/add-one smoothing")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    assert not any("MLE unavailable" in item.value for item in app.warning)


def test_markov_when_prefix_mode_changes_explains_distinct_fit_semantics() -> None:
    # Given
    app = _workspace()
    mode = next(
        item for item in app.selectbox if item.label == "Prefix prediction mode"
    )

    # When
    _ = mode.set_value("Re-estimate from each prefix")
    _ = app.run()

    # Then
    text = "\n".join(item.value for item in app.markdown)
    assert "model estimates update with evidence, not higher-order memory" in text


def test_all_methods_when_calculated_compare_predictive_and_descriptive_rows() -> None:
    # Given
    app = _workspace(METHOD_OPTIONS)

    # When
    app = _calculate(app)

    # Then
    subheaders = [item.value for item in app.subheader]
    assert "Markov Chain" in subheaders
    assert "Hidden Markov Model" in subheaders
    assert "Observed-symbol Shannon entropy" in subheaders
    frames = [item.value for item in app.dataframe]
    comparison = next(frame for frame in frames if "Method" in frame.columns)
    shannon = comparison.loc[comparison["Method"] == "Observed Shannon Entropy"].iloc[0]
    assert shannon["P(next A)"] == "N/A"
    assert shannon["Prediction"] == "N/A"
    assert shannon["Observed entropy (bits)"] == "0.985"


def test_results_when_input_changes_hide_stale_numbers_and_downloads() -> None:
    # Given
    app = _calculate(_workspace())
    assert app.metric

    # When
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("A, B")
    _ = app.run()

    # Then
    assert any("Recalculation required" in item.value for item in app.warning)
    assert not app.metric
    assert not any(
        item.label.startswith("Download Markov") for item in app.download_button
    )
