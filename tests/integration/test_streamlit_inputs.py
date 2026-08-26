from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
TARGET_LABEL: Final = "Optional observed next target — for surprisal calculation only"


def _workspace() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    _ = next(button for button in app.button if button.label == "Continue").click()
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


def test_target_when_single_input_is_visible_states_evaluation_only_contract() -> None:
    # Given / When
    app = _workspace()

    # Then
    target = next(item for item in app.radio if item.label == TARGET_LABEL)
    assert "evaluates the existing prediction" in target.help
    assert "does not change it" in target.help


def test_custom_smoothing_when_selected_reveals_only_then_numeric_alpha() -> None:
    # Given
    app = _workspace()
    assert "Custom smoothing alpha" not in [item.label for item in app.number_input]

    # When
    _ = next(
        item for item in app.selectbox if item.label == "Estimation method"
    ).set_value("Custom additive smoothing alpha")
    _ = app.run()

    # Then
    assert "Custom smoothing alpha" in [item.label for item in app.number_input]


def test_txt_upload_when_calculated_preserves_physical_line_boundaries() -> None:
    # Given
    app = _workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "TXT upload"
    )
    _ = app.run()
    _ = app.file_uploader[0].set_value(("sequences.txt", b"A,A\nB,B\n", "text/plain"))
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    text = "\n".join(item.value for item in app.markdown)
    assert "2 independent sequences" in text
    assert "2 transitions" in text


def test_batch_errors_when_calculated_show_one_atomic_aggregate_failure() -> None:
    # Given
    app = _workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "Batch paste"
    )
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Batch sequences"
    ).set_value("A,A\nA,C\nB,D")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    assert len(app.error) == 1
    message = app.error[0].value
    assert "sequence-002" in message
    assert "sequence-003" in message
    assert not app.metric
    assert not app.dataframe


def test_csv_upload_when_present_exposes_explicit_columns_and_row_targets() -> None:
    # Given
    app = _workspace()
    _ = next(item for item in app.selectbox if item.label == "Input mode").set_value(
        "CSV upload"
    )
    _ = app.run()
    payload = b'id,sequence,target\nalpha,"A,A,B",A\nbeta,"B,A",B\n'
    _ = app.file_uploader[0].set_value(("sequences.csv", payload, "text/csv"))
    _ = app.run()

    # When
    selectors = {item.label: item for item in app.selectbox}
    _ = selectors["ID column"].set_value("id")
    _ = selectors["Sequence column"].set_value("sequence")
    _ = selectors["Target column (optional)"].set_value("target")
    _ = app.run()
    app = _calculate(app)

    # Then
    frames = [item.value for item in app.dataframe]
    summary = next(frame for frame in frames if "Sequence ID" in frame.columns)
    assert summary["Sequence ID"].tolist() == ["alpha", "beta"]
    assert summary["Observed target"].tolist() == ["A", "B"]


def test_observable_labels_when_they_contain_spaces_parse_as_whole_symbols() -> None:
    # Given
    app = _workspace()
    labels = {item.label: item for item in app.text_input}
    _ = labels["Observable A label"].set_value("light red")
    _ = labels["Observable B label"].set_value("deep blue")
    _ = app.run()
    _ = next(
        item for item in app.text_area if item.label == "Observed sequence"
    ).set_value("light red, deep blue\nlight red")
    _ = app.run()

    # When
    app = _calculate(app)

    # Then
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "P(next light red)" in metrics
    assert "P(next deep blue)" in metrics
