"""Shared labels and boundary-preserving Streamlit data intake."""

import csv
import io
from dataclasses import dataclass
from typing import Final

import streamlit as st

from binary_entropy.batch_parsing import CsvBatchColumns
from binary_entropy.errors import BatchParseError
from binary_entropy.ui.state import ActualTargetChoice
from binary_entropy.ui.text import joined_text
from binary_entropy.ui.workbench_state import INPUT_MODE_OPTIONS, InputMode, IntakeForm

OBSERVABLE_A_KEY: Final = "shared_observable_a"
OBSERVABLE_B_KEY: Final = "shared_observable_b"
INPUT_MODE_KEY: Final = "shared_input_mode"
SINGLE_TEXT_KEY: Final = "shared_single_sequence"
BATCH_TEXT_KEY: Final = "shared_batch_sequences"
SEQUENCE_ID_KEY: Final = "shared_sequence_id"
TARGET_KEY: Final = "shared_actual_target"
TXT_UPLOAD_KEY: Final = "shared_txt_upload"
CSV_UPLOAD_KEY: Final = "shared_csv_upload"
CSV_ID_KEY: Final = "shared_csv_id_column"
CSV_SEQUENCE_KEY: Final = "shared_csv_sequence_column"
CSV_TARGET_KEY: Final = "shared_csv_target_column"
NO_TARGET_COLUMN: Final = "None"
TARGET_LABEL: Final = "Optional observed next target — for surprisal calculation only"
TARGET_HELP: Final = (
    "This selection evaluates the existing prediction and does not change it."
)


@dataclass(frozen=True, slots=True)
class CsvHeaderSuccess:
    """Strictly decoded CSV header names in source order."""

    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CsvHeaderFailure:
    """CSV header failure suitable for local UI feedback."""

    message: str


type CsvHeaderOutcome = CsvHeaderSuccess | CsvHeaderFailure


def initialize_intake_widgets() -> None:
    """Populate stable shared intake keys once with starter values."""
    defaults: tuple[tuple[str, str | ActualTargetChoice], ...] = (
        (OBSERVABLE_A_KEY, "A"),
        (OBSERVABLE_B_KEY, "B"),
        (INPUT_MODE_KEY, InputMode.SINGLE.value),
        (SINGLE_TEXT_KEY, "A, B, B, A, A, A, B"),
        (BATCH_TEXT_KEY, "A, B, B\nB, A, A"),
        (SEQUENCE_ID_KEY, "sequence-001"),
        (TARGET_KEY, ActualTargetChoice.NONE),
    )
    for key, value in defaults:
        if key not in st.session_state:
            st.session_state[key] = value


def render_observable_labels() -> tuple[str, str]:
    """Render the two shared observable labels."""
    initialize_intake_widgets()
    _ = st.subheader("Observable labels")
    columns = st.columns(2)
    label_a = columns[0].text_input("Observable A label", key=OBSERVABLE_A_KEY)
    label_b = columns[1].text_input("Observable B label", key=OBSERVABLE_B_KEY)
    _ = st.caption("Labels are trimmed, nonempty, and distinct. Spaces are allowed.")
    return label_a or "", label_b or ""


def render_intake(observable_labels: tuple[str, str]) -> IntakeForm:
    """Render one shared intake mode and preserve its native boundaries."""
    initialize_intake_widgets()
    _ = st.subheader("Data intake")
    selected_mode = st.selectbox(
        "Input mode",
        options=tuple(mode.value for mode in INPUT_MODE_OPTIONS),
        key=INPUT_MODE_KEY,
    )
    mode = InputMode(selected_mode or InputMode.SINGLE.value)
    text = ""
    payload: bytes | None = None
    csv_columns: CsvBatchColumns | None = None
    sequence_id = "sequence-001"
    match mode:
        case InputMode.SINGLE:
            text = (
                st.text_area(
                    "Observed sequence",
                    key=SINGLE_TEXT_KEY,
                    height=112,
                    help=joined_text(
                        (
                            "Commas, spaces, tabs, and line breaks belong to one ",
                            "sequence.",
                        )
                    ),
                )
                or ""
            )
            sequence_id = (
                st.text_input(
                    "Sequence ID",
                    key=SEQUENCE_ID_KEY,
                )
                or ""
            )
        case InputMode.BATCH:
            text = (
                st.text_area(
                    "Batch sequences",
                    key=BATCH_TEXT_KEY,
                    height=144,
                    help="Each nonblank physical line is one independent sequence.",
                )
                or ""
            )
        case InputMode.TXT:
            upload = st.file_uploader(
                "Upload TXT sequences",
                type=("txt",),
                accept_multiple_files=False,
                key=TXT_UPLOAD_KEY,
                help="Each nonblank physical line is one independent sequence.",
            )
            payload = upload.getvalue() if upload is not None else None
        case InputMode.CSV:
            upload = st.file_uploader(
                "Upload CSV sequences",
                type=("csv",),
                accept_multiple_files=False,
                key=CSV_UPLOAD_KEY,
            )
            payload = upload.getvalue() if upload is not None else None
            if payload is not None:
                csv_columns = _render_csv_columns(payload)
    actual_target = _render_target(observable_labels, mode)
    _ = st.caption(
        joined_text(
            (
                "Positions are reported from 1. Batch and file record boundaries ",
                "are never concatenated.",
            )
        )
    )
    return IntakeForm(
        observable_labels=observable_labels,
        mode=mode,
        text=text,
        upload_payload=payload,
        csv_columns=csv_columns,
        actual_target=actual_target,
        sequence_id=sequence_id,
    )


def parse_csv_header(payload: bytes) -> CsvHeaderOutcome:
    """Decode only the CSV header so the user can map columns explicitly."""
    try:
        text = payload.decode("utf-8-sig", errors="strict")
        row = next(csv.reader(io.StringIO(text, newline=""), strict=True), None)
    except (UnicodeDecodeError, csv.Error) as error:
        return CsvHeaderFailure(str(error))
    if row is None:
        return CsvHeaderFailure("CSV header is missing")
    columns = tuple(name.strip() for name in row)
    if any(not name for name in columns) or len(set(columns)) != len(columns):
        return CsvHeaderFailure("CSV header names must be nonempty and distinct")
    return CsvHeaderSuccess(columns)


def _render_csv_columns(payload: bytes) -> CsvBatchColumns | None:
    match parse_csv_header(payload):
        case CsvHeaderFailure(message=message):
            _ = st.error(message)
            return None
        case CsvHeaderSuccess(columns=columns):
            id_column = st.selectbox("ID column", options=columns, key=CSV_ID_KEY)
            sequence_index = 1 if len(columns) > 1 else 0
            sequence_column = st.selectbox(
                "Sequence column",
                options=columns,
                index=sequence_index,
                key=CSV_SEQUENCE_KEY,
            )
            target_column = st.selectbox(
                "Target column (optional)",
                options=(NO_TARGET_COLUMN, *columns),
                key=CSV_TARGET_KEY,
            )
    target = None if target_column == NO_TARGET_COLUMN else target_column
    try:
        return CsvBatchColumns(id_column or "", sequence_column or "", target)
    except BatchParseError as error:
        _ = st.error(str(error))
        return None


def _render_target(
    observable_labels: tuple[str, str],
    mode: InputMode,
) -> ActualTargetChoice:
    if mode is InputMode.CSV:
        return ActualTargetChoice.NONE
    labels = {
        ActualTargetChoice.NONE: "None",
        ActualTargetChoice.FIRST: observable_labels[0] or "Observable A",
        ActualTargetChoice.SECOND: observable_labels[1] or "Observable B",
    }
    target = st.radio(
        TARGET_LABEL,
        options=tuple(ActualTargetChoice),
        format_func=labels.__getitem__,
        key=TARGET_KEY,
        horizontal=True,
        help=TARGET_HELP,
    )
    return target or ActualTargetChoice.NONE
