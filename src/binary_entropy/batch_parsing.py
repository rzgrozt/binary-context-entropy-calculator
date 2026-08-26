"""Manual, TXT, and CSV boundaries for independent sequence batches."""

import csv
import io
from dataclasses import dataclass

from binary_entropy.domain import BinaryLabels, ObservableIndex
from binary_entropy.errors import (
    BatchParseError,
    BatchParseErrorCode,
    BatchRecordError,
    BatchRecordIssue,
    InvalidSequenceTokenError,
)
from binary_entropy.parsing import parse_sequence
from binary_entropy.records import SequenceDataset, SequenceRecord


@dataclass(frozen=True, slots=True, init=False)
class CsvBatchColumns:
    """Explicit CSV columns for identifiers, sequences, and optional targets."""

    id_column: str
    sequence_column: str
    actual_target_column: str | None

    def __init__(
        self,
        id_column: str,
        sequence_column: str,
        actual_target_column: str | None = None,
    ) -> None:
        """Trim and require distinct nonempty mapped column names."""
        normalized_id = id_column.strip()
        normalized_sequence = sequence_column.strip()
        normalized_target = (
            actual_target_column.strip() if actual_target_column is not None else None
        )
        names = (normalized_id, normalized_sequence)
        if normalized_target is not None:
            names += (normalized_target,)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise BatchParseError(
                code=BatchParseErrorCode.INVALID_COLUMNS,
                detail="mapped column names must be nonempty and distinct",
            )
        object.__setattr__(self, "id_column", normalized_id)
        object.__setattr__(self, "sequence_column", normalized_sequence)
        object.__setattr__(self, "actual_target_column", normalized_target)


@dataclass(frozen=True, slots=True)
class _CsvLayout:
    width: int
    id_index: int
    sequence_index: int
    target_index: int | None


@dataclass(frozen=True, slots=True)
class _CsvRow:
    number: int
    cells: tuple[str, ...]
    layout: _CsvLayout

    @property
    def record_id(self) -> str | None:
        if self.layout.id_index >= len(self.cells):
            return None
        return self.cells[self.layout.id_index].strip() or None

    def issue(
        self,
        code: BatchParseErrorCode,
        detail: str,
        token_position: int | None = None,
    ) -> BatchRecordIssue:
        return BatchRecordIssue(
            self.number, self.record_id, token_position, code, detail
        )


def parse_manual_batch(text: str, labels: BinaryLabels) -> SequenceDataset:
    """Parse nonblank physical lines as independent records."""
    records: list[SequenceRecord] = []
    issues: list[BatchRecordIssue] = []
    record_number = 0
    for row_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record_number += 1
        record_id = f"sequence-{record_number:03d}"
        try:
            sequence = parse_sequence(line, labels)
        except InvalidSequenceTokenError as error:
            issues.append(
                BatchRecordIssue(
                    row=row_number,
                    record_id=record_id,
                    token_position=error.position,
                    code=BatchParseErrorCode.INVALID_SEQUENCE,
                    detail=str(error),
                )
            )
        else:
            records.append(SequenceRecord(record_id, sequence))
    if issues:
        raise BatchRecordError(tuple(issues))
    return SequenceDataset(labels, records)


def parse_txt_batch(payload: bytes, labels: BinaryLabels) -> SequenceDataset:
    """Decode strict UTF-8 TXT bytes and parse independent physical lines."""
    return parse_manual_batch(_decode_utf8(payload), labels)


def parse_csv_batch(
    payload: bytes,
    labels: BinaryLabels,
    columns: CsvBatchColumns,
) -> SequenceDataset:
    """Decode and atomically parse explicitly mapped CSV columns."""
    text = _decode_utf8(payload)
    try:
        parsed_rows = tuple(
            tuple(row) for row in csv.reader(io.StringIO(text, newline=""), strict=True)
        )
    except csv.Error as error:
        raise BatchParseError(BatchParseErrorCode.MALFORMED_ROW, str(error)) from error
    if not parsed_rows or not parsed_rows[0]:
        raise BatchParseError(
            BatchParseErrorCode.MISSING_COLUMN, "CSV header is missing", 1
        )
    header = tuple(name.strip() for name in parsed_rows[0])
    if any(not name for name in header) or len(set(header)) != len(header):
        raise BatchParseError(
            BatchParseErrorCode.INVALID_COLUMNS,
            "CSV header names must be nonempty and distinct",
            1,
        )
    mapped_names = (columns.id_column, columns.sequence_column)
    if columns.actual_target_column is not None:
        mapped_names += (columns.actual_target_column,)
    for name in mapped_names:
        if name not in header:
            raise BatchParseError(
                BatchParseErrorCode.MISSING_COLUMN,
                f"required column {name!r} is missing",
                1,
            )
    layout = _CsvLayout(
        width=len(header),
        id_index=header.index(columns.id_column),
        sequence_index=header.index(columns.sequence_column),
        target_index=(
            header.index(columns.actual_target_column)
            if columns.actual_target_column is not None
            else None
        ),
    )
    records, issues = _parse_csv_records(parsed_rows[1:], labels, layout)
    if issues:
        raise BatchRecordError(tuple(issues))
    return SequenceDataset(labels, records)


def _parse_csv_records(
    rows: tuple[tuple[str, ...], ...],
    labels: BinaryLabels,
    layout: _CsvLayout,
) -> tuple[list[SequenceRecord], list[BatchRecordIssue]]:
    records: list[SequenceRecord] = []
    issues: list[BatchRecordIssue] = []
    seen_ids: set[str] = set()
    for row_number, cells in enumerate(rows, start=2):
        if not cells:
            continue
        record_id, record, row_issues = _parse_csv_record(
            _CsvRow(row_number, cells, layout), labels, seen_ids
        )
        if record_id is not None:
            seen_ids.add(record_id)
        issues.extend(row_issues)
        if record is not None:
            records.append(record)
    return records, issues


def _parse_csv_record(
    csv_row: _CsvRow,
    labels: BinaryLabels,
    seen_ids: set[str],
) -> tuple[str | None, SequenceRecord | None, tuple[BatchRecordIssue, ...]]:
    record_id = csv_row.record_id
    issues: list[BatchRecordIssue] = []
    if len(csv_row.cells) != csv_row.layout.width:
        issues.append(
            csv_row.issue(
                BatchParseErrorCode.MALFORMED_ROW,
                f"expected {csv_row.layout.width} cells; got {len(csv_row.cells)}",
            )
        )
    id_issue = _csv_id_issue(csv_row, seen_ids)
    if id_issue is not None:
        issues.append(id_issue)
    sequence: tuple[ObservableIndex, ...] = ()
    if csv_row.layout.sequence_index < len(csv_row.cells):
        sequence, sequence_issue = _parse_sequence_cell(
            csv_row.cells[csv_row.layout.sequence_index],
            labels,
            (csv_row, BatchParseErrorCode.INVALID_SEQUENCE),
        )
        if sequence_issue is not None:
            issues.append(sequence_issue)
    actual_target_index: ObservableIndex | None = None
    target_index = csv_row.layout.target_index
    if target_index is not None and target_index < len(csv_row.cells):
        target_sequence, target_issue = _parse_sequence_cell(
            csv_row.cells[target_index],
            labels,
            (csv_row, BatchParseErrorCode.INVALID_TARGET),
        )
        if target_issue is not None:
            issues.append(target_issue)
        elif len(target_sequence) > 1:
            issues.append(
                csv_row.issue(
                    BatchParseErrorCode.INVALID_TARGET,
                    "actual target must contain at most one symbol",
                    token_position=2,
                )
            )
        elif target_sequence:
            actual_target_index = target_sequence[0]
    record = (
        SequenceRecord(record_id, sequence, actual_target_index)
        if not issues and record_id is not None
        else None
    )
    return record_id, record, tuple(issues)


def _csv_id_issue(
    csv_row: _CsvRow,
    seen_ids: set[str],
) -> BatchRecordIssue | None:
    if csv_row.layout.id_index >= len(csv_row.cells):
        return None
    record_id = csv_row.record_id
    if record_id is None:
        return csv_row.issue(
            BatchParseErrorCode.INVALID_RECORD_ID, "record ID must be nonblank"
        )
    if record_id in seen_ids:
        return csv_row.issue(
            BatchParseErrorCode.DUPLICATE_RECORD_ID,
            f"record ID {record_id!r} duplicates an earlier row",
        )
    return None


def _parse_sequence_cell(
    text: str,
    labels: BinaryLabels,
    location: tuple[_CsvRow, BatchParseErrorCode],
) -> tuple[tuple[ObservableIndex, ...], BatchRecordIssue | None]:
    try:
        return parse_sequence(text, labels), None
    except InvalidSequenceTokenError as error:
        csv_row, code = location
        return (), csv_row.issue(code, str(error), error.position)


def _decode_utf8(payload: bytes) -> str:
    try:
        return payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise BatchParseError(BatchParseErrorCode.INVALID_UTF8, str(error)) from error
