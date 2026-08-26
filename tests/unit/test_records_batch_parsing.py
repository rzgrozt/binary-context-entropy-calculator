import pytest

from binary_entropy.batch_parsing import (
    CsvBatchColumns,
    parse_csv_batch,
    parse_manual_batch,
    parse_txt_batch,
)
from binary_entropy.domain import BinaryLabels
from binary_entropy.errors import (
    BatchParseError,
    BatchParseErrorCode,
    BatchRecordError,
    BatchRecordIssue,
    DatasetValidationError,
)
from binary_entropy.records import SequenceDataset, SequenceRecord


def _labels() -> BinaryLabels:
    return BinaryLabels(states=("S1", "S2"), observables=("A", "B"))


def test_sequence_record_when_id_has_whitespace_trims_and_allows_empty_sequence() -> (
    None
):
    # Given
    sequence_id = "  candidate-1  "

    # When
    result = SequenceRecord(sequence_id, ())

    # Then
    assert result.sequence_id == "candidate-1"
    assert result.sequence == ()


def test_sequence_dataset_when_no_records_rejects_empty_collection() -> None:
    # Given
    records: tuple[SequenceRecord, ...] = ()

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceDataset(_labels(), records)


def test_sequence_dataset_when_trimmed_ids_repeat_rejects_duplicate() -> None:
    # Given
    records = (
        SequenceRecord("candidate-1", (0,)),
        SequenceRecord(" candidate-1 ", (1,)),
    )

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceDataset(_labels(), records)


def test_sequence_record_when_id_is_blank_rejects_record() -> None:
    # Given
    sequence_id = " \t "

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = SequenceRecord(sequence_id, ())


def test_parse_manual_batch_when_lines_are_nonblank_keeps_them_independent() -> None:
    # Given
    text = "A,A,B\n\n B,A \n"

    # When
    result = parse_manual_batch(text, _labels())

    # Then
    assert tuple(record.sequence for record in result.records) == ((0, 0, 1), (1, 0))
    assert tuple(record.sequence_id for record in result.records) == (
        "sequence-001",
        "sequence-002",
    )


def test_parse_manual_batch_when_all_lines_are_blank_rejects_empty_dataset() -> None:
    # Given
    text = " \n\t\n"

    # When / Then
    with pytest.raises(DatasetValidationError):
        _ = parse_manual_batch(text, _labels())


def test_parse_manual_batch_when_multiple_lines_are_invalid_reports_every_line() -> (
    None
):
    # Given
    text = "A,C\n\nB\nA,D"

    # When / Then
    with pytest.raises(BatchRecordError) as captured:
        _ = parse_manual_batch(text, _labels())

    assert captured.value.issues == (
        BatchRecordIssue(
            row=1,
            record_id="sequence-001",
            token_position=2,
            code=BatchParseErrorCode.INVALID_SEQUENCE,
            detail="invalid sequence token 'C' at position 2",
        ),
        BatchRecordIssue(
            row=4,
            record_id="sequence-003",
            token_position=2,
            code=BatchParseErrorCode.INVALID_SEQUENCE,
            detail="invalid sequence token 'D' at position 2",
        ),
    )


def test_parse_txt_batch_when_utf8_has_bom_decodes_strictly() -> None:
    # Given
    payload = b"\xef\xbb\xbfA B\nB\n"

    # When
    result = parse_txt_batch(payload, _labels())

    # Then
    assert tuple(record.sequence for record in result.records) == ((0, 1), (1,))


def test_parse_txt_batch_when_utf8_is_invalid_rejects_entire_payload() -> None:
    # Given
    payload = b"A\n\xff\nB"

    # When / Then
    with pytest.raises(BatchParseError):
        _ = parse_txt_batch(payload, _labels())


def test_parse_txt_batch_when_multiple_lines_are_invalid_reports_physical_lines() -> (
    None
):
    # Given
    payload = b"A\n\nB,C\nD\n"

    # When / Then
    with pytest.raises(BatchRecordError) as captured:
        _ = parse_txt_batch(payload, _labels())

    assert tuple(
        (issue.row, issue.record_id, issue.token_position)
        for issue in captured.value.issues
    ) == ((3, "sequence-002", 2), (4, "sequence-003", 1))


def test_csv_columns_when_names_are_not_distinct_rejects_mapping() -> None:
    # Given
    duplicate_name = "sequence"

    # When / Then
    with pytest.raises(BatchParseError):
        _ = CsvBatchColumns(duplicate_name, duplicate_name)


def test_parse_csv_batch_when_columns_are_explicit_parses_ids_sequences_targets() -> (
    None
):
    # Given
    payload = b'\xef\xbb\xbfsubject,observations,next\n alpha ,"A,B,A",B\nbeta,,\n'
    columns = CsvBatchColumns("subject", "observations", "next")

    # When
    result = parse_csv_batch(payload, _labels(), columns)

    # Then
    assert tuple(record.sequence_id for record in result.records) == ("alpha", "beta")
    assert tuple(record.sequence for record in result.records) == ((0, 1, 0), ())
    assert tuple(record.actual_target_index for record in result.records) == (1, None)


def test_parse_csv_batch_when_required_header_is_missing_rejects_payload() -> None:
    # Given
    payload = b"subject,wrong\nalpha,A\n"
    columns = CsvBatchColumns("subject", "observations")

    # When / Then
    with pytest.raises(BatchParseError):
        _ = parse_csv_batch(payload, _labels(), columns)


def test_parse_csv_batch_when_late_sequence_is_invalid_fails_atomically() -> None:
    # Given
    payload = b"id,sequence\nfirst,A\nsecond,C\n"
    columns = CsvBatchColumns("id", "sequence")

    # When / Then
    with pytest.raises(BatchRecordError) as captured:
        _ = parse_csv_batch(payload, _labels(), columns)

    assert captured.value.issues == (
        BatchRecordIssue(
            row=3,
            record_id="second",
            token_position=1,
            code=BatchParseErrorCode.INVALID_SEQUENCE,
            detail="invalid sequence token 'C' at position 1",
        ),
    )


def test_parse_csv_batch_when_target_has_multiple_symbols_rejects_target() -> None:
    # Given
    payload = b'id,sequence,target\nfirst,A,"A,B"\n'
    columns = CsvBatchColumns("id", "sequence", "target")

    # When / Then
    with pytest.raises(BatchRecordError) as captured:
        _ = parse_csv_batch(payload, _labels(), columns)

    assert captured.value.issues[0].code is BatchParseErrorCode.INVALID_TARGET
    assert captured.value.issues[0].token_position == 2


def test_parse_csv_batch_when_rows_have_independent_issues_reports_all_in_order() -> (
    None
):
    # Given
    payload = (
        b'id,sequence,target\nalpha,A,A\nbeta,C,D\n   ,B,B\n alpha ,A,"A,B"\nshort,A\n'
    )
    columns = CsvBatchColumns("id", "sequence", "target")

    # When / Then
    with pytest.raises(BatchRecordError) as captured:
        _ = parse_csv_batch(payload, _labels(), columns)

    assert captured.value.issues == (
        BatchRecordIssue(
            row=3,
            record_id="beta",
            token_position=1,
            code=BatchParseErrorCode.INVALID_SEQUENCE,
            detail="invalid sequence token 'C' at position 1",
        ),
        BatchRecordIssue(
            row=3,
            record_id="beta",
            token_position=1,
            code=BatchParseErrorCode.INVALID_TARGET,
            detail="invalid sequence token 'D' at position 1",
        ),
        BatchRecordIssue(
            row=4,
            record_id=None,
            token_position=None,
            code=BatchParseErrorCode.INVALID_RECORD_ID,
            detail="record ID must be nonblank",
        ),
        BatchRecordIssue(
            row=5,
            record_id="alpha",
            token_position=None,
            code=BatchParseErrorCode.DUPLICATE_RECORD_ID,
            detail="record ID 'alpha' duplicates an earlier row",
        ),
        BatchRecordIssue(
            row=5,
            record_id="alpha",
            token_position=2,
            code=BatchParseErrorCode.INVALID_TARGET,
            detail="actual target must contain at most one symbol",
        ),
        BatchRecordIssue(
            row=6,
            record_id="short",
            token_position=None,
            code=BatchParseErrorCode.MALFORMED_ROW,
            detail="expected 3 cells; got 2",
        ),
    )


def test_batch_record_errors_when_rendered_has_stable_actionable_lines() -> None:
    # Given
    error = BatchRecordError(
        (
            BatchRecordIssue(
                row=2,
                record_id="alpha",
                token_position=2,
                code=BatchParseErrorCode.INVALID_SEQUENCE,
                detail="invalid sequence token 'C'",
            ),
            BatchRecordIssue(
                row=4,
                record_id=None,
                token_position=None,
                code=BatchParseErrorCode.INVALID_RECORD_ID,
                detail="record ID must be nonblank",
            ),
        )
    )

    # When
    result = str(error)

    # Then
    assert result == (
        "batch parse failed with 2 record issues:\n"
        "- row 2, record 'alpha', token 2 (invalid_sequence): "
        "invalid sequence token 'C'\n"
        "- row 4, record <unavailable> (invalid_record_id): "
        "record ID must be nonblank"
    )


def test_parse_csv_batch_when_quoting_is_malformed_fails_immediately() -> None:
    # Given
    payload = b'id,sequence\nalpha,"A\n'
    columns = CsvBatchColumns("id", "sequence")

    # When / Then
    with pytest.raises(BatchParseError) as captured:
        _ = parse_csv_batch(payload, _labels(), columns)

    assert captured.value.code is BatchParseErrorCode.MALFORMED_ROW


def test_parse_csv_batch_when_header_names_repeat_fails_immediately() -> None:
    # Given
    payload = b"id,sequence,id\nalpha,A,other\n"
    columns = CsvBatchColumns("id", "sequence")

    # When / Then
    with pytest.raises(BatchParseError) as captured:
        _ = parse_csv_batch(payload, _labels(), columns)

    assert captured.value.code is BatchParseErrorCode.INVALID_COLUMNS
