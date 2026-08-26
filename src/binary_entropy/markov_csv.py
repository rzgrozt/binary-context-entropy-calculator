"""Shared high-precision CSV mechanics for Markov exports."""

import csv
import io
import math
from collections.abc import Sequence

from binary_entropy.errors import NumericalInvariantError

type CsvCell = str | int | float | None


def markov_csv_text(
    columns: Sequence[str],
    rows: Sequence[Sequence[CsvCell]],
) -> str:
    """Serialize cells without rounding finite float64 values."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(tuple(_format_cell(value) for value in row))
    return buffer.getvalue()


def _format_cell(value: CsvCell) -> str:
    match value:
        case None:
            return ""
        case str() as text:
            return text
        case int() as integer:
            return str(integer)
        case float() as number:
            if math.isnan(number):
                raise NumericalInvariantError(
                    quantity="Markov CSV value",
                    value=number,
                )
            if not math.isfinite(number):
                return str(number)
            shortest_round_trip = format(number, ".17g")
            mantissa, exponent_marker, exponent = shortest_round_trip.partition("e")
            whole, decimal_marker, fraction = mantissa.partition(".")
            if not decimal_marker:
                fraction = ""
            padded_mantissa = f"{whole}.{fraction.ljust(12, '0')}"
            return (
                padded_mantissa
                if not exponent_marker
                else f"{padded_mantissa}e{exponent}"
            )
