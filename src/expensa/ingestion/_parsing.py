"""Shared low-level CSV parsing helpers.

Encoding detection plus German-locale date/amount parsing, used by the
bank-export loader (:mod:`expensa.ingestion.csv_loader`) and by
secondary-source adapters under :mod:`expensa.ingestion.sources`.
Kept dependency-free so adapters can reuse it without importing the bank
loader.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Encodings to try, in order. The first that decodes the whole file wins.
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


class CsvParseError(ValueError):
    """Unrecoverable problem parsing a CSV row."""


def detect_encoding(path: Path) -> str:
    """Try encodings until one decodes the entire file."""
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise CsvParseError(f"Could not decode {path} with any of {_ENCODINGS}")


def parse_german_amount(raw: str) -> Decimal:
    """German format: '1.234,56' or '-12,34' -> Decimal."""
    if raw is None or raw == "":
        raise CsvParseError("empty amount")
    s = raw.strip().replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise CsvParseError(f"bad amount {raw!r}") from e


def parse_german_date(raw: str) -> date | None:
    """Parse a date cell. Supports the formats emitted by German bank /
    payment-processor exports:

      * ``DD.MM.YYYY`` -- canonical 4-digit form (e.g. ``08.05.2026``)
      * ``DD.MM.YY``   -- 2-digit form; POSIX ``%y`` (00-68 -> 2000-2068)
      * ``YYYY-MM-DD`` -- ISO
      * ``DD/MM/YYYY`` -- slash variant
    """
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise CsvParseError(f"bad date {raw!r}")
