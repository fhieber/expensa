"""Parse German bank-export CSVs into ParsedRow records.

The format is the "Buchungen-Export" produced by most German banks:

    Buchungsdatum;Wertstellung;Status;Zahlungspflichtige*r;Zahlungsempfänger*in;
    Verwendungszweck;Umsatztyp;IBAN;Betrag (€);Gläubiger-ID;Mandatsreferenz;Kundenreferenz

- ``;`` separator, ``,`` decimal.
- Date format ``DD.MM.YYYY``.
- Encoding is usually cp1252 or utf-8-sig — we autodetect.
- A few banks prefix metadata rows before the header; we skip until we find the
  header row.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from expensa.ingestion._parsing import (
    CsvParseError,
    detect_encoding,
    parse_german_amount,
    parse_german_date,
)

# Many banks (and some PayPal exports) emit multi-line cell content using
# embedded newlines or tabs that get unquoted into the cell as whitespace
# runs -- the UI then shows "PayPal Europe    22-24 Boulevard..." with
# huge gaps. The normalized columns already collapse whitespace, but the
# raw column we display does not, so we do it here at ingest time.
_WS_RUN = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    """Strip leading/trailing whitespace and collapse any internal run of
    whitespace (spaces, tabs, embedded newlines) to a single space.

    Idempotent. Returns ``""`` for None / empty so caller code can pass
    raw dict lookups straight in.
    """
    if not s:
        return ""
    return _WS_RUN.sub(" ", s).strip()

# Private aliases kept so existing imports/tests (e.g. csv_loader._parse_amount)
# keep working after the parsing helpers moved to _parsing.
_detect_encoding = detect_encoding
_parse_amount = parse_german_amount
_parse_date = parse_german_date

# Re-exported for backward compatibility with existing imports/tests.
__all__ = [
    "EXPECTED_HEADERS",
    "ParsedRow",
    "CsvParseError",
    "parse_csv",
]

# The canonical headers we expect, after lowercasing + ASCII-folding the asterisks/umlauts.
EXPECTED_HEADERS = [
    "buchungsdatum",
    "wertstellung",
    "status",
    "zahlungspflichtiger",
    "zahlungsempfaenger",
    "verwendungszweck",
    "umsatztyp",
    "iban",
    "betrag",
    "glaeubiger_id",
    "mandatsreferenz",
    "kundenreferenz",
]


@dataclass(frozen=True)
class ParsedRow:
    """One row of a German bank export, with native types."""

    buchungsdatum: date
    wertstellung: date | None
    status: str
    zahlungspflichtiger: str
    zahlungsempfaenger: str
    verwendungszweck: str
    umsatztyp: str
    iban: str
    betrag: Decimal
    glaeubiger_id: str
    mandatsreferenz: str
    kundenreferenz: str
    source_file: str
    source_row: int  # 1-based index in the source file (excluding skipped header)

    @property
    def betrag_cents(self) -> int:
        return int((self.betrag * 100).to_integral_value())


def _normalize_header(h: str) -> str:
    """Map the German header (with umlauts and asterisks) to a Python-friendly key."""
    s = h.strip().lower()
    s = s.replace("*r", "r").replace("*in", "")  # Zahlungspflichtige*r -> zahlungspflichtiger
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = s.replace(" (€)", "").replace("(€)", "").strip()
    s = s.replace("-", "_").replace(" ", "_")
    return s


def _find_header_index(reader_rows: list[list[str]]) -> int:
    """Some banks prefix metadata. Find the first row that looks like the header."""
    target_first = "buchungsdatum"
    for i, row in enumerate(reader_rows):
        if row and _normalize_header(row[0]) == target_first:
            return i
    raise CsvParseError("Could not find header row starting with 'Buchungsdatum'")


def parse_csv(path: Path) -> list[ParsedRow]:
    """Parse a German bank-export CSV. Returns a list of ParsedRow.

    Raises CsvParseError on the first unrecoverable row. (We intentionally
    don't silently skip — the caller can wrap in try/except per file.)
    """
    encoding = _detect_encoding(path)
    text = path.read_text(encoding=encoding)
    reader = csv.reader(text.splitlines(), delimiter=";", quotechar='"')
    all_rows = list(reader)
    if not all_rows:
        return []

    header_idx = _find_header_index(all_rows)
    headers = [_normalize_header(h) for h in all_rows[header_idx]]

    # Sanity: every expected column must be present (some optional ones may not).
    for col in ("buchungsdatum", "betrag", "verwendungszweck"):
        if col not in headers:
            raise CsvParseError(f"missing required column {col!r} in {path.name}")

    out: list[ParsedRow] = []
    for offset, raw_row in enumerate(all_rows[header_idx + 1 :], start=1):
        if not any(cell.strip() for cell in raw_row):
            continue  # blank line
        # Pad short rows so dict mapping doesn't fail.
        if len(raw_row) < len(headers):
            raw_row = raw_row + [""] * (len(headers) - len(raw_row))
        rec = dict(zip(headers, raw_row, strict=False))
        try:
            out.append(
                ParsedRow(
                    buchungsdatum=_parse_date(rec["buchungsdatum"]) or _raise("buchungsdatum required"),
                    wertstellung=_parse_date(rec.get("wertstellung", "")),
                    status=_clean_text(rec.get("status", "")),
                    zahlungspflichtiger=_clean_text(rec.get("zahlungspflichtiger", "")),
                    zahlungsempfaenger=_clean_text(rec.get("zahlungsempfaenger", "")),
                    verwendungszweck=_clean_text(rec.get("verwendungszweck", "")),
                    umsatztyp=_clean_text(rec.get("umsatztyp", "")),
                    # IBANs are intentionally rendered with grouping spaces by
                    # some banks ("DE89 3704 0044 ..."); strip every internal
                    # space so equality/lookup still works.
                    iban=(rec.get("iban") or "").strip().replace(" ", ""),
                    betrag=_parse_amount(rec["betrag"]),
                    glaeubiger_id=_clean_text(rec.get("glaeubiger_id", "")),
                    mandatsreferenz=_clean_text(rec.get("mandatsreferenz", "")),
                    kundenreferenz=_clean_text(rec.get("kundenreferenz", "")),
                    source_file=path.name,
                    source_row=offset,
                )
            )
        except CsvParseError as e:
            raise CsvParseError(f"{path.name} row {offset}: {e}") from e
    return out


def _raise(msg: str):  # tiny helper for inline raise in expression context
    raise CsvParseError(msg)
