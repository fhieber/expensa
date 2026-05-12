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
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

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


# Encodings to try, in order. The first that decodes the whole file wins.
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


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


class CsvParseError(ValueError):
    """Unrecoverable problem parsing a CSV row."""


def _detect_encoding(path: Path) -> str:
    """Try encodings until one decodes the entire file."""
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise CsvParseError(f"Could not decode {path} with any of {_ENCODINGS}")


def _normalize_header(h: str) -> str:
    """Map the German header (with umlauts and asterisks) to a Python-friendly key."""
    s = h.strip().lower()
    s = s.replace("*r", "r").replace("*in", "")  # Zahlungspflichtige*r -> zahlungspflichtiger
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = s.replace(" (€)", "").replace("(€)", "").strip()
    s = s.replace("-", "_").replace(" ", "_")
    return s


def _parse_amount(raw: str) -> Decimal:
    """German format: '1.234,56' or '-12,34' -> Decimal."""
    if raw is None or raw == "":
        raise CsvParseError("empty amount")
    s = raw.strip().replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise CsvParseError(f"bad amount {raw!r}") from e


def _parse_date(raw: str) -> date | None:
    """Parse a date cell. Supports the formats actually emitted by German
    bank exports:

      * ``DD.MM.YYYY``  -- canonical 4-digit form (e.g. ``08.05.2026``)
      * ``DD.MM.YY``    -- 2-digit form (e.g. ``08.05.26``); see below
      * ``YYYY-MM-DD``  -- ISO, occasionally emitted
      * ``DD/MM/YYYY``  -- slash variant

    The 2-digit ``%y`` directive follows POSIX: 00-68 → 2000-2068,
    69-99 → 1969-1999. That covers any realistic bank transaction we'll
    see.
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
                    status=(rec.get("status") or "").strip(),
                    zahlungspflichtiger=(rec.get("zahlungspflichtiger") or "").strip(),
                    zahlungsempfaenger=(rec.get("zahlungsempfaenger") or "").strip(),
                    verwendungszweck=(rec.get("verwendungszweck") or "").strip(),
                    umsatztyp=(rec.get("umsatztyp") or "").strip(),
                    iban=(rec.get("iban") or "").strip().replace(" ", ""),
                    betrag=_parse_amount(rec["betrag"]),
                    glaeubiger_id=(rec.get("glaeubiger_id") or "").strip(),
                    mandatsreferenz=(rec.get("mandatsreferenz") or "").strip(),
                    kundenreferenz=(rec.get("kundenreferenz") or "").strip(),
                    source_file=path.name,
                    source_row=offset,
                )
            )
        except CsvParseError as e:
            raise CsvParseError(f"{path.name} row {offset}: {e}") from e
    return out


def _raise(msg: str):  # tiny helper for inline raise in expression context
    raise CsvParseError(msg)
