"""PayPal "Aktivitäten" CSV adapter.

.. warning::

   **The exact PayPal export layout is provisional.** Column names,
   delimiter and date/amount formatting are a best guess at the German
   "Aktivitäten herunterladen" export and should be corrected against a
   real sample. Everything PayPal-specific is contained in this module —
   the generic schema, engine, CLI and UI do not depend on any of it, so
   adjusting the format here is a localized change.

Typical German header (comma-separated, quoted), of which we use a handful:

    "Datum","Uhrzeit","Zeitzone","Name","Typ","Status","Währung","Brutto",
    "Gebühr","Netto",...,"Transaktionscode",...,"Artikelbezeichnung",...,"Hinweis"
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from expense_analyzer.ingestion._parsing import (
    CsvParseError,
    detect_encoding,
    parse_german_amount,
    parse_german_date,
)
from expense_analyzer.ingestion.sources import EnrichmentRecord

# Header keys (after folding) we rely on. `Brutto` is the buyer-paid gross,
# which is what hits the bank as a Lastschrift; we fall back to `Netto`.
_KEY_DATE = "datum"
_KEY_NAME = "name"
_KEY_STATUS = "status"
_KEY_GROSS = "brutto"
_KEY_NET = "netto"
_KEY_REF = "transaktionscode"
_KEY_ITEM = "artikelbezeichnung"
_KEY_NOTE = "hinweis"
_KEY_TYPE = "typ"

# Status values that mean the transaction did not actually settle, so it
# can't correspond to a bank debit.
_REJECT_STATUS = {
    "storniert", "fehlgeschlagen", "abgelehnt", "ausstehend",
    "cancelled", "failed", "denied", "pending",
}


def _fold(s: str) -> str:
    return (
        s.strip().lower()
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        .replace(" ", "_")
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    """Decode the file, auto-detect the delimiter, and return a list of
    {folded_header: value} dicts. Raises CsvParseError if no header with a
    Transaktionscode column is found under any delimiter."""
    text = path.read_text(encoding=detect_encoding(path))
    lines = text.splitlines()
    if not lines:
        return []
    for delim in (",", ";", "\t"):
        rows = list(csv.reader(lines, delimiter=delim, quotechar='"'))
        if not rows:
            continue
        headers = [_fold(h) for h in rows[0]]
        if _KEY_REF in headers:
            out: list[dict[str, str]] = []
            for raw in rows[1:]:
                if not any(cell.strip() for cell in raw):
                    continue
                if len(raw) < len(headers):
                    raw = raw + [""] * (len(headers) - len(raw))
                out.append(dict(zip(headers, raw, strict=False)))
            return out
    raise CsvParseError(
        f"{path.name}: no PayPal header (column {_KEY_REF!r}) found under , ; or tab"
    )


class PaypalAdapter:
    name = "paypal"

    def sniff(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding=detect_encoding(path))
        except (OSError, CsvParseError):
            return False
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        folded = _fold(first)
        return _KEY_REF in folded and _KEY_NAME in folded and (
            _KEY_GROSS in folded or _KEY_NET in folded
        )

    def parse(self, path: Path) -> list[EnrichmentRecord]:
        records: list[EnrichmentRecord] = []
        for rec in _read_table(path):
            ref = (rec.get(_KEY_REF) or "").strip()
            if not ref or _fold(rec.get(_KEY_STATUS, "")) in _REJECT_STATUS:
                continue
            raw_amount = (rec.get(_KEY_GROSS) or rec.get(_KEY_NET) or "").strip()
            raw_date = (rec.get(_KEY_DATE) or "").strip()
            if not raw_amount or not raw_date:
                continue
            try:
                amount = parse_german_amount(raw_amount)
                txn_date = parse_german_date(raw_date)
            except CsvParseError:
                continue
            if txn_date is None:
                continue
            description = (
                (rec.get(_KEY_ITEM) or "").strip()
                or (rec.get(_KEY_NOTE) or "").strip()
                or (rec.get(_KEY_TYPE) or "").strip()
            )
            records.append(
                EnrichmentRecord(
                    txn_date=txn_date,
                    amount=amount,
                    counterparty=(rec.get(_KEY_NAME) or "").strip(),
                    description=description,
                    source_ref=ref,
                )
            )
        return records

    def candidate_filter(self, expense_row: Mapping[str, object]) -> bool:
        # PayPal debits show up with the raw counterparty containing "paypal"
        # (normalization strips the word, so we must read the raw columns).
        for col in ("zahlungsempfaenger", "zahlungspflichtiger"):
            if "paypal" in str(expense_row.get(col) or "").lower():
                return True
        return False
