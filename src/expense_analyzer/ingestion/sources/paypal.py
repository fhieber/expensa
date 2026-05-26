"""PayPal "Aktivitäten" CSV adapter (simplified).

Reads a German PayPal export and produces one EnrichmentRecord per
settled payment row, using only:

  Datum, Netto, Name, Absender E-Mail-Adresse, Transaktionscode

Rows without a Transaktionscode or Name, or with an unparseable date
or Netto amount, are skipped.

The ``description`` field on each EnrichmentRecord carries the
pre-formatted Verwendungszweck string that the enrichment engine writes
directly into the database (e.g. "Etsy Inc" or "John Doe (john@example.com)").
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

_KEY_DATE = "datum"
_KEY_NET = "netto"
_KEY_REF = "transaktionscode"
_KEY_NAME = "name"
_KEY_EMAIL = "absender_e-mail-adresse"


def _fold(s: str) -> str:
    """Lowercase + umlaut-fold + collapse spaces to underscores.
    Hyphens are kept so "e-mail" stays "e-mail" in the key."""
    return (
        s.strip().lower()
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        .replace(" ", "_")
    )


def _read_table(path: Path) -> list[dict[str, str]]:
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
        f"{path.name}: no PayPal header (column {_KEY_REF!r}) found"
    )


def make_paypal_vz(name: str, email: str) -> str:
    """Format the Verwendungszweck for an enriched PayPal record.

    Returns ``"{name} ({email})"`` when email is non-empty, otherwise
    just ``"{name}"``. The counterparty and enrichment_source columns
    already identify the row as PayPal; no prefix needed here.
    """
    name = name.strip()
    email = email.strip()
    return f"{name} ({email})" if email else name


class PaypalAdapter:
    name = "paypal"

    def sniff(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding=detect_encoding(path))
        except (OSError, CsvParseError):
            return False
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        folded = _fold(first)
        return _KEY_REF in folded and _KEY_NAME in folded and _KEY_NET in folded

    def parse(self, path: Path) -> list[EnrichmentRecord]:
        """Parse the file into EnrichmentRecords.

        Each record carries the merchant name in ``counterparty`` and the
        pre-formatted new Verwendungszweck in ``description``.  The engine
        writes ``description`` directly to the bank expense's
        ``verwendungszweck`` column on a match.
        """
        records: list[EnrichmentRecord] = []
        for rec in _read_table(path):
            ref = (rec.get(_KEY_REF) or "").strip()
            if not ref:
                continue
            name = (rec.get(_KEY_NAME) or "").strip()
            if not name:
                continue
            raw_netto = (rec.get(_KEY_NET) or "").strip()
            raw_date = (rec.get(_KEY_DATE) or "").strip()
            if not raw_netto or not raw_date:
                continue
            try:
                amount = parse_german_amount(raw_netto)
                txn_date = parse_german_date(raw_date)
            except CsvParseError:
                continue
            if txn_date is None:
                continue
            email = (rec.get(_KEY_EMAIL) or "").strip()
            records.append(
                EnrichmentRecord(
                    txn_date=txn_date,
                    amount=amount,
                    counterparty=name,
                    description=make_paypal_vz(name, email),
                    source_ref=ref,
                )
            )
        return records

    def candidate_filter(self, expense_row: Mapping[str, object]) -> bool:
        for col in ("zahlungsempfaenger", "zahlungspflichtiger"):
            if "paypal" in str(expense_row.get(col) or "").lower():
                return True
        return False
