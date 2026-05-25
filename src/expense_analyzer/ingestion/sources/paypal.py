"""PayPal "Aktivitäten" CSV adapter.

The German PayPal export ("Aktivitäten herunterladen → CSV (kompletter
Bericht)") emits the following columns, in order:

    "Datum","Uhrzeit","Zeitzone","Beschreibung","Währung","Brutto",
    "Entgelt","Netto","Guthaben","Transaktionscode",
    "Absender E-Mail-Adresse","Name","Name der Bank","Bankkonto",
    "Versand- und Bearbeitungsgebühr","Umsatzsteuer","Rechnungsnummer",
    "Zugehöriger Transaktionscode"

We only consume a handful of these:

  * ``Datum`` → transaction date
  * ``Brutto`` (with ``Netto`` as fallback) → signed amount
  * ``Name`` → counterparty (the merchant)
  * ``Beschreibung`` → free-text description (PayPal's own classification
    of the transaction, e.g. "Allgemeine Zahlung", "Express-Zahlung",
    "Rückzahlung", or a merchant-supplied memo)
  * ``Transaktionscode`` → unique source reference, used to dedupe and
    to link enrichments back to the original PayPal row

There is no ``Status`` column in this export, so every row in the file
is treated as a settled transaction. Rows without a ``Transaktionscode``
are skipped (they can't be deduplicated against a future re-export).

All format-specific knowledge lives in this module — the generic schema,
engine, CLI and UI never depend on it, so future PayPal export changes
are a localized fix.
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

# Header keys we read (folded form -- see ``_fold``). Everything else in
# the export is currently unused; if we ever want to surface the bank
# account, invoice number etc. they're trivial to add without changing
# the public contract.
_KEY_DATE = "datum"
_KEY_NAME = "name"
_KEY_GROSS = "brutto"
_KEY_NET = "netto"
_KEY_REF = "transaktionscode"
_KEY_DESCRIPTION = "beschreibung"
# Used to resolve "wrapper" rows (the +X bank-pull credit that PayPal
# emits alongside every -X bank-funded purchase) back to the actual
# purchase row that carries the merchant name.
_KEY_LINKED_REF = "zugehoeriger_transaktionscode"

# Substrings (in the folded Beschreibung) that identify a row as a
# "wrapper" -- i.e. the bank-account pull that mirrors a separate
# purchase row, not a real merchant transaction. Without resolving
# these we end up writing "Bankgutschrift auf PayPal-Konto" as the
# enriched description on every PayPal Lastschrift, which is useless
# boilerplate. Substring matching (vs exact) tolerates the export's
# minor wording shifts between PayPal locales / versions.
_WRAPPER_BESCHREIBUNG_HINTS: tuple[str, ...] = (
    "bankgutschrift",       # "Bankgutschrift auf PayPal-Konto" (most common)
    "aufladung",            # "Allgemeine Aufladung mit Banküberweisung"
    "abbuchung_vom_bankkonto",
)


def _is_wrapper_beschreibung(beschreibung: str) -> bool:
    folded = _fold(beschreibung)
    return any(hint in folded for hint in _WRAPPER_BESCHREIBUNG_HINTS)


def _fold(s: str) -> str:
    """Normalize a header to a stable lookup key.

    Lowercases, replaces German umlauts with their ASCII expansions,
    and turns whitespace into underscores so multi-word headers like
    ``"Absender E-Mail-Adresse"`` are still addressable as a dict key.
    """
    return (
        s.strip().lower()
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        .replace(" ", "_")
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    """Decode the file, auto-detect the delimiter, and return a list of
    ``{folded_header: value}`` dicts.

    Raises ``CsvParseError`` if no header with a ``Transaktionscode``
    column is found under any of comma/semicolon/tab. The auto-delimiter
    sweep makes us tolerant of region-localized exports (PayPal switches
    between ``,`` and ``;`` depending on the user's locale).
    """
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
        """Cheap detection: read the first non-empty line and check for
        the three columns we actually use (Transaktionscode + Name +
        either Brutto or Netto). We also accept the older export shape
        that lacked ``Beschreibung`` -- the format change was only
        about adding/removing peripheral columns, the four core fields
        have always been there.
        """
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
        """Parse the file into ``EnrichmentRecord``s.

        Two-pass: first build a Transaktionscode → row map so wrapper
        rows can resolve their linked purchase, then emit records.

        Per-row outcomes:

        * **Wrapper with a valid link** (Beschreibung looks like a
          bank-pull AND ``Zugehöriger Transaktionscode`` points to
          another row in the file) → emit one record with the
          wrapper's date/amount/source_ref (those are what the bank
          sees) and the linked purchase row's Name + Beschreibung
          (the actual merchant + memo). This is the case that fixes
          the "every PayPal Lastschrift gets ``Bankgutschrift auf
          PayPal-Konto`` as the description" pollution.
        * **Wrapper without a link** → dropped. The merchant info
          isn't in this file, so the only thing we'd write is the
          boilerplate description -- which is exactly the noise we're
          trying to remove.
        * **Anything else** → emit as-is. Covers regular merchant
          purchases (refunds, balance-funded payments, etc.) which
          may or may not match a bank row depending on funding source.

        Rows without a Transaktionscode are dropped (can't be deduped
        or linked back). Rows without a parseable date or amount are
        also dropped silently.
        """
        raw_rows = _read_table(path)
        # First pass: index by Transaktionscode so wrapper rows can
        # resolve their linked purchase. Skip rows with no ref -- they
        # can't be the target of a link anyway.
        by_ref: dict[str, dict[str, str]] = {}
        for rec in raw_rows:
            ref = (rec.get(_KEY_REF) or "").strip()
            if ref:
                by_ref[ref] = rec

        records: list[EnrichmentRecord] = []
        for rec in raw_rows:
            ref = (rec.get(_KEY_REF) or "").strip()
            if not ref:
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

            description = (rec.get(_KEY_DESCRIPTION) or "").strip()
            name = (rec.get(_KEY_NAME) or "").strip()
            linked_ref = (rec.get(_KEY_LINKED_REF) or "").strip()

            if _is_wrapper_beschreibung(description):
                # Resolve via the linked purchase if available; drop
                # otherwise (orphan wrappers are just bank-pull noise).
                linked = by_ref.get(linked_ref) if linked_ref else None
                if linked is None:
                    continue
                name = (linked.get(_KEY_NAME) or "").strip() or name
                description = (linked.get(_KEY_DESCRIPTION) or "").strip() or description

            records.append(
                EnrichmentRecord(
                    txn_date=txn_date,
                    amount=amount,
                    counterparty=name,
                    description=description,
                    source_ref=ref,
                )
            )
        return records

    def candidate_filter(self, expense_row: Mapping[str, object]) -> bool:
        """Restrict matching to bank rows that look like a PayPal debit.

        The PayPal counterparty on a German bank statement is almost
        always ``"PayPal (Europe) S.a.r.l. et Cie., S.C.A."`` or a
        truncated variant. We check both the raw payer and payee
        columns because normalization strips the word ``PayPal``
        from ``counterparty_normalized``, so reading the normalized
        column would always miss.
        """
        for col in ("zahlungsempfaenger", "zahlungspflichtiger"):
            if "paypal" in str(expense_row.get(col) or "").lower():
                return True
        return False
