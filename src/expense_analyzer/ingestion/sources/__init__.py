"""Secondary-source enrichment adapters.

A *secondary source* is a second CSV the user exports from somewhere else
(PayPal, a card issuer, a marketplace) that describes transactions already
present in the bank export, but with more / better information — the real
merchant, the purchased item, a memo.

Each adapter turns one such CSV format into a list of source-agnostic
:class:`EnrichmentRecord` objects and tells the generic engine
(:mod:`expense_analyzer.enrichment.secondary`) which bank expenses are even
candidates for matching. All format-specific knowledge lives in the adapter;
the engine, the DB schema and the UI never know which source produced a record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from expense_analyzer.ingestion._parsing import CsvParseError


@dataclass(frozen=True)
class EnrichmentRecord:
    """One transaction as seen by a secondary source.

    ``amount`` is signed in the source's own convention; matching only ever
    compares absolute cents, so the sign doesn't need to agree with the bank.
    """

    txn_date: date
    amount: Decimal
    counterparty: str
    description: str
    source_ref: str  # stable id in the source (e.g. PayPal Transaktionscode)

    @property
    def amount_cents(self) -> int:
        return int((self.amount * 100).to_integral_value())


class SourceAdapter(Protocol):
    """Pluggable parser for one secondary-CSV format."""

    name: str

    def sniff(self, path: Path) -> bool:
        """Cheaply decide whether ``path`` is this adapter's format
        (typically by inspecting the header row)."""
        ...

    def parse(self, path: Path) -> list[EnrichmentRecord]:
        """Parse the whole file into enrichment records."""
        ...

    def candidate_filter(self, expense_row: Mapping[str, object]) -> bool:
        """Return True if ``expense_row`` (a mapping of bank-expense fields,
        e.g. a ``sqlite3.Row``) could plausibly be enriched by this source.
        Lets e.g. the PayPal adapter restrict matching to PayPal direct-debit
        lines and avoid spurious amount+date collisions with unrelated
        transactions."""
        ...


def _adapters() -> list[SourceAdapter]:
    # Imported lazily so the registry has no import cost until used and new
    # adapters don't create import cycles.
    from expense_analyzer.ingestion.sources.paypal import PaypalAdapter

    return [PaypalAdapter()]


def get_adapter(name: str) -> SourceAdapter:
    """Look up an adapter by its ``name`` (case-insensitive)."""
    for a in _adapters():
        if a.name.lower() == name.lower():
            return a
    known = ", ".join(a.name for a in _adapters())
    raise CsvParseError(f"unknown enrichment source {name!r} (known: {known})")


def detect_adapter(path: Path) -> SourceAdapter:
    """Auto-detect which adapter handles ``path`` via each adapter's
    ``sniff``. Raises if none (or more than one) match."""
    matches = [a for a in _adapters() if a.sniff(path)]
    if not matches:
        raise CsvParseError(
            f"could not detect a secondary-source format for {path.name}; "
            "pass an explicit --source"
        )
    if len(matches) > 1:
        names = ", ".join(a.name for a in matches)
        raise CsvParseError(
            f"{path.name} matched multiple sources ({names}); pass --source to disambiguate"
        )
    return matches[0]
