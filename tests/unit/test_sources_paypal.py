"""Tests for the PayPal secondary-source adapter."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from expense_analyzer.ingestion.sources import detect_adapter
from expense_analyzer.ingestion.sources.paypal import PaypalAdapter


def test_sniff_accepts_paypal_rejects_bank(fixtures_dir: Path) -> None:
    adapter = PaypalAdapter()
    assert adapter.sniff(fixtures_dir / "sample_paypal.csv") is True
    assert adapter.sniff(fixtures_dir / "sample_de.csv") is False


def test_detect_adapter_picks_paypal(fixtures_dir: Path) -> None:
    assert detect_adapter(fixtures_dir / "sample_paypal.csv").name == "paypal"


def test_parse_skips_rows_without_transaktionscode(fixtures_dir: Path) -> None:
    """Drops: rows without a Transaktionscode (can't be deduped or
    linked back) and orphan wrapper rows (Bankgutschrift without a
    linked purchase -- just bank-pull noise with no merchant info).

    Keeps: regular purchases, refunds, AND wrapper rows that DO link
    to a purchase row in the same file (the wrapper is resolved to
    carry the linked merchant's name/description)."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    # 11 data rows in the fixture; one empty-Transaktionscode row and
    # one orphan-wrapper row are dropped → 9 parsed.
    assert len(records) == 9
    refs = {r.source_ref for r in records}
    assert "" not in refs
    assert "TXN-REFUND" in refs
    # Wrapper rows with a linked purchase survive (resolved to the
    # merchant); orphan wrappers are dropped because they'd only add
    # boilerplate to the enrichment.
    assert "TXN-BETTER-WRAPPER" in refs
    assert "TXN-ORPHAN-WRAPPER" not in refs


def test_parse_maps_fields_from_real_columns(fixtures_dir: Path) -> None:
    """Verify the four columns we actually consume map correctly:
    Datum → date, Brutto → amount, Name → counterparty, Beschreibung
    → description."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    etsy = next(r for r in records if r.source_ref == "TXN-ETSY-1")
    assert etsy.counterparty == "Etsy Inc"
    # Beschreibung is PayPal's free-text description -- either a
    # merchant-supplied memo (as here) or, for many txns, the
    # generic PayPal classification like "Allgemeine Zahlung".
    assert etsy.description == "Bestellung Handmade Keramik Tasse"
    assert etsy.txn_date == date(2026, 3, 3)
    assert etsy.amount == Decimal("-19.80")
    assert etsy.amount_cents == -1980


def test_parse_handles_generic_paypal_txn_type_descriptions(
    fixtures_dir: Path,
) -> None:
    """For txns without a merchant memo PayPal puts its own
    classification in Beschreibung (e.g. "Express-Zahlung"). That's
    less informative than a memo but still useful as a tiebreaker
    for the embedding model."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    steam = next(r for r in records if r.source_ref == "TXN-STEAM-1")
    assert steam.description == "Express-Zahlung"


def test_parse_uses_brutto_with_netto_as_fallback(fixtures_dir: Path) -> None:
    """The Steam row has both Brutto and Netto populated; we should
    prefer Brutto. Verified by the value (-45.00 lives in both
    columns in this fixture but the contract is "Brutto first")."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    steam = next(r for r in records if r.source_ref == "TXN-STEAM-1")
    assert steam.amount == Decimal("-45.00")


def test_parse_handles_positive_refund_amounts(fixtures_dir: Path) -> None:
    """A Rückzahlung carries a positive Brutto; the adapter must not
    coerce signs (the matching engine compares absolute cents)."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    refund = next(r for r in records if r.source_ref == "TXN-REFUND")
    assert refund.amount == Decimal("15.00")
    assert refund.description == "Rückzahlung"


def test_wrapper_row_resolves_to_linked_purchase_merchant(
    fixtures_dir: Path,
) -> None:
    """The headline fix: a "Bankgutschrift auf PayPal-Konto" row used
    to enrich every PayPal bank Lastschrift with the boilerplate
    description, hiding the actual merchant. With the
    Zugehöriger-Transaktionscode resolution, the wrapper now carries
    the linked purchase's Name + Beschreibung -- so the enrichment
    surfaces the real merchant (betterplace.org) instead of
    "Bankgutschrift auf PayPal-Konto"."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    wrapper = next(r for r in records if r.source_ref == "TXN-BETTER-WRAPPER")
    # source_ref + amount + date are still the wrapper's own (those
    # are what the bank sees and dedupes against).
    assert wrapper.amount == Decimal("14.00")
    assert wrapper.txn_date == date(2026, 4, 4)
    # name + description come from the linked purchase row.
    assert wrapper.counterparty == "betterplace.org gGmbH"
    assert wrapper.description == "Bestellung betterplace Spende"
    assert "Bankgutschrift" not in wrapper.description


def test_orphan_wrapper_row_is_dropped(fixtures_dir: Path) -> None:
    """A "Bankgutschrift" row without a Zugehöriger Transaktionscode
    has no merchant info we can recover. Better to skip it entirely
    than to enrich a bank row with boilerplate."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    assert all(r.source_ref != "TXN-ORPHAN-WRAPPER" for r in records)
    # Defensive: no surviving record carries the boilerplate description.
    assert all("Bankgutschrift" not in r.description for r in records)


def test_candidate_filter_matches_paypal_bank_rows() -> None:
    """The bank-side counterparty contains 'PayPal' (raw, before
    normalisation strips the word). Confirm both column variants."""
    adapter = PaypalAdapter()
    assert adapter.candidate_filter({"zahlungsempfaenger": "PayPal Europe"}) is True
    assert adapter.candidate_filter({"zahlungspflichtiger": "paypal s.a.r.l."}) is True
    assert adapter.candidate_filter({"zahlungsempfaenger": "REWE Markt GmbH"}) is False
    # Defensive: missing keys must not raise.
    assert adapter.candidate_filter({}) is False
