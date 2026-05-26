"""Tests for the simplified PayPal secondary-source adapter."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from expense_analyzer.ingestion.sources import detect_adapter
from expense_analyzer.ingestion.sources.paypal import PaypalAdapter, make_paypal_vz


def test_sniff_accepts_paypal_rejects_bank(fixtures_dir: Path) -> None:
    adapter = PaypalAdapter()
    assert adapter.sniff(fixtures_dir / "sample_paypal.csv") is True
    assert adapter.sniff(fixtures_dir / "sample_de.csv") is False


def test_detect_adapter_picks_paypal(fixtures_dir: Path) -> None:
    assert detect_adapter(fixtures_dir / "sample_paypal.csv").name == "paypal"


def test_parse_skips_rows_without_transaktionscode(fixtures_dir: Path) -> None:
    """Drops rows with an empty Transaktionscode (can't be deduped or linked)."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    refs = {r.source_ref for r in records}
    assert "" not in refs
    # The no-ref row (NoRef Shop) must be dropped.
    assert all(r.source_ref for r in records)


def test_parse_skips_rows_without_name(fixtures_dir: Path) -> None:
    """Rows without a Name have nothing useful to put in the VZ — skip them."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    assert all(r.counterparty for r in records)


def test_parse_correct_record_count(fixtures_dir: Path) -> None:
    """6 data rows; 1 has an empty Transaktionscode → 5 parsed."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    assert len(records) == 5


def test_parse_maps_date_netto_name(fixtures_dir: Path) -> None:
    """Datum → txn_date, Netto → amount, Name → counterparty."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    alpha = next(r for r in records if r.source_ref == "TXN-ALPHA-1")
    assert alpha.counterparty == "Haendler Alpha GmbH"
    assert alpha.txn_date == date(2026, 3, 3)
    assert alpha.amount == Decimal("-19.80")
    assert alpha.amount_cents == -1980


def test_parse_formats_description_without_email(fixtures_dir: Path) -> None:
    """When Absender E-Mail-Adresse is empty, description is 'PayPal . <name>'."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    beta = next(r for r in records if r.source_ref == "TXN-BETA-1")
    assert beta.description == "PayPal . Haendler Beta GmbH"


def test_parse_formats_description_with_email(fixtures_dir: Path) -> None:
    """When email is present, description is 'PayPal . <name> (<email>)'."""
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    gamma = next(r for r in records if r.source_ref == "TXN-GAMMA")
    assert gamma.description == "PayPal . Haendler Gamma GmbH (vendor-a@example.com)"


def test_make_paypal_vz_with_email() -> None:
    assert make_paypal_vz("Haendler Alpha GmbH", "buyer@example.com") == "PayPal . Haendler Alpha GmbH (buyer@example.com)"


def test_make_paypal_vz_without_email() -> None:
    assert make_paypal_vz("Haendler Beta GmbH", "") == "PayPal . Haendler Beta GmbH"


def test_make_paypal_vz_strips_whitespace() -> None:
    assert make_paypal_vz("  Shop  ", "  test@x.com  ") == "PayPal . Shop (test@x.com)"


def test_candidate_filter_matches_paypal_bank_rows() -> None:
    adapter = PaypalAdapter()
    assert adapter.candidate_filter({"zahlungsempfaenger": "PayPal Europe"}) is True
    assert adapter.candidate_filter({"zahlungspflichtiger": "paypal s.a.r.l."}) is True
    assert adapter.candidate_filter({"zahlungsempfaenger": "Markt Alpha GmbH"}) is False
    assert adapter.candidate_filter({}) is False
