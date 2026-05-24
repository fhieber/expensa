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


def test_parse_skips_cancelled_and_unreferenced(fixtures_dir: Path) -> None:
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    # 8 data rows; one cancelled + one without a Transaktionscode are dropped.
    assert len(records) == 6
    refs = {r.source_ref for r in records}
    assert "TXN-CANCEL" not in refs
    assert "" not in refs


def test_parse_maps_fields(fixtures_dir: Path) -> None:
    records = PaypalAdapter().parse(fixtures_dir / "sample_paypal.csv")
    etsy = next(r for r in records if r.source_ref == "TXN-ETSY-1")
    assert etsy.counterparty == "Etsy Inc"
    assert etsy.description == "Handmade Keramik Tasse"
    assert etsy.txn_date == date(2026, 3, 3)
    assert etsy.amount == Decimal("-19.80")
    assert etsy.amount_cents == -1980
