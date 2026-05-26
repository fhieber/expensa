"""Normalizer tests."""

from __future__ import annotations

from expense_analyzer.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)


def test_normalize_counterparty_lowercases_and_folds() -> None:
    assert normalize_counterparty("Müller GmbH") == "mueller"
    assert normalize_counterparty("Markt Alpha GmbH") == "markt alpha"


def test_normalize_counterparty_strips_legal_suffixes() -> None:
    for raw, expected in [
        ("Allianz Versicherungs-AG", "allianz versicherungs"),
        ("Telekom Deutschland GmbH", "telekom deutschland"),
        ("Beispiel KG", "beispiel"),
        ("Foo SE", "foo"),
    ]:
        assert normalize_counterparty(raw) == expected


def test_normalize_counterparty_idempotent() -> None:
    s = "Markt Alpha GmbH"
    once = normalize_counterparty(s)
    twice = normalize_counterparty(once)
    assert once == twice


def test_normalize_counterparty_strips_long_digits() -> None:
    assert normalize_counterparty("Onlinehandel EU Bestellung 12345678") == "onlinehandel eu bestellung"


def test_normalize_verwendungszweck_strips_iban_and_long_digits() -> None:
    raw = "Ueberweisung an DE89370400440532013000 Refnr 998877665544"
    out = normalize_verwendungszweck(raw)
    assert "DE89370400440532013000" not in out
    assert "998877665544" not in out


def test_normalize_verwendungszweck_strips_sepa_label() -> None:
    out = normalize_verwendungszweck("SEPA-LASTSCHRIFT Abo-Dienst Abo")
    assert "sepa" not in out.lower()
    assert "abo" in out


def test_normalize_verwendungszweck_strips_urls() -> None:
    out = normalize_verwendungszweck("Bestellung https://example.com/order/123")
    assert "https" not in out


def test_combined_text_handles_empties() -> None:
    assert combined_text("", "") == ""
    assert combined_text("vendor", "") == "vendor"
    assert combined_text("", "kauf") == "kauf"
    assert combined_text("vendor", "kauf") == "vendor | kauf"
