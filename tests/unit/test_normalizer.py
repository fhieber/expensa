"""Normalizer tests."""

from __future__ import annotations

from expense_analyzer.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)


def test_normalize_counterparty_lowercases_and_folds() -> None:
    assert normalize_counterparty("Müller GmbH") == "mueller"
    assert normalize_counterparty("REWE Markt GmbH") == "rewe markt"


def test_normalize_counterparty_strips_legal_suffixes() -> None:
    for raw, expected in [
        ("Allianz Versicherungs-AG", "allianz versicherungs"),
        ("Telekom Deutschland GmbH", "telekom deutschland"),
        ("Beispiel KG", "beispiel"),
        ("Foo SE", "foo"),
    ]:
        assert normalize_counterparty(raw) == expected


def test_normalize_counterparty_idempotent() -> None:
    s = "REWE Markt GmbH"
    once = normalize_counterparty(s)
    twice = normalize_counterparty(once)
    assert once == twice


def test_normalize_counterparty_strips_long_digits() -> None:
    assert normalize_counterparty("Amazon EU Bestellung 12345678") == "amazon eu bestellung"


def test_normalize_verwendungszweck_strips_iban_and_long_digits() -> None:
    raw = "Ueberweisung an DE89370400440532013000 Refnr 998877665544"
    out = normalize_verwendungszweck(raw)
    assert "DE89370400440532013000" not in out
    assert "998877665544" not in out


def test_normalize_verwendungszweck_strips_sepa_label() -> None:
    out = normalize_verwendungszweck("SEPA-LASTSCHRIFT Spotify Premium Familie")
    assert "sepa" not in out.lower()
    assert "spotify" in out


def test_normalize_verwendungszweck_strips_urls() -> None:
    out = normalize_verwendungszweck("Bestellung https://amazon.de/order/123")
    assert "https" not in out


def test_combined_text_handles_empties() -> None:
    assert combined_text("", "") == ""
    assert combined_text("rewe", "") == "rewe"
    assert combined_text("", "einkauf") == "einkauf"
    assert combined_text("rewe", "einkauf") == "rewe | einkauf"
