"""IBAN classifier tests."""

from __future__ import annotations

from expense_analyzer.features.iban import classify_iban


def test_valid_german_iban_decomposed() -> None:
    info = classify_iban("DE89370400440532013000")
    assert info.country == "DE"
    assert info.is_valid is True
    assert info.is_foreign is False
    assert info.blz == "37040044"


def test_foreign_iban_flagged() -> None:
    info = classify_iban("LU280019400644750000")
    assert info.country == "LU"
    assert info.is_foreign is True
    assert info.blz is None


def test_empty_iban_safe() -> None:
    info = classify_iban("")
    assert info.country is None
    assert info.is_valid is False
    assert info.is_foreign is False


def test_known_self_iban_flagged() -> None:
    own = {"DE89370400440532013000"}
    info = classify_iban("DE89370400440532013000", own_ibans=own)
    assert info.is_known_self is True


def test_self_iban_does_not_match_other() -> None:
    own = {"DE12500105170648489890"}
    info = classify_iban("DE89370400440532013000", own_ibans=own)
    assert info.is_known_self is False


def test_iban_whitespace_tolerated() -> None:
    info = classify_iban("DE89 3704 0044 0532 0130 00")
    assert info.country == "DE"
    assert info.blz == "37040044"
