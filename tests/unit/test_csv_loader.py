"""CSV-loader tests against the synthetic German fixture."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from expense_analyzer.ingestion.csv_loader import (
    CsvParseError,
    parse_csv,
    _parse_amount,
    _parse_date,
    _normalize_header,
)


def test_parse_sample_de_yields_expected_rows(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    assert len(rows) == 50


def test_first_row_typed_correctly(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    r0 = rows[0]
    assert r0.buchungsdatum == date(2026, 1, 1)
    assert r0.zahlungsempfaenger == "Vermieter Schmidt"
    assert r0.betrag == Decimal("-950.00")
    assert r0.iban == "DE89370400440532013000"
    assert r0.umsatztyp == "Dauerauftrag"


def test_betrag_cents_int(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    salary = next(r for r in rows if r.zahlungspflichtiger == "Arbeitgeber AG")
    assert salary.betrag_cents == 320050


def test_iban_whitespace_stripped(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    assert all(" " not in r.iban for r in rows)


def test_normalize_header_handles_umlauts_and_asterisks() -> None:
    assert _normalize_header("Zahlungspflichtige*r") == "zahlungspflichtiger"
    assert _normalize_header("Zahlungsempfänger*in") == "zahlungsempfaenger"
    assert _normalize_header("Betrag (€)") == "betrag"
    assert _normalize_header("Gläubiger-ID") == "glaeubiger_id"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.234,56", Decimal("1234.56")),
        ("-12,34", Decimal("-12.34")),
        ("0,01", Decimal("0.01")),
        ("3200,50", Decimal("3200.50")),
        ("-1.000.000,00", Decimal("-1000000.00")),
    ],
)
def test_parse_amount(raw: str, expected: Decimal) -> None:
    assert _parse_amount(raw) == expected


def test_parse_amount_rejects_garbage() -> None:
    with pytest.raises(CsvParseError):
        _parse_amount("nonsense")


@pytest.mark.parametrize(
    "raw,expected",
    [("01.01.2026", date(2026, 1, 1)), ("31.12.2025", date(2025, 12, 31)), ("", None)],
)
def test_parse_date(raw: str, expected) -> None:
    assert _parse_date(raw) == expected


def test_parse_date_rejects_garbage() -> None:
    with pytest.raises(CsvParseError):
        _parse_date("not-a-date")


def test_missing_header_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("foo;bar\n1;2\n", encoding="utf-8")
    with pytest.raises(CsvParseError):
        parse_csv(bad)


def test_cp1252_encoding_handled(tmp_path: Path) -> None:
    """A file written in cp1252 with umlauts should still parse correctly."""
    content = (
        '"Buchungsdatum";"Wertstellung";"Status";"Zahlungspflichtige*r";'
        '"Zahlungsempfänger*in";"Verwendungszweck";"Umsatztyp";"IBAN";'
        '"Betrag (€)";"Gläubiger-ID";"Mandatsreferenz";"Kundenreferenz"\n'
        '"01.01.2026";"01.01.2026";"Gebucht";"";"Müller GmbH";"Tankstelle Öl";'
        '"Kartenzahlung";"DE21701500000000123456";"-12,34";"";"";""\n'
    )
    p = tmp_path / "cp1252.csv"
    p.write_bytes(content.encode("cp1252"))
    rows = parse_csv(p)
    assert len(rows) == 1
    assert rows[0].zahlungsempfaenger == "Müller GmbH"
