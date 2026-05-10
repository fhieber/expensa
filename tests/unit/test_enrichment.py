"""Notes CRUD + vendor-lookup privacy invariants."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from expense_analyzer.config import VendorLookupConfig
from expense_analyzer.enrichment.notes import delete_note, get_note, set_note
from expense_analyzer.enrichment.vendor_web import (
    VendorLookupDisabled,
    _heuristic_industry,
    lookup_vendor,
)


def _insert_expense(conn: sqlite3.Connection, h: str = "h1") -> int:
    conn.execute(
        "INSERT INTO expenses(buchungsdatum, betrag_cents, dedup_hash, is_income, is_round) "
        "VALUES ('2026-01-01', -100, ?, 0, 0)",
        (h,),
    )
    return int(conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"])


def test_set_get_note(tmp_db: sqlite3.Connection) -> None:
    eid = _insert_expense(tmp_db)
    set_note(tmp_db, eid, "Steuerlich absetzbar")
    assert get_note(tmp_db, eid) == "Steuerlich absetzbar"


def test_set_note_replaces(tmp_db: sqlite3.Connection) -> None:
    eid = _insert_expense(tmp_db)
    set_note(tmp_db, eid, "first")
    set_note(tmp_db, eid, "second")
    assert get_note(tmp_db, eid) == "second"


def test_empty_note_deletes(tmp_db: sqlite3.Connection) -> None:
    eid = _insert_expense(tmp_db)
    set_note(tmp_db, eid, "first")
    set_note(tmp_db, eid, "   ")
    assert get_note(tmp_db, eid) is None


def test_delete_note(tmp_db: sqlite3.Connection) -> None:
    eid = _insert_expense(tmp_db)
    set_note(tmp_db, eid, "x")
    delete_note(tmp_db, eid)
    assert get_note(tmp_db, eid) is None


def test_vendor_lookup_disabled_by_default(tmp_db: sqlite3.Connection) -> None:
    cfg = VendorLookupConfig()
    assert cfg.enabled is False
    with pytest.raises(VendorLookupDisabled):
        lookup_vendor(tmp_db, "rewe markt", cfg)


def test_vendor_lookup_only_sends_counterparty(tmp_db: sqlite3.Connection) -> None:
    """**Privacy test**: the search backend must only ever receive the
    normalized counterparty name. Never amount, IBAN, or Verwendungszweck."""
    cfg = VendorLookupConfig(enabled=True, backend="duckduckgo")
    captured: list[str] = []

    def fake_ddg(query: str, max_results: int = 3) -> str:
        captured.append(query)
        return "REWE is a German supermarket chain"

    with patch("expense_analyzer.enrichment.vendor_web._ddg_search", side_effect=fake_ddg):
        info = lookup_vendor(tmp_db, "rewe markt", cfg)

    # The single outbound call must equal the counterparty - nothing else.
    assert captured == ["rewe markt"]
    assert info.counterparty_normalized == "rewe markt"
    assert info.industry == "supermarket"


def test_vendor_lookup_uses_cache_on_second_call(tmp_db: sqlite3.Connection) -> None:
    cfg = VendorLookupConfig(enabled=True, backend="duckduckgo")
    calls = []

    def fake(query: str, max_results: int = 3) -> str:
        calls.append(query)
        return "Edeka Supermarkt"

    with patch("expense_analyzer.enrichment.vendor_web._ddg_search", side_effect=fake):
        lookup_vendor(tmp_db, "edeka sued", cfg)
        lookup_vendor(tmp_db, "edeka sued", cfg)

    assert len(calls) == 1, "second lookup should hit the cache"


def test_heuristic_industry_classifies_known_categories() -> None:
    assert _heuristic_industry("rewe markt", "supermarkt edeka") == "supermarket"
    assert _heuristic_industry("netflix international", "streaming") == "streaming"
    assert _heuristic_industry("telekom deutschland", "mobilfunk") == "telco"
    assert _heuristic_industry("vermieter schmidt", "miete januar") == "rent"


def test_vendor_lookup_empty_counterparty(tmp_db: sqlite3.Connection) -> None:
    cfg = VendorLookupConfig(enabled=True)
    info = lookup_vendor(tmp_db, "", cfg)
    assert info.summary == ""
    assert info.industry == "other"
