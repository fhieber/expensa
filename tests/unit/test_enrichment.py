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
    assert info.industry == "Supermarkt"


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
    """Industry labels are German (multilingual NLI + DE embedding both
    benefit from the same-language signal)."""
    assert _heuristic_industry("rewe markt", "supermarkt edeka") == "Supermarkt"
    assert _heuristic_industry("netflix international", "streaming") == "Streaming"
    assert _heuristic_industry("telekom deutschland", "mobilfunk") == "Telekommunikation"
    assert _heuristic_industry("vermieter schmidt", "miete januar") == "Miete"
    # Fallback is the German "Sonstige" sentinel, not the English "other".
    assert _heuristic_industry("xyzzy", "nothing here") == "Sonstige"


def test_vendor_lookup_empty_counterparty(tmp_db: sqlite3.Connection) -> None:
    cfg = VendorLookupConfig(enabled=True)
    info = lookup_vendor(tmp_db, "", cfg)
    assert info.summary == ""
    assert info.industry == "Sonstige"


def test_normalize_industry_translates_legacy_english() -> None:
    """Legacy cached "supermarket"/"telco"/etc. rows from before the
    German-label switch must surface as German on read so the UI and
    cascade never see two dialects."""
    from expense_analyzer.enrichment.vendor_web import normalize_industry

    assert normalize_industry("supermarket") == "Supermarkt"
    assert normalize_industry("telco") == "Telekommunikation"
    assert normalize_industry("rent") == "Miete"
    assert normalize_industry("other") == "Sonstige"
    # Already-German values pass through unchanged.
    assert normalize_industry("Supermarkt") == "Supermarkt"
    # Empty / None passes through to empty (caller decides what to do).
    assert normalize_industry("") == ""
    assert normalize_industry(None) == ""
    # Case-insensitive: "TELCO" still hits the legacy map.
    assert normalize_industry("TELCO") == "Telekommunikation"


def test_is_meaningful_industry_filters_no_signal_values() -> None:
    """Cascade stages call this to skip enrichment that would only
    add a dead token like "Sonstige" to the premise / lexical overlap."""
    from expense_analyzer.enrichment.vendor_web import is_meaningful_industry

    assert is_meaningful_industry("Supermarkt") is True
    assert is_meaningful_industry("Miete") is True
    # Both spellings of the "no real signal" sentinel are filtered.
    assert is_meaningful_industry("Sonstige") is False
    assert is_meaningful_industry("sonstige") is False
    assert is_meaningful_industry("other") is False
    assert is_meaningful_industry("") is False
    assert is_meaningful_industry(None) is False


def test_cached_read_migrates_legacy_english_label(
    tmp_db: sqlite3.Connection,
) -> None:
    """A cache row written before the German-label change should
    surface as German on read, even without re-running the lookup."""
    # Insert a legacy-style row directly.
    tmp_db.execute(
        "INSERT INTO vendor_cache(counterparty_normalized, summary, industry) "
        "VALUES (?, ?, ?)",
        ("legacy vendor", "old snippet", "supermarket"),
    )
    cfg = VendorLookupConfig(enabled=True, cache_ttl_days=10)
    info = lookup_vendor(tmp_db, "legacy vendor", cfg)
    assert info.industry == "Supermarkt"
    assert info.summary == "old snippet"
