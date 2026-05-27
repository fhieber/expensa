"""Tests for storage/own_ibans.py — add / remove / reflag round-trips."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from expensa.ingestion import ingest_csv
from expensa.storage.own_ibans import (
    add_own_iban,
    list_own_ibans,
    reflag_all_expenses,
    remove_own_iban,
    update_label,
)

# Valid German IBAN with correct check digits. Shows up in sample_de.csv
# as Arbeitgeber GmbH's "from" IBAN -- so ingesting + adding it as an own
# IBAN should retroactively flag the salary row.
_SALARY_IBAN = "DE12500105170648489890"


def test_list_empty(tmp_db: sqlite3.Connection) -> None:
    assert list_own_ibans(tmp_db) == []


def test_add_normalises_and_persists(tmp_db: sqlite3.Connection) -> None:
    # Mixed-case, spaces, lowercase country code -- all should normalise.
    rep = add_own_iban(tmp_db, " de12 5001 0517 0648 4898 90 ", label=" Main ")
    assert rep.n_now_self == 0  # no expenses ingested yet
    rows = list_own_ibans(tmp_db)
    assert len(rows) == 1
    assert rows[0].iban == _SALARY_IBAN
    assert rows[0].label == "Main"


def test_add_rejects_garbage(tmp_db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        add_own_iban(tmp_db, "")
    with pytest.raises(ValueError):
        add_own_iban(tmp_db, "not-an-iban")
    with pytest.raises(ValueError):
        add_own_iban(tmp_db, "DE00000000")  # too short / bad checksum


def test_add_upserts_label(tmp_db: sqlite3.Connection) -> None:
    add_own_iban(tmp_db, _SALARY_IBAN, label="Old")
    add_own_iban(tmp_db, _SALARY_IBAN, label="New")  # same iban
    rows = list_own_ibans(tmp_db)
    assert len(rows) == 1
    assert rows[0].label == "New"


def test_add_reflags_matching_expenses(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # At ingest, no own IBANs exist -> no row flagged.
    n_flagged = tmp_db.execute(
        "SELECT COUNT(*) FROM expenses WHERE iban_is_known_self = 1"
    ).fetchone()[0]
    assert n_flagged == 0

    rep = add_own_iban(tmp_db, _SALARY_IBAN, label="Salary inbound")
    # Should have flipped the salary rows.
    assert rep.n_now_self >= 1
    n_flagged = tmp_db.execute(
        "SELECT COUNT(*) FROM expenses WHERE iban_is_known_self = 1"
    ).fetchone()[0]
    assert n_flagged == rep.n_now_self


def test_remove_clears_flag(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    add_rep = add_own_iban(tmp_db, _SALARY_IBAN)
    assert add_rep.n_now_self >= 1

    rm_rep = remove_own_iban(tmp_db, _SALARY_IBAN)
    assert rm_rep.n_was_self == add_rep.n_now_self
    n_flagged = tmp_db.execute(
        "SELECT COUNT(*) FROM expenses WHERE iban_is_known_self = 1"
    ).fetchone()[0]
    assert n_flagged == 0
    assert list_own_ibans(tmp_db) == []


def test_remove_unknown_iban_is_noop(tmp_db: sqlite3.Connection) -> None:
    rep = remove_own_iban(tmp_db, _SALARY_IBAN)
    assert rep.n_now_self == 0 and rep.n_was_self == 0


def test_update_label(tmp_db: sqlite3.Connection) -> None:
    add_own_iban(tmp_db, _SALARY_IBAN, label="first")
    assert update_label(tmp_db, _SALARY_IBAN, "second") is True
    rows = list_own_ibans(tmp_db)
    assert rows[0].label == "second"
    # Clearing the label is allowed.
    assert update_label(tmp_db, _SALARY_IBAN, "  ") is True
    assert list_own_ibans(tmp_db)[0].label is None
    # Unknown IBAN -> False.
    assert update_label(tmp_db, "DE00000000000000000000", "x") is False


def test_reflag_all_repairs_drift(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Simulate a drift (manual SQL: own_ibans inserted bypassing the
    add helper). reflag_all_expenses() should set the flag retroactively."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    tmp_db.execute("INSERT INTO own_ibans(iban) VALUES (?)", (_SALARY_IBAN,))
    n_before = tmp_db.execute(
        "SELECT COUNT(*) FROM expenses WHERE iban_is_known_self = 1"
    ).fetchone()[0]
    assert n_before == 0  # drift -- not retroactively flagged

    rep = reflag_all_expenses(tmp_db)
    assert rep.n_now_self >= 1
    n_after = tmp_db.execute(
        "SELECT COUNT(*) FROM expenses WHERE iban_is_known_self = 1"
    ).fetchone()[0]
    assert n_after == rep.n_now_self
