"""Tests for storage/backup.py — export, validate, restore round-trip."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.storage.backup import (
    REQUIRED_TABLES,
    export_database,
    restore_database,
    validate_backup,
)
from expense_analyzer.storage.categories import add_label, upsert_category
from expense_analyzer.storage.database import get_or_create_database


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        for t in REQUIRED_TABLES
    }


def test_export_creates_valid_sqlite(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    dest = tmp_path / "backup.sqlite"
    out = export_database(tmp_db, dest)
    assert out == dest
    assert dest.exists() and dest.stat().st_size > 0
    # File should be openable as a sqlite db with the SAME row counts.
    src_counts = _row_counts(tmp_db)
    with sqlite3.connect(str(dest)) as copy:
        copy.row_factory = sqlite3.Row
        for t, n in src_counts.items():
            got = int(copy.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            assert got == n, f"row count mismatch for {t}: src={n} copy={got}"


def test_validate_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")
    r = validate_backup(empty)
    assert r.ok is False
    assert any("empty" in e.lower() for e in r.errors)


def test_validate_rejects_non_sqlite(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.sqlite"
    bogus.write_bytes(b"not a sqlite file at all")
    r = validate_backup(bogus)
    assert r.ok is False
    assert any("magic" in e.lower() or "sqlite" in e.lower() for e in r.errors)


def test_validate_rejects_sqlite_without_required_tables(tmp_path: Path) -> None:
    """A real sqlite file but with the wrong schema gets rejected."""
    stranger = tmp_path / "stranger.sqlite"
    conn = sqlite3.connect(str(stranger))
    conn.execute("CREATE TABLE foo (id INTEGER)")
    conn.commit()
    conn.close()
    r = validate_backup(stranger)
    assert r.ok is False
    assert any("missing required table" in e.lower() for e in r.errors)


def test_validate_accepts_fresh_db(tmp_db: sqlite3.Connection, tmp_path: Path) -> None:
    """A freshly-initialised DB (no data) is still a valid backup."""
    dest = tmp_path / "fresh.sqlite"
    export_database(tmp_db, dest)
    r = validate_backup(dest)
    assert r.ok is True
    assert set(r.table_counts.keys()) == REQUIRED_TABLES


def test_restore_round_trip(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    # Seed source: ingest + add a user label.
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    eid = int(tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"])
    add_label(tmp_db, eid, cid, "user")
    src_counts = _row_counts(tmp_db)

    # Backup.
    bk = tmp_path / "backup.sqlite"
    export_database(tmp_db, bk)
    # Close the source so windows can swap files.
    tmp_db.close()

    # Restore into a brand-new path.
    target_dir = tmp_path / "restored"
    target_dir.mkdir()
    target = target_dir / "db.sqlite"
    # Make the target a valid (empty) DB first so the file exists.
    get_or_create_database(target).close()
    report = restore_database(target, bk, keep_safety=True)

    # Reopen and check.
    conn2 = sqlite3.connect(str(target))
    conn2.row_factory = sqlite3.Row
    try:
        for t, n in src_counts.items():
            got = int(conn2.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            assert got == n
    finally:
        conn2.close()

    assert report.restored_from == bk
    assert report.safety_copy is not None
    assert report.safety_copy.exists()
    assert ".pre-restore." in report.safety_copy.name


def test_restore_rejects_invalid_source(tmp_path: Path) -> None:
    target = tmp_path / "current.sqlite"
    get_or_create_database(target).close()
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"junk")
    with pytest.raises(ValueError, match="not valid"):
        restore_database(target, bad)


def test_restore_no_safety_copy_when_disabled(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    bk = tmp_path / "bk.sqlite"
    export_database(tmp_db, bk)
    tmp_db.close()

    target = tmp_path / "target.sqlite"
    get_or_create_database(target).close()
    report = restore_database(target, bk, keep_safety=False)
    assert report.safety_copy is None
    # The target directory should NOT contain a *.pre-restore.*.sqlite.
    pre = list(tmp_path.glob("**/*.pre-restore.*.sqlite"))
    assert pre == []


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sqlcipher3") is None,
    reason="SQLCipher driver not installed",
)
def test_encrypted_backup_validate_and_restore_round_trip(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """An encrypted account backs up to an encrypted file; validate/restore
    require the password and yield an encrypted database."""
    from expense_analyzer.storage import crypto

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    eid = int(tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"])
    add_label(tmp_db, eid, cid, "user")
    src_counts = _row_counts(tmp_db)

    # Stand up an encrypted account DB, then export the encrypted backup.
    account_db = tmp_path / "account.sqlite"
    export_database(tmp_db, account_db)
    tmp_db.close()
    crypto.encrypt_file(account_db, "secret", keep_safety=False)

    bk = tmp_path / "backup.enc.sqlite"
    crypto.export_encrypted_copy(account_db, "secret", bk)
    assert crypto.looks_encrypted(bk) is True

    # Validation: flagged-not-accepted without a password; rejected on wrong.
    no_pw = validate_backup(bk)
    assert no_pw.ok is False and no_pw.encrypted is True
    assert validate_backup(bk, password="nope").ok is False
    ok = validate_backup(bk, password="secret")
    assert ok.ok is True and ok.encrypted is True

    # Restore into a fresh target; it stays encrypted under the backup key.
    target = tmp_path / "restored" / "db.sqlite"
    target.parent.mkdir()
    get_or_create_database(target).close()
    report = restore_database(target, bk, keep_safety=True, password="secret")
    assert crypto.looks_encrypted(target) is True
    assert ".pre-restore." in report.safety_copy.name

    conn2 = get_or_create_database(target, "secret")
    try:
        for t, n in src_counts.items():
            got = int(conn2.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            assert got == n
    finally:
        conn2.close()
