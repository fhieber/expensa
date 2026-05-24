"""Tests for the schema migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.storage.database import get_or_create_database
from expense_analyzer.storage.migrations import apply_migrations


def test_fresh_db_is_at_current_schema_version(tmp_path: Path) -> None:
    conn = get_or_create_database(tmp_path / "fresh.sqlite")
    try:
        v = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        # Bump expected version when adding migrations.
        assert int(v) == 3
    finally:
        conn.close()


def test_fresh_db_no_pending_migrations(tmp_path: Path) -> None:
    conn = get_or_create_database(tmp_path / "fresh.sqlite")
    try:
        # Second call: no migrations should be applied (already current).
        assert apply_migrations(conn) == []
    finally:
        conn.close()


def test_latest_label_view_exists(tmp_path: Path) -> None:
    """The `latest_label` view is referenced from query builders all
    over the codebase. Make sure init_schema actually creates it."""
    conn = get_or_create_database(tmp_path / "v.sqlite")
    try:
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE name='latest_label'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["type"] == "view"
    finally:
        conn.close()


def test_v1_to_latest_drops_cluster_id_and_adds_is_savings(tmp_path: Path) -> None:
    """Build an old-style v1 DB by hand (cluster_id present, categories
    without is_savings), run apply_migrations and verify both the v2 drop
    and the v3 add land."""
    db_path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE expenses (
            id INTEGER PRIMARY KEY,
            buchungsdatum DATE NOT NULL,
            betrag_cents INTEGER NOT NULL,
            cluster_id INTEGER,
            dedup_hash TEXT NOT NULL UNIQUE
        );
        CREATE INDEX idx_expenses_cluster ON expenses(cluster_id);
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            color TEXT
        );
        INSERT INTO categories(name) VALUES ('Sparen'), ('Lebensmittel');
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1');
        """
    )
    # Insert a row to make sure the migration preserves data.
    conn.execute(
        "INSERT INTO expenses(buchungsdatum, betrag_cents, cluster_id, dedup_hash) "
        "VALUES ('2026-01-01', 1234, 7, 'h1')"
    )
    conn.commit()

    applied = apply_migrations(conn)
    assert applied == [2, 3]

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(expenses)").fetchall()]
    assert "cluster_id" not in cols
    # Pre-existing rows survive the column drop.
    row = conn.execute("SELECT buchungsdatum, betrag_cents FROM expenses").fetchone()
    assert int(row["betrag_cents"]) == 1234

    v = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert int(v) == 3

    # Re-running is a no-op.
    assert apply_migrations(conn) == []
    conn.close()


def test_v2_to_v3_adds_is_savings_and_backfills_sparen(tmp_path: Path) -> None:
    """A v2 DB gains the is_savings column; a pre-existing 'Sparen'
    category is flagged on, others stay off."""
    db_path = tmp_path / "v2.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            color TEXT
        );
        INSERT INTO categories(name) VALUES ('Sparen'), ('Lebensmittel');
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_meta(key, value) VALUES ('schema_version', '2');
        """
    )
    conn.commit()

    assert apply_migrations(conn) == [3]

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(categories)").fetchall()]
    assert "is_savings" in cols
    flags = {
        r["name"]: int(r["is_savings"])
        for r in conn.execute("SELECT name, is_savings FROM categories").fetchall()
    }
    assert flags == {"Sparen": 1, "Lebensmittel": 0}

    # Re-running is a no-op.
    assert apply_migrations(conn) == []
    conn.close()
