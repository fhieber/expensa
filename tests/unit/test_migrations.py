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
        assert int(v) == 2
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


def test_v1_to_v2_drops_cluster_id(tmp_path: Path) -> None:
    """Build an old-style v1 DB by hand (cluster_id present, no view),
    then run apply_migrations and verify cluster_id is gone."""
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
    assert applied == [2]

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(expenses)").fetchall()]
    assert "cluster_id" not in cols
    # Pre-existing rows survive the column drop.
    row = conn.execute("SELECT buchungsdatum, betrag_cents FROM expenses").fetchone()
    assert int(row["betrag_cents"]) == 1234

    v = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert int(v) == 2

    # Re-running is a no-op.
    assert apply_migrations(conn) == []
    conn.close()
