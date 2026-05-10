"""Storage-layer tests: schema applies, tables exist, FK enforcement on."""

from __future__ import annotations

import sqlite3

from expense_analyzer.storage.database import get_or_create_database, transaction


def test_schema_creates_expected_tables(tmp_db: sqlite3.Connection) -> None:
    rows = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    expected = {
        "expenses",
        "embeddings",
        "categories",
        "labels",
        "notes",
        "vendor_cache",
        "own_ibans",
        "model_versions",
        "schema_meta",
    }
    assert expected.issubset(names), f"missing tables: {expected - names}"


def test_schema_is_idempotent(tmp_db: sqlite3.Connection, tmp_path) -> None:
    # Re-open the same file: init_schema should be a no-op.
    tmp_db.close()
    db_path = tmp_path / "db.sqlite"
    conn = get_or_create_database(db_path)
    conn = get_or_create_database(db_path)  # second call must not error
    conn.close()


def test_foreign_keys_enforced(tmp_db: sqlite3.Connection) -> None:
    # Inserting a label for a non-existent expense should fail.
    tmp_db.execute(
        "INSERT INTO categories(name, description, color) VALUES (?, ?, ?)",
        ("Test", "x", "#000"),
    )
    cat_id = tmp_db.execute("SELECT id FROM categories WHERE name='Test'").fetchone()["id"]
    try:
        tmp_db.execute(
            "INSERT INTO labels(expense_id, category_id, source) VALUES (?, ?, 'user')",
            (9999, cat_id),
        )
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "FK violation on labels.expense_id should have raised"


def test_dedup_hash_unique_constraint(tmp_db: sqlite3.Connection) -> None:
    sql = (
        "INSERT INTO expenses(buchungsdatum, betrag_cents, dedup_hash, is_income, is_round) "
        "VALUES (?, ?, ?, 0, 0)"
    )
    tmp_db.execute(sql, ("2026-01-01", -1000, "abc"))
    try:
        tmp_db.execute(sql, ("2026-01-02", -2000, "abc"))
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "duplicate dedup_hash must be rejected"


def test_transaction_rolls_back_on_error(tmp_db: sqlite3.Connection) -> None:
    try:
        with transaction(tmp_db):
            tmp_db.execute(
                "INSERT INTO categories(name) VALUES ('Rollme')",
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    rows = tmp_db.execute("SELECT name FROM categories WHERE name='Rollme'").fetchall()
    assert rows == []
