"""Stats helpers: per-category aggregates."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.storage.categories import add_label, upsert_category
from expense_analyzer.storage.stats import (
    CategoryStat,
    category_stats,
    database_overview,
    uncategorized_stat,
)


def test_category_stats_empty_db_returns_zero_rows(tmp_db: sqlite3.Connection) -> None:
    upsert_category(tmp_db, "Empty")
    stats = category_stats(tmp_db)
    assert len(stats) == 1
    only = stats[0]
    assert only.name == "Empty"
    assert only.n_expenses == 0
    assert only.total_eur == 0.0
    assert only.last_seen is None


def test_category_stats_counts_labeled_expenses(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Lebensmittel", color="#0f0")
    rent = upsert_category(tmp_db, "Miete", color="#5e35b1")

    for cp in ("markt alpha", "markt beta", "markt gamma"):
        for r in tmp_db.execute(
            "SELECT id FROM expenses WHERE counterparty_normalized = ?", (cp,)
        ).fetchall():
            add_label(tmp_db, int(r["id"]), food, "user")
    for r in tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'vermieter'"
    ).fetchall():
        add_label(tmp_db, int(r["id"]), rent, "user")

    by_name = {s.name: s for s in category_stats(tmp_db)}
    assert by_name["Lebensmittel"].n_expenses > 0
    assert by_name["Lebensmittel"].abs_total_eur > 0
    assert by_name["Miete"].n_expenses == 2  # one per month
    assert by_name["Miete"].abs_total_eur == 1900.0
    # Stats descend by abs_total: Miete (1900 €) outweighs the supermarket spend.
    ordered = [s.name for s in category_stats(tmp_db) if s.n_expenses > 0]
    assert ordered[0] == "Miete"


def test_category_stats_uses_latest_label(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """If a record was labeled A then later re-labeled B, only B should count."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    a = upsert_category(tmp_db, "A")
    b = upsert_category(tmp_db, "B")
    rid = int(tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"])
    add_label(tmp_db, rid, a, "user")
    add_label(tmp_db, rid, b, "user")
    by_name = {s.name: s for s in category_stats(tmp_db)}
    assert by_name["A"].n_expenses == 0
    assert by_name["B"].n_expenses == 1


def test_uncategorized_stat_counts_unlabeled(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Lebensmittel")
    rewe = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'markt alpha' ORDER BY id LIMIT 1"
    ).fetchone()
    add_label(tmp_db, int(rewe["id"]), food, "user")

    u = uncategorized_stat(tmp_db)
    assert isinstance(u, CategoryStat)
    assert u.id == -1
    assert u.n_expenses == 49  # 50 ingested minus the 1 labeled


def test_database_overview_reports_structure(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "Lebensmittel")

    ov = database_overview(tmp_db)
    names = {t.name for t in ov.tables}
    # Core tables are present; the latest_label view is NOT counted as a table.
    assert {"expenses", "categories", "labels"} <= names
    assert "latest_label" not in names
    assert "latest_label" in ov.views

    expenses = next(t for t in ov.tables if t.name == "expenses")
    assert expenses.n_rows == 50
    col_names = {c.name for c in expenses.columns}
    assert {"id", "betrag_cents", "dedup_hash"} <= col_names
    # `id` is the primary key.
    assert next(c for c in expenses.columns if c.name == "id").pk is True

    assert ov.n_tables == len(ov.tables)
    assert ov.n_rows_total >= 50
    assert ov.n_columns_total == sum(t.n_columns for t in ov.tables)
    assert ov.schema_version == 5
