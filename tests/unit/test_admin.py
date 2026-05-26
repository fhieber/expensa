"""Destructive-admin tests: remove_category, reset_data, reset_all."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.storage.admin import (
    category_removal_impact,
    remove_category,
    reset_all,
    reset_data,
)
from expense_analyzer.storage.categories import (
    add_label,
    list_categories,
    upsert_category,
)


def test_category_removal_impact_missing(tmp_db: sqlite3.Connection) -> None:
    impact = category_removal_impact(tmp_db, "Nope")
    assert impact.exists is False
    assert impact.n_labels == 0


def test_category_removal_impact_counts_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Food")
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized='markt alpha'"
    ).fetchall()
    for r in rows:
        add_label(tmp_db, int(r["id"]), cid, "user")
    impact = category_removal_impact(tmp_db, "Food")
    assert impact.exists is True
    assert impact.n_labels == len(rows) > 0


def test_remove_category_with_no_labels(tmp_db: sqlite3.Connection) -> None:
    upsert_category(tmp_db, "Empty")
    result = remove_category(tmp_db, "Empty")
    assert result.deleted is True
    assert result.n_labels_deleted == 0
    assert "Empty" not in [c.name for c in list_categories(tmp_db)]


def test_remove_category_cascades_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Food")
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized='markt alpha' LIMIT 3"
    ).fetchall()
    for r in rows:
        add_label(tmp_db, int(r["id"]), cid, "user")
    n_before = tmp_db.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
    result = remove_category(tmp_db, "Food")
    assert result.deleted is True
    assert result.n_labels_deleted == 3
    n_after = tmp_db.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
    assert n_after == n_before - 3


def test_remove_nonexistent_category(tmp_db: sqlite3.Connection) -> None:
    result = remove_category(tmp_db, "Ghost")
    assert result.deleted is False
    assert result.n_labels_deleted == 0


def test_reset_data_keeps_categories(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "KeepMe")
    report = reset_data(tmp_db)
    assert report.table_counts["expenses"] == 50
    n_exp = tmp_db.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    assert n_exp == 0
    # Categories are preserved.
    assert "KeepMe" in [c.name for c in list_categories(tmp_db)]


def test_reset_all_drops_categories_too(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "WipeMe")
    report = reset_all(tmp_db)
    assert report.table_counts["expenses"] == 50
    assert report.table_counts["categories"] >= 1
    assert list_categories(tmp_db) == []


def test_reset_data_on_empty_db(tmp_db: sqlite3.Connection) -> None:
    report = reset_data(tmp_db)
    assert report.total == 0


def test_delete_user_labels_keeps_model_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """delete_user_labels() should wipe source='user' rows and leave
    source='model' rows alone, so subsequent latest_label queries fall
    back to the model entry."""
    from expense_analyzer.storage.admin import delete_user_labels

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Food")
    other = upsert_category(tmp_db, "Other")
    # Pick a row, give it both a model label (older) and a user label (newer).
    eid = int(
        tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"]
    )
    add_label(tmp_db, eid, other, "model", confidence=0.4)
    add_label(tmp_db, eid, food, "user")

    n_user_before = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE source='user'"
    ).fetchone()["n"]
    n_model_before = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE source='model'"
    ).fetchone()["n"]
    assert n_user_before == 1
    assert n_model_before == 1

    n_deleted = delete_user_labels(tmp_db)
    assert n_deleted == 1

    n_user_after = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE source='user'"
    ).fetchone()["n"]
    n_model_after = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE source='model'"
    ).fetchone()["n"]
    assert n_user_after == 0
    assert n_model_after == 1  # model entries kept


def test_delete_user_labels_returns_zero_when_empty(tmp_db: sqlite3.Connection) -> None:
    from expense_analyzer.storage.admin import delete_user_labels

    assert delete_user_labels(tmp_db) == 0


# ─── Whitespace backfill ──────────────────────────────────────────────


def test_collapse_text_whitespace_cleans_existing_rows(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Pre-cleanup-fix rows would have multi-space / tab runs in the
    raw text columns. This backfill helper rewrites them in place
    without touching ids, amounts, dates, labels or embeddings."""
    from expense_analyzer.storage.admin import collapse_text_whitespace

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Plant noisy whitespace directly into a few rows to simulate
    # data ingested under the old (pre-fix) code-path.
    tmp_db.execute(
        "UPDATE expenses SET counterparty = ?, verwendungszweck = ? WHERE id = 1",
        ("PayPal Europe S.a.r.l.   et Cie\t22-24 Boulevard\tRoyal",
         "1234/PP.5678.PP/.    Ihr Einkauf  bei"),
    )
    tmp_db.execute(
        "UPDATE expenses SET zahlungspflichtiger = ? WHERE id = 2",
        ("Some\t\tPayer  Name  ",),
    )
    # zahlungsempfaenger is a separate column on `expenses` (not just
    # an alias for `counterparty`). The first version of this backfill
    # left it out; pinning it here so the regression can't sneak back.
    tmp_db.execute(
        "UPDATE expenses SET zahlungsempfaenger = ? WHERE id = 3",
        ("PayPal Europe  S.a.r.l.\t\t22-24 Boulevard Royal, 2449 Luxembourg",),
    )

    report = collapse_text_whitespace(tmp_db)
    assert report.rows_scanned >= 3
    # Three rows had noise; all three should report as updated.
    assert report.rows_updated == 3
    # Four cells changed across those three rows.
    assert report.fields_changed == 4

    row1 = tmp_db.execute(
        "SELECT counterparty, verwendungszweck FROM expenses WHERE id = 1"
    ).fetchone()
    assert row1["counterparty"] == "PayPal Europe S.a.r.l. et Cie 22-24 Boulevard Royal"
    assert row1["verwendungszweck"] == "1234/PP.5678.PP/. Ihr Einkauf bei"
    row3 = tmp_db.execute(
        "SELECT zahlungsempfaenger FROM expenses WHERE id = 3"
    ).fetchone()
    assert row3["zahlungsempfaenger"] == (
        "PayPal Europe S.a.r.l. 22-24 Boulevard Royal, 2449 Luxembourg"
    )
    row2 = tmp_db.execute(
        "SELECT zahlungspflichtiger FROM expenses WHERE id = 2"
    ).fetchone()
    assert row2["zahlungspflichtiger"] == "Some Payer Name"


def test_collapse_text_whitespace_is_idempotent(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Running the backfill on an already-clean DB must report zero
    updates -- otherwise users would dirty their DBs by re-running."""
    from expense_analyzer.storage.admin import collapse_text_whitespace

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # First pass (the fixture is already clean -- ingestion strips
    # whitespace -- so this should be a no-op too).
    first = collapse_text_whitespace(tmp_db)
    assert first.rows_updated == 0
    # Second pass on the post-first state: still nothing.
    second = collapse_text_whitespace(tmp_db)
    assert second.rows_updated == 0
    assert second.fields_changed == 0


def test_collapse_text_whitespace_leaves_none_columns_alone(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """enrichment_ref is NULL for un-enriched rows. The backfill must
    not turn NULL into an empty string."""
    from expense_analyzer.storage.admin import collapse_text_whitespace

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    before = tmp_db.execute(
        "SELECT enrichment_ref FROM expenses WHERE id = 1"
    ).fetchone()
    assert before["enrichment_ref"] is None  # baseline
    collapse_text_whitespace(tmp_db)
    after = tmp_db.execute(
        "SELECT enrichment_ref FROM expenses WHERE id = 1"
    ).fetchone()
    assert after["enrichment_ref"] is None
