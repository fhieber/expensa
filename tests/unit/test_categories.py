"""Categories + labels CRUD tests."""

from __future__ import annotations

import sqlite3

from expense_analyzer.storage.categories import (
    add_label,
    get_category_by_name,
    import_categories_from_yaml,
    labeled_ids_with_categories,
    latest_label,
    latest_user_label,
    list_categories,
    upsert_category,
    vendor_label_distribution,
)


def test_upsert_category_idempotent(tmp_db: sqlite3.Connection) -> None:
    a = upsert_category(tmp_db, "Lebensmittel", "Supermarkt", "#0f0")
    b = upsert_category(tmp_db, "Lebensmittel", "Aldi/Edeka/REWE", "#0a0")
    assert a == b
    cat = get_category_by_name(tmp_db, "Lebensmittel")
    assert cat is not None
    assert cat.description == "Aldi/Edeka/REWE"
    assert cat.color == "#0a0"


def test_list_categories_sorted(tmp_db: sqlite3.Connection) -> None:
    upsert_category(tmp_db, "Zoo")
    upsert_category(tmp_db, "Apfel")
    names = [c.name for c in list_categories(tmp_db)]
    assert names == sorted(names)


def test_import_categories_from_yaml(tmp_db: sqlite3.Connection) -> None:
    items = [{"name": "A"}, {"name": "B", "description": "second", "color": "#abc"}]
    n = import_categories_from_yaml(tmp_db, items)
    assert n == 2
    assert {c.name for c in list_categories(tmp_db)} == {"A", "B"}


def test_add_and_query_labels(tmp_db: sqlite3.Connection) -> None:
    cid = upsert_category(tmp_db, "Test")
    # Need an expense row to FK against.
    tmp_db.execute(
        "INSERT INTO expenses(buchungsdatum, betrag_cents, dedup_hash, is_income, is_round) "
        "VALUES ('2026-01-01', -1000, 'h1', 0, 1)"
    )
    eid = tmp_db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    add_label(tmp_db, eid, cid, "user", confidence=None)
    add_label(tmp_db, eid, cid, "model", confidence=0.42)
    last = latest_label(tmp_db, eid)
    assert last is not None
    assert last[1] == "model"
    user = latest_user_label(tmp_db, eid)
    assert user == cid


def test_labeled_ids_with_categories_returns_latest_user_label(
    tmp_db: sqlite3.Connection,
) -> None:
    a = upsert_category(tmp_db, "A")
    b = upsert_category(tmp_db, "B")
    tmp_db.execute(
        "INSERT INTO expenses(buchungsdatum, betrag_cents, dedup_hash, is_income, is_round) "
        "VALUES ('2026-01-01', -1, 'x1', 0, 0)"
    )
    eid = tmp_db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    add_label(tmp_db, eid, a, "user")
    add_label(tmp_db, eid, b, "user")  # newer should win
    pairs = labeled_ids_with_categories(tmp_db, source="user")
    assert pairs == [(eid, b)]


def test_vendor_label_distribution(tmp_db: sqlite3.Connection) -> None:
    a = upsert_category(tmp_db, "A")
    b = upsert_category(tmp_db, "B")
    for h, cat in [("e1", a), ("e2", a), ("e3", a), ("e4", b)]:
        tmp_db.execute(
            "INSERT INTO expenses(buchungsdatum, betrag_cents, dedup_hash, is_income, is_round, "
            "counterparty_normalized) VALUES ('2026-01-01', -1, ?, 0, 0, 'rewe markt')",
            (h,),
        )
        eid = tmp_db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        add_label(tmp_db, eid, cat, "user")
    dist = vendor_label_distribution(tmp_db, "rewe markt")
    assert dist == {a: 3, b: 1}
