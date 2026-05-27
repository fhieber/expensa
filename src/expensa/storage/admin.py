"""Destructive admin operations: removing categories, resetting the DB.

Every function here returns a small dataclass describing what was
deleted so the CLI / UI can show meaningful feedback before and after
the operation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Tables we wipe on a "data" reset (everything ingested + ML state).
# Categories and own_ibans are config-like; reset_data() leaves them alone.
_DATA_TABLES = (
    "labels",
    "notes",
    "embeddings",
    "vendor_cache",
    "model_versions",
    "expenses",
)

_CONFIG_TABLES = (
    "categories",
    "own_ibans",
)


@dataclass
class CategoryRemovalImpact:
    name: str
    exists: bool
    n_labels: int  # how many label rows reference this category


@dataclass
class CategoryRemoval:
    name: str
    deleted: bool
    n_labels_deleted: int


@dataclass
class ResetReport:
    table_counts: dict[str, int]  # table -> rows deleted

    @property
    def total(self) -> int:
        return sum(self.table_counts.values())


def category_removal_impact(conn: sqlite3.Connection, name: str) -> CategoryRemovalImpact:
    """Show what removing this category would touch."""
    row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
    if row is None:
        return CategoryRemovalImpact(name=name, exists=False, n_labels=0)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE category_id = ?", (int(row["id"]),)
    ).fetchone()["n"]
    return CategoryRemovalImpact(name=name, exists=True, n_labels=int(n))


def remove_category(conn: sqlite3.Connection, name: str) -> CategoryRemoval:
    """Delete a category. Cascades to its labels via FK ON DELETE CASCADE.

    Caller is responsible for prompting the user when n_labels > 0.
    """
    row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
    if row is None:
        return CategoryRemoval(name=name, deleted=False, n_labels_deleted=0)
    cat_id = int(row["id"])
    n_labels = conn.execute(
        "SELECT COUNT(*) AS n FROM labels WHERE category_id = ?", (cat_id,)
    ).fetchone()["n"]
    conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
    return CategoryRemoval(name=name, deleted=True, n_labels_deleted=int(n_labels))


def _row_counts(conn: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tables:
        out[t] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"])
    return out


def reset_data(conn: sqlite3.Connection) -> ResetReport:
    """Wipe every ingested expense plus all derived ML state.

    Keeps categories and own_ibans (config-like)."""
    counts = _row_counts(conn, _DATA_TABLES)
    # Order matters only for symmetry; FK cascades handle the rest.
    for t in _DATA_TABLES:
        conn.execute(f"DELETE FROM {t}")
    return ResetReport(table_counts=counts)


def reset_all(conn: sqlite3.Connection) -> ResetReport:
    """Wipe **everything** including categories and own_ibans."""
    tables = _DATA_TABLES + _CONFIG_TABLES
    counts = _row_counts(conn, tables)
    for t in tables:
        conn.execute(f"DELETE FROM {t}")
    return ResetReport(table_counts=counts)


def delete_user_labels(conn: sqlite3.Connection) -> int:
    """Delete every row in ``labels`` with ``source='user'``.

    Useful when you want to re-run auto-label across the whole DB without
    your previous user confirmations dominating the cascade. Model labels
    are kept, so rows that had both a user and a model label keep their
    visible category (via the latest_label CTE picking the remaining
    model entry); rows that only had a user label become uncategorized.

    Returns the number of rows deleted.
    """
    cur = conn.execute("DELETE FROM labels WHERE source = 'user'")
    return cur.rowcount or 0


# Text columns on ``expenses`` that may carry tab/multi-space runs from
# bank exports. Ordered to keep the SQL update list readable; not
# semantically meaningful.
_WHITESPACE_BACKFILL_COLUMNS: tuple[str, ...] = (
    "counterparty",
    "verwendungszweck",
    "zahlungspflichtiger",
    "zahlungsempfaenger",
    "status",
    "umsatztyp",
    "glaeubiger_id",
    "mandatsreferenz",
    "kundenreferenz",
)


@dataclass
class WhitespaceBackfillReport:
    rows_scanned: int
    rows_updated: int
    fields_changed: int  # total cell-level changes (one row can contribute many)


def collapse_text_whitespace(conn: sqlite3.Connection) -> WhitespaceBackfillReport:
    """Rewrite every text column on ``expenses`` to collapse internal
    whitespace runs (tabs / multi-space) to a single space.

    Idempotent -- re-running on an already-clean DB updates nothing.
    Useful as a one-shot backfill after the equivalent fix landed at
    ingest time, so existing rows imported under the old code-path
    get the same treatment without losing labels, categories or
    embeddings (none of which depend on the raw text fields directly).
    """
    # Lazy import to keep this module free of an ingestion dep when
    # only the cleanup is needed (e.g. inside a fresh REPL).
    from expensa.ingestion.csv_loader import _clean_text

    cols = _WHITESPACE_BACKFILL_COLUMNS
    select_cols = ", ".join(cols)
    rows = conn.execute(
        f"SELECT id, {select_cols} FROM expenses"
    ).fetchall()

    rows_updated = 0
    fields_changed = 0
    for row in rows:
        updates: dict[str, str] = {}
        for col in cols:
            raw = row[col]
            if raw is None:
                continue
            cleaned = _clean_text(raw)
            if cleaned != raw:
                updates[col] = cleaned
        if not updates:
            continue
        set_clause = ", ".join(f"{c} = ?" for c in updates)
        conn.execute(
            f"UPDATE expenses SET {set_clause} WHERE id = ?",
            (*updates.values(), int(row["id"])),
        )
        rows_updated += 1
        fields_changed += len(updates)
    return WhitespaceBackfillReport(
        rows_scanned=len(rows),
        rows_updated=rows_updated,
        fields_changed=fields_changed,
    )


def clear_labels_for_expense(conn: sqlite3.Connection, expense_id: int) -> int:
    """Delete EVERY label row (user + model) for a single expense.

    Used by the UI when the user explicitly blanks the Category cell --
    they want the row to read as uncategorized. We can't express "no label"
    via INSERT because ``labels.category_id`` is NOT NULL, so we drop the
    existing rows instead. The latest_label CTE will then return nothing
    for this expense and the row shows as ``(unkategorisiert)``.
    """
    cur = conn.execute("DELETE FROM labels WHERE expense_id = ?", (int(expense_id),))
    return cur.rowcount or 0
