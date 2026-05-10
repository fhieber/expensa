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
