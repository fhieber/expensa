"""Category and label CRUD."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    description: str
    color: str
    is_savings: bool = False


def upsert_category(
    conn: sqlite3.Connection, name: str, description: str = "", color: str = "#888"
) -> int:
    """Insert or update a category by name. Returns its id."""
    conn.execute(
        """
        INSERT INTO categories(name, description, color) VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description=excluded.description, color=excluded.color
        """,
        (name, description, color),
    )
    row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
    return int(row["id"])


def list_categories(conn: sqlite3.Connection) -> list[Category]:
    rows = conn.execute(
        "SELECT id, name, description, color, is_savings FROM categories ORDER BY name"
    ).fetchall()
    return [
        Category(
            int(r["id"]), r["name"], r["description"] or "", r["color"] or "#888",
            bool(r["is_savings"]),
        )
        for r in rows
    ]


def get_category_by_name(conn: sqlite3.Connection, name: str) -> Category | None:
    r = conn.execute(
        "SELECT id, name, description, color, is_savings FROM categories WHERE name = ?",
        (name,),
    ).fetchone()
    if r is None:
        return None
    return Category(
        int(r["id"]), r["name"], r["description"] or "", r["color"] or "#888",
        bool(r["is_savings"]),
    )


def set_category_savings(
    conn: sqlite3.Connection, category_id: int, is_savings: bool
) -> None:
    """Flag (or unflag) a category as a savings category. Rows in a savings
    category are treated as neutral by the dashboard aggregates."""
    conn.execute(
        "UPDATE categories SET is_savings = ? WHERE id = ?",
        (1 if is_savings else 0, category_id),
    )


def savings_category_names(conn: sqlite3.Connection) -> list[str]:
    """Names of all categories flagged as savings (``is_savings = 1``)."""
    rows = conn.execute(
        "SELECT name FROM categories WHERE is_savings = 1 ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def add_label(
    conn: sqlite3.Connection,
    expense_id: int,
    category_id: int,
    source: str = "user",
    confidence: float | None = None,
) -> int:
    if source not in {"user", "model"}:
        raise ValueError(f"source must be 'user' or 'model', got {source!r}")
    cur = conn.execute(
        "INSERT INTO labels(expense_id, category_id, source, confidence) VALUES (?, ?, ?, ?)",
        (expense_id, category_id, source, confidence),
    )
    return int(cur.lastrowid)


def latest_label(conn: sqlite3.Connection, expense_id: int) -> tuple[int, str, float | None] | None:
    """Returns (category_id, source, confidence) for the most-recent label, or None."""
    r = conn.execute(
        """
        SELECT category_id, source, confidence
        FROM labels WHERE expense_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (expense_id,),
    ).fetchone()
    if r is None:
        return None
    return int(r["category_id"]), r["source"], r["confidence"]


def latest_user_label(conn: sqlite3.Connection, expense_id: int) -> int | None:
    r = conn.execute(
        """
        SELECT category_id FROM labels
        WHERE expense_id = ? AND source = 'user'
        ORDER BY id DESC LIMIT 1
        """,
        (expense_id,),
    ).fetchone()
    return int(r["category_id"]) if r else None


def labeled_ids_with_categories(
    conn: sqlite3.Connection, source: str = "user"
) -> list[tuple[int, int]]:
    """List of (expense_id, category_id) using the most recent label per expense
    of the given source. Used for training data."""
    rows = conn.execute(
        """
        SELECT l.expense_id, l.category_id
        FROM labels l
        JOIN (
            SELECT expense_id, MAX(id) AS max_id
            FROM labels WHERE source = ?
            GROUP BY expense_id
        ) latest ON l.id = latest.max_id
        """,
        (source,),
    ).fetchall()
    return [(int(r["expense_id"]), int(r["category_id"])) for r in rows]


def vendor_label_distribution(
    conn: sqlite3.Connection, counterparty_normalized: str
) -> dict[int, int]:
    """How many user labels each category has received for a given vendor."""
    rows = conn.execute(
        """
        SELECT l.category_id, COUNT(*) AS n
        FROM labels l
        JOIN expenses e ON e.id = l.expense_id
        WHERE l.source = 'user'
          AND e.counterparty_normalized = ?
        GROUP BY l.category_id
        """,
        (counterparty_normalized,),
    ).fetchall()
    return {int(r["category_id"]): int(r["n"]) for r in rows}


def import_categories_from_yaml(conn: sqlite3.Connection, items: list[dict]) -> int:
    """Bulk upsert from a list of {name, description, color} dicts."""
    n = 0
    for c in items:
        upsert_category(conn, c["name"], c.get("description", ""), c.get("color", "#888"))
        n += 1
    return n
