"""Read-only aggregates used by dashboards and the Categories tab."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryStat:
    id: int
    name: str
    description: str
    color: str
    n_expenses: int
    total_eur: float       # signed sum
    abs_total_eur: float   # sum of |betrag|
    last_seen: str | None  # ISO date or None


_CATEGORY_STATS_SQL = """
SELECT
    c.id, c.name, c.description, c.color,
    COUNT(e.id)                                                AS n_expenses,
    COALESCE(SUM(e.betrag_cents) / 100.0, 0.0)                 AS total_eur,
    COALESCE(SUM(ABS(e.betrag_cents)) / 100.0, 0.0)            AS abs_total_eur,
    MAX(e.buchungsdatum)                                       AS last_seen
FROM categories c
LEFT JOIN latest_label ll ON ll.category_id = c.id
LEFT JOIN expenses e ON e.id = ll.expense_id
GROUP BY c.id, c.name, c.description, c.color
ORDER BY abs_total_eur DESC, c.name
"""


def category_stats(conn: sqlite3.Connection) -> list[CategoryStat]:
    """Per-category aggregates over all time. Uses the most-recent label
    per expense, so model-assigned categories count too."""
    rows = conn.execute(_CATEGORY_STATS_SQL).fetchall()
    return [
        CategoryStat(
            id=int(r["id"]),
            name=r["name"],
            description=r["description"] or "",
            color=r["color"] or "#888",
            n_expenses=int(r["n_expenses"] or 0),
            total_eur=float(r["total_eur"] or 0.0),
            abs_total_eur=float(r["abs_total_eur"] or 0.0),
            last_seen=(str(r["last_seen"]) if r["last_seen"] is not None else None),
        )
        for r in rows
    ]


def uncategorized_stat(conn: sqlite3.Connection) -> CategoryStat:
    """Pseudo-row for expenses with no label (or whose category was deleted)."""
    row = conn.execute(
        """
        SELECT
            COUNT(e.id)                                  AS n,
            COALESCE(SUM(e.betrag_cents) / 100.0, 0.0)   AS total,
            COALESCE(SUM(ABS(e.betrag_cents)) / 100.0, 0.0) AS abs_total,
            MAX(e.buchungsdatum)                         AS last_seen
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE c.id IS NULL
        """
    ).fetchone()
    return CategoryStat(
        id=-1,
        name="(unkategorisiert)",
        description="Records with no category assigned (or whose category was deleted).",
        color="#bbbbbb",
        n_expenses=int(row["n"] or 0),
        total_eur=float(row["total"] or 0.0),
        abs_total_eur=float(row["abs_total"] or 0.0),
        last_seen=(str(row["last_seen"]) if row["last_seen"] is not None else None),
    )
