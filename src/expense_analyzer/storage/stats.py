"""Read-only aggregates used by dashboards and the Categories tab."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


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
    is_savings: bool = False


_CATEGORY_STATS_SQL = """
SELECT
    c.id, c.name, c.description, c.color, c.is_savings,
    COUNT(e.id)                                                AS n_expenses,
    COALESCE(SUM(e.betrag_cents) / 100.0, 0.0)                 AS total_eur,
    COALESCE(SUM(ABS(e.betrag_cents)) / 100.0, 0.0)            AS abs_total_eur,
    MAX(e.buchungsdatum)                                       AS last_seen
FROM categories c
LEFT JOIN latest_label ll ON ll.category_id = c.id
LEFT JOIN expenses e ON e.id = ll.expense_id
GROUP BY c.id, c.name, c.description, c.color, c.is_savings
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
            is_savings=bool(r["is_savings"]),
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


# ---------------------------------------------------------------------------
# Database structure overview (Settings → Database)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    pk: bool


@dataclass(frozen=True)
class TableInfo:
    name: str
    n_rows: int
    columns: list[ColumnInfo]

    @property
    def n_columns(self) -> int:
        return len(self.columns)


@dataclass(frozen=True)
class DatabaseOverview:
    schema_version: int | None
    tables: list[TableInfo]
    views: list[str] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    @property
    def n_rows_total(self) -> int:
        return sum(t.n_rows for t in self.tables)

    @property
    def n_columns_total(self) -> int:
        return sum(t.n_columns for t in self.tables)


def _schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return None


def database_overview(conn: sqlite3.Connection) -> DatabaseOverview:
    """Introspect the live DB: user tables with row + column counts, plus
    the views and indexes. Powers the detailed Settings → Database stats.

    Reads ``sqlite_master`` and ``PRAGMA table_info`` -- works identically
    on plaintext and encrypted (already-unlocked) connections."""
    obj_rows = conn.execute(
        """
        SELECT name, type FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    table_names = [r["name"] for r in obj_rows if r["type"] == "table"]
    views = [r["name"] for r in obj_rows if r["type"] == "view"]
    indexes = [r["name"] for r in obj_rows if r["type"] == "index"]

    tables: list[TableInfo] = []
    for name in table_names:
        # Identifiers can't be bound; quote defensively against odd names.
        quoted = '"' + name.replace('"', '""') + '"'
        cols = [
            ColumnInfo(
                name=c["name"],
                type=(c["type"] or ""),
                notnull=bool(c["notnull"]),
                pk=bool(c["pk"]),
            )
            for c in conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        ]
        n_rows = conn.execute(f"SELECT COUNT(*) AS n FROM {quoted}").fetchone()["n"]
        tables.append(TableInfo(name=name, n_rows=int(n_rows), columns=cols))

    return DatabaseOverview(
        schema_version=_schema_version(conn),
        tables=tables,
        views=views,
        indexes=indexes,
    )
