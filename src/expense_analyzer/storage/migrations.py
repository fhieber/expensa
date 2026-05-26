"""Schema migrations.

Each migration is keyed by the target ``schema_version`` it bumps the DB
to. Migrations run in order; ``init_schema`` is idempotent so applying a
migration to a fresh DB is a no-op (the CREATE TABLE IF NOT EXISTS lines
in ``schema.sql`` already cover it).

Bumping the schema version:
1. Add a new entry to ``_MIGRATIONS`` with the SQL to upgrade FROM the
   previous version.
2. Bump the literal in ``schema.sql``'s
   ``INSERT OR IGNORE INTO schema_meta(...)`` line.
3. Add ``IF NOT EXISTS`` / ``IF EXISTS`` guards in the migration so it's
   safe on a partially-upgraded DB (we run migrations on every open).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    """``ALTER TABLE ... ADD COLUMN`` guarded by a column-existence check,
    since SQLite has no ``ADD COLUMN IF NOT EXISTS``. Makes the migration
    safe to retry on a partially-applied DB.  Silently skips if the table
    itself doesn't exist (can happen in synthetic test schemas)."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        return
    existing = {row["name"] for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _migrate_v3(conn: sqlite3.Connection) -> None:
    # v2 -> v3: per-category savings flag.
    _add_column_if_missing(conn, "categories", "is_savings", "INTEGER NOT NULL DEFAULT 0")
    rows = conn.execute("PRAGMA table_info(categories)").fetchall()
    if rows:
        conn.execute("UPDATE categories SET is_savings = 1 WHERE name = 'Sparen'")


def _migrate_v4(conn: sqlite3.Connection) -> None:
    # v3 -> v4: add enrichment tracking + the now-removed enriched_* columns.
    for column, decl in (
        ("enrichment_source", "TEXT"),
        ("enrichment_ref", "TEXT"),
        ("enriched_counterparty", "TEXT"),
        ("enriched_description", "TEXT"),
        ("enriched_at", "TIMESTAMP"),
    ):
        _add_column_if_missing(conn, "expenses", column, decl)


def _migrate_v5(conn: sqlite3.Connection) -> None:
    # v4 -> v5: direct VZ rewriting replaces display-time enrichment.
    #
    # 1. For rows previously enriched via the old approach (enriched_counterparty
    #    is set), write the merchant name directly into verwendungszweck and
    #    re-derive the normalised columns.
    # 2. For existing PayPal bank rows where the merchant is already in the
    #    VZ ("Ihr Einkauf bei <merchant>"), apply the same ingest-time
    #    simplification retroactively.
    # 3. Drop the three columns that are no longer needed.
    import re as _re

    from expense_analyzer.ingestion.normalizer import combined_text as _ct
    from expense_analyzer.ingestion.normalizer import normalize_counterparty as _ncp
    from expense_analyzer.ingestion.normalizer import normalize_verwendungszweck as _nvz

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(expenses)").fetchall()}
    if not cols:
        return

    # Step 1: migrate previously-enriched rows.
    migrated_ids: list[int] = []
    if "enriched_counterparty" in cols:
        for row in conn.execute(
            "SELECT id, enriched_counterparty FROM expenses "
            "WHERE enrichment_source = 'paypal' "
            "AND enriched_counterparty IS NOT NULL AND enriched_counterparty != ''"
        ).fetchall():
            name = row["enriched_counterparty"]
            new_vz = name
            cp_norm = _ncp(name)
            conn.execute(
                "UPDATE expenses SET verwendungszweck = ?, "
                "verwendungszweck_normalized = ?, counterparty_normalized = ?, "
                "combined_text = ? WHERE id = ?",
                (new_vz, _nvz(new_vz), cp_norm, _ct(cp_norm, _nvz(new_vz)), row["id"]),
            )
            migrated_ids.append(row["id"])
    # Invalidate embeddings for migrated rows — combined_text changed so
    # any previously stored vector is now stale.
    if migrated_ids:
        ph = ",".join("?" * len(migrated_ids))
        conn.execute(f"DELETE FROM embeddings WHERE expense_id IN ({ph})", migrated_ids)

    # Step 2: simplify existing PayPal rows with "Ihr Einkauf bei <merchant>" in VZ.
    _step2_cols = {"verwendungszweck", "zahlungsempfaenger", "zahlungspflichtiger"}
    if _step2_cols.issubset(cols):
        _IEB = _re.compile(r"Ihr\s+Einkauf\s+bei\s+(.+)$", _re.IGNORECASE | _re.DOTALL)
        for row in conn.execute(
            "SELECT id, verwendungszweck, zahlungsempfaenger, zahlungspflichtiger "
            "FROM expenses WHERE enrichment_ref IS NULL"
        ).fetchall():
            zep = (row["zahlungsempfaenger"] or "").lower()
            zpf = (row["zahlungspflichtiger"] or "").lower()
            if "paypal" not in zep and "paypal" not in zpf:
                continue
            vz = row["verwendungszweck"] or ""
            m = _IEB.search(vz)
            if not m:
                continue
            merchant = m.group(1).strip()
            if not merchant:
                continue
            new_vz = merchant
            cp_norm = _ncp(merchant)
            conn.execute(
                "UPDATE expenses SET verwendungszweck = ?, "
                "verwendungszweck_normalized = ?, counterparty_normalized = ?, "
                "combined_text = ? WHERE id = ?",
                (new_vz, _nvz(new_vz), cp_norm, _ct(cp_norm, _nvz(new_vz)), row["id"]),
            )

    # Step 3: drop columns superseded by direct VZ rewriting.
    for col in ("enriched_counterparty", "enriched_description", "enriched_at"):
        if col in cols:
            conn.execute(f"ALTER TABLE expenses DROP COLUMN {col}")


# (target_version, migration) — applied in order to any DB whose current
# schema_version is *less than* the target. A migration is either a SQL
# string (run via executescript) or a callable taking the connection.
# Empty list = no pending migrations.
_MIGRATIONS: list[tuple[int, str | Callable[[sqlite3.Connection], None]]] = [
    (
        2,
        # v1 -> v2: drop the unused cluster_id column + index. SQLite
        # 3.35+ supports DROP COLUMN; we declared >=3.35 in schema.sql.
        # IF EXISTS guards make this safe on fresh DBs where the column
        # was never there.
        """
        DROP INDEX IF EXISTS idx_expenses_cluster;
        ALTER TABLE expenses DROP COLUMN cluster_id;
        """,
    ),
    (3, _migrate_v3),
    (4, _migrate_v4),
    (5, _migrate_v5),
]


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the DB's recorded schema_version, defaulting to 1 if the
    schema_meta row hasn't been written yet (i.e. ``init_schema`` is
    about to / has just installed it)."""
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        # schema_meta doesn't exist yet (very old DB, or pre-init).
        return 0
    if row is None:
        return 1
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 1


def _set_version(conn: sqlite3.Connection, v: int) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(v),),
    )


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Run any pending migrations. Returns the list of target versions
    that were applied (empty if the DB was already up to date).

    Safe to call repeatedly: a no-op when there's nothing pending.
    Each migration runs in its own implicit transaction (sqlite3's
    autocommit-with-executescript semantics); a failure leaves the
    schema_version at the prior value so the next launch retries.
    """
    applied: list[int] = []
    current = _current_version(conn)
    for target, migration in _MIGRATIONS:
        if current >= target:
            continue
        # SQL migrations: `executescript` issues an implicit COMMIT before
        # running; their IF EXISTS / IF NOT EXISTS guards (or the
        # column-existence checks in callable migrations) make re-running
        # safe on a partially-applied DB.
        if callable(migration):
            migration(conn)
        else:
            conn.executescript(migration)
        _set_version(conn, target)
        applied.append(target)
        current = target
    return applied
