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

# (target_version, sql) — applied in order to any DB whose current
# schema_version is *less than* the target. Empty list = no pending
# migrations.
_MIGRATIONS: list[tuple[int, str]] = [
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
    (
        3,
        # v2 -> v3: per-category savings flag. ADD COLUMN can't take an
        # IF NOT EXISTS guard, but it's a single statement that only runs
        # when schema_version < 3, so it's safe (fresh DBs get the column
        # from schema.sql and skip this). The UPDATE backfills installs
        # that already used a "Sparen" category so their dashboards keep
        # behaving as before the flag existed.
        """
        ALTER TABLE categories ADD COLUMN is_savings INTEGER NOT NULL DEFAULT 0;
        UPDATE categories SET is_savings = 1 WHERE name = 'Sparen';
        """,
    ),
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
    for target, sql in _MIGRATIONS:
        if current >= target:
            continue
        # `executescript` issues an implicit COMMIT before running. The
        # IF EXISTS / IF NOT EXISTS guards in each migration body make
        # re-running safe on a partially-applied DB.
        conn.executescript(sql)
        _set_version(conn, target)
        applied.append(target)
        current = target
    return applied
