"""SQLite-backed storage. Uses raw sqlite3 (no SQLAlchemy ORM) to keep
the dependency surface small and the SQL transparent."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from importlib import resources
from pathlib import Path

SCHEMA_RESOURCE = ("expensa.storage", "schema.sql")


def _register_adapters() -> None:
    """Re-add the date/datetime adapters Python 3.12+ deprecated.

    We standardize on ISO-8601 (YYYY-MM-DD / YYYY-MM-DDTHH:MM:SS) which
    sorts lexicographically and round-trips cleanly.
    """
    sqlite3.register_adapter(date, lambda d: d.isoformat())
    sqlite3.register_adapter(datetime, lambda d: d.isoformat(sep=" "))
    sqlite3.register_converter("DATE", lambda b: date.fromisoformat(b.decode("ascii")))
    sqlite3.register_converter(
        "TIMESTAMP",
        lambda b: datetime.fromisoformat(b.decode("ascii").replace(" ", "T")),
    )


_register_adapters()


def _read_schema() -> str:
    pkg, name = SCHEMA_RESOURCE
    return resources.files(pkg).joinpath(name).read_text(encoding="utf-8")


def connect(db_path: Path, password: str | None = None) -> sqlite3.Connection:
    """Open a connection with sane defaults: foreign keys, row factory, WAL.

    When ``password`` is given, or the file on disk is already a SQLCipher
    database, the connection is opened through the optional SQLCipher driver
    (see :mod:`expensa.storage.crypto`) with the same configuration.
    Plaintext databases keep using the stdlib ``sqlite3`` driver, so the
    encryption dependency stays optional.

    `check_same_thread=False` is intentional: Streamlit reruns the script on
    different worker threads, and the CLI ships a single-user, single-process
    application. SQLite itself is built in serialized threading mode, and
    Streamlit serializes reruns within a session, so sharing one connection
    across threads is safe here.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Local import keeps the optional SQLCipher dependency off the hot path
    # for plaintext databases (the common case).
    from expensa.storage import crypto

    if password is not None or crypto.looks_encrypted(db_path):
        return crypto.open_connection(db_path, password)

    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        isolation_level=None,  # autocommit; explicit BEGIN where needed
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply ``schema.sql`` (idempotent) then run any pending migrations.

    `schema.sql` reflects the *current* shape of the DB and uses
    ``CREATE TABLE IF NOT EXISTS`` everywhere, so on a fresh DB it lands
    the v-N schema directly and migrations are no-ops. On an older DB
    the IF NOT EXISTS lines skip existing tables and the migration
    runner brings the rest up to date.
    """
    from expensa.storage.migrations import apply_migrations

    conn.executescript(_read_schema())
    apply_migrations(conn)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction. Uses BEGIN IMMEDIATE to avoid lock surprises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def get_or_create_database(
    db_path: Path, password: str | None = None
) -> sqlite3.Connection:
    """Open the DB, applying the schema if the file is new.

    Pass ``password`` for an encrypted account (see :func:`connect`)."""
    conn = connect(db_path, password)
    init_schema(conn)
    return conn
