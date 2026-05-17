"""Backup / restore helpers.

A backup is a complete copy of the SQLite database file --- created via
SQLite's online ``conn.backup()`` API so it's safe to take even with the
live UI connection open.  The file is fully self-contained: you can open
it in any SQLite browser, drop it back in via ``restore_database()``, or
ship it to another machine.

Restore replaces the current DB file in-place.  A timestamped safety copy
of the pre-restore DB is created next to it by default so a bad backup
can be rolled back manually.  Callers (notably the Streamlit UI) are
responsible for closing any live ``sqlite3.Connection`` and invalidating
caches before calling :func:`restore_database`.

The module is deliberately Streamlit-free so it's exercisable in tests.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Tables a valid backup must contain. Bumping this list is a breaking
# change -- old backups would be rejected.
REQUIRED_TABLES = frozenset({
    "expenses",
    "labels",
    "categories",
    "notes",
    "embeddings",
    "vendor_cache",
    "model_versions",
    "own_ibans",
})

# Bytes a SQLite 3 file starts with. Lets us fail fast on obviously
# non-sqlite uploads (HTML error pages, random files etc.).
_SQLITE_MAGIC = b"SQLite format 3\x00"


@dataclass
class ValidationResult:
    ok: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class RestoreReport:
    restored_from: Path
    safety_copy: Path | None
    table_counts: dict[str, int]


def export_database(conn: sqlite3.Connection, dest_path: Path) -> Path:
    """Write a full copy of ``conn``'s database to ``dest_path``.

    Uses SQLite's online backup API so concurrent reads/writes on the
    source are safe. The destination file is created (or truncated) and
    is a complete, self-contained SQLite 3 file.

    Returns the resolved destination path.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # If a stale file from a prior backup is here, drop it so we don't
    # end up with two databases attached (the API errors otherwise).
    if dest_path.exists():
        dest_path.unlink()
    # `with sqlite3.connect(...)` ensures the dest connection closes even
    # on error.  pages=-1 copies the whole DB in one pass.
    with sqlite3.connect(str(dest_path)) as dest:
        conn.backup(dest, pages=-1)
    return dest_path


def validate_backup(source_path: Path) -> ValidationResult:
    """Inspect ``source_path`` and report whether it's a usable backup.

    Checks (in order):
        * file exists and is non-empty
        * first 16 bytes are the SQLite 3 magic
        * file opens as a SQLite DB without ``DatabaseError``
        * every name in :data:`REQUIRED_TABLES` exists
    Returns a :class:`ValidationResult` with row counts (empty when
    invalid) and a list of human-readable error messages.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        return ValidationResult(ok=False, errors=[f"file not found: {source_path}"])
    if source_path.stat().st_size == 0:
        return ValidationResult(ok=False, errors=["file is empty"])

    try:
        with source_path.open("rb") as fh:
            head = fh.read(16)
    except OSError as e:
        return ValidationResult(ok=False, errors=[f"cannot read file: {e}"])
    if not head.startswith(_SQLITE_MAGIC):
        return ValidationResult(
            ok=False,
            errors=["not a SQLite 3 file (magic bytes don't match)"],
        )

    counts: dict[str, int] = {}
    errors: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            present = {r[0] for r in rows}
            missing = REQUIRED_TABLES - present
            if missing:
                errors.append(
                    "missing required table(s): " + ", ".join(sorted(missing))
                )
            # Count rows in every required (and present) table.
            for t in sorted(REQUIRED_TABLES & present):
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    counts[t] = int(n)
                except sqlite3.DatabaseError as e:
                    errors.append(f"count {t}: {e}")
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return ValidationResult(ok=False, errors=[f"sqlite open failed: {e}"])

    return ValidationResult(ok=not errors, table_counts=counts, errors=errors)


def restore_database(
    current_path: Path,
    source_path: Path,
    keep_safety: bool = True,
) -> RestoreReport:
    """Replace the DB at ``current_path`` with ``source_path``.

    Validates the source first (raises ``ValueError`` if invalid). If
    ``keep_safety`` is True (default) and a current DB exists, a
    timestamped copy is saved alongside it as
    ``<stem>.pre-restore.<unix_ts>.sqlite`` so the user can roll back
    manually.

    Caller MUST close any live ``sqlite3.Connection`` to ``current_path``
    before invoking this (e.g. ``conn.close()`` + clear streamlit's
    cache). Failing to do so on Windows raises ``PermissionError`` on
    the file replace.
    """
    current_path = Path(current_path)
    source_path = Path(source_path)
    result = validate_backup(source_path)
    if not result.ok:
        raise ValueError(
            "backup is not valid: " + "; ".join(result.errors or ["unknown"])
        )

    safety_copy: Path | None = None
    if keep_safety and current_path.exists():
        ts = int(time.time())
        safety_copy = current_path.with_suffix(f".pre-restore.{ts}.sqlite")
        shutil.copy2(current_path, safety_copy)

    current_path.parent.mkdir(parents=True, exist_ok=True)
    # shutil.copy2 (instead of move) preserves the source backup file in
    # case the caller hands us their original upload.
    shutil.copy2(source_path, current_path)

    return RestoreReport(
        restored_from=source_path,
        safety_copy=safety_copy,
        table_counts=result.table_counts,
    )
