"""Optional SQLCipher-backed encryption for account databases.

Encryption is **opt-in per account** and provided by the ``sqlcipher3``
driver (install the optional ``[encryption]`` extra). When the driver is
not installed only plaintext (stdlib ``sqlite3``) databases work and the
encryption helpers raise :class:`EncryptionUnavailable`.

Whether a given file is encrypted is decided by its on-disk header: a
plaintext SQLite 3 file starts with ``b"SQLite format 3\\x00"``; a
SQLCipher file does not (its header is part of the encrypted payload).
That makes the **file itself the single source of truth** -- there is no
separate flag in ``accounts.yaml`` to keep in sync.

This module is deliberately Streamlit-free so it's exercisable in tests.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

# A plaintext SQLite 3 file begins with these 16 bytes. SQLCipher files
# do not (the page-1 header is encrypted), which is how we tell them apart.
SQLITE_MAGIC = b"SQLite format 3\x00"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EncryptionError(Exception):
    """Base class for encryption-related failures."""


class EncryptionUnavailable(EncryptionError):
    """The ``sqlcipher3`` driver isn't installed."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Database encryption requires the optional SQLCipher dependency. "
                "Install it with `pip install expensa[encryption]`."
            )
        )


class PasswordRequired(EncryptionError):
    """An encrypted DB was opened without a password."""


class WrongPassword(EncryptionError):
    """The supplied password did not decrypt the database."""


# ---------------------------------------------------------------------------
# Availability + detection
# ---------------------------------------------------------------------------


def encryption_available() -> bool:
    """True when the SQLCipher driver can be imported."""
    try:
        import sqlcipher3  # noqa: F401
    except Exception:
        return False
    return True


def _driver():
    """Return the ``sqlcipher3.dbapi2`` module or raise EncryptionUnavailable."""
    try:
        from sqlcipher3 import dbapi2 as drv
    except Exception as e:  # ImportError, or a broken native build.
        raise EncryptionUnavailable() from e
    return drv


def looks_encrypted(db_path: Path | str) -> bool:
    """True when ``db_path`` exists, is non-empty, and is *not* a plaintext
    SQLite file (i.e. it's almost certainly a SQLCipher database)."""
    p = Path(db_path)
    try:
        if not p.is_file() or p.stat().st_size == 0:
            return False
        with p.open("rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    return not head.startswith(SQLITE_MAGIC)


# ---------------------------------------------------------------------------
# SQL literal escaping
# ---------------------------------------------------------------------------


def _q(s: str) -> str:
    """Escape ``s`` as a single-quoted SQL string literal.

    Used for ``PRAGMA key``/``rekey`` and ``ATTACH DATABASE`` paths, which
    don't accept bound parameters. Doubling embedded single quotes is the
    standard SQLite escaping and prevents injection through passwords or
    file paths."""
    return "'" + s.replace("'", "''") + "'"


_adapters_registered: set[str] = set()


def _register_adapters(drv) -> None:
    """Mirror ``database._register_adapters`` onto the SQLCipher driver.

    The stdlib ``sqlite3`` adapter registry is module-scoped, so the
    separate ``sqlcipher3`` module needs its own date/datetime adapters to
    round-trip the same ISO-8601 representation."""
    name = getattr(drv, "__name__", "sqlcipher3.dbapi2")
    if name in _adapters_registered:
        return
    drv.register_adapter(date, lambda d: d.isoformat())
    drv.register_adapter(datetime, lambda d: d.isoformat(sep=" "))
    drv.register_converter("DATE", lambda b: date.fromisoformat(b.decode("ascii")))
    drv.register_converter(
        "TIMESTAMP",
        lambda b: datetime.fromisoformat(b.decode("ascii").replace(" ", "T")),
    )
    _adapters_registered.add(name)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def open_connection(db_path: Path | str, password: str | None):
    """Open an encrypted connection, applying ``PRAGMA key`` and verifying it.

    Mirrors the configuration of :func:`database.connect` (row factory,
    foreign keys, WAL, type detection) so callers can use the returned
    connection interchangeably with a plaintext one.

    Raises :class:`PasswordRequired` when ``password`` is empty,
    :class:`WrongPassword` when the key doesn't decrypt the file, and
    :class:`EncryptionUnavailable` when the driver is missing.
    """
    if not password:
        raise PasswordRequired("an encrypted database requires a password")
    drv = _driver()
    _register_adapters(drv)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = drv.connect(
        str(db_path),
        detect_types=drv.PARSE_DECLTYPES | drv.PARSE_COLNAMES,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = drv.Row
    conn.execute("PRAGMA key = " + _q(password))
    try:
        # Touching the schema forces SQLCipher to decrypt page 1; a wrong
        # key surfaces here as a DatabaseError rather than silently later.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except drv.DatabaseError as e:
        conn.close()
        raise WrongPassword(
            "incorrect password (or the file is not an encrypted database)"
        ) from e
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def verify_password(db_path: Path | str, password: str | None) -> bool:
    """True when ``password`` successfully decrypts ``db_path``.

    Propagates :class:`EncryptionUnavailable` (the caller should report the
    missing dependency) but swallows wrong/empty-password errors as False."""
    try:
        conn = open_connection(db_path, password)
    except (WrongPassword, PasswordRequired):
        return False
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Encrypt / decrypt / rekey
# ---------------------------------------------------------------------------


def _safety_copy(path: Path, tag: str) -> Path:
    """Copy ``path`` to a timestamped ``<stem>.<tag>.<ts>.sqlite`` sibling."""
    ts = int(time.time())
    dest = path.with_suffix(f".{tag}.{ts}.sqlite")
    shutil.copy2(path, dest)
    return dest


def _remove_wal_sidecars(path: Path) -> None:
    """Delete stale ``-wal`` / ``-shm`` files next to ``path``.

    After we replace the main DB file with one of a different cipher
    format, a leftover WAL from the old file would be applied on the next
    open and corrupt it. The export targets are written without WAL, so
    dropping the sidecars is safe."""
    for suffix in ("-wal", "-shm"):
        side = Path(str(path) + suffix)
        try:
            side.unlink()
        except OSError:
            pass


def encrypt_file(
    plain_path: Path | str, password: str, keep_safety: bool = True
) -> Path | None:
    """Encrypt a plaintext SQLite DB in place with ``password``.

    Exports the plaintext DB into a fresh SQLCipher file (via
    ``sqlcipher_export``) then atomically replaces the original. When
    ``keep_safety`` is True a timestamped **plaintext** copy is left
    alongside as ``<stem>.pre-encrypt.<ts>.sqlite`` so a forgotten password
    doesn't mean data loss -- the caller is expected to warn the user that
    this copy is unencrypted and should be deleted once the password is
    confirmed working.

    Returns the safety-copy path (or None). Raises if the file is missing,
    already encrypted, the password is empty, or the driver is unavailable.
    """
    if not password:
        raise PasswordRequired("a non-empty password is required to encrypt")
    drv = _driver()
    _register_adapters(drv)
    plain_path = Path(plain_path)
    if not plain_path.is_file():
        raise EncryptionError(f"no database file at {plain_path}")
    if looks_encrypted(plain_path):
        raise EncryptionError("database is already encrypted")

    tmp = Path(str(plain_path) + f".enc-tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    # Open the plaintext DB with the SQLCipher driver and *no* key, attach
    # the encrypted target, and copy every page across. autocommit so the
    # checkpoint + export take effect immediately.
    conn = drv.connect(str(plain_path), isolation_level=None)
    try:
        # Fold any committed WAL frames back into the main file *before* we
        # copy it as the plaintext safety copy below -- otherwise that copy
        # (a bare file copy of the main DB) could miss recent rows.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute(f"ATTACH DATABASE {_q(str(tmp))} AS encrypted KEY {_q(password)}")
        conn.execute("SELECT sqlcipher_export('encrypted')")
        conn.execute("DETACH DATABASE encrypted")
    except Exception:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()

    safety: Path | None = None
    if keep_safety:
        safety = _safety_copy(plain_path, "pre-encrypt")
    os.replace(tmp, plain_path)
    _remove_wal_sidecars(plain_path)
    return safety


def decrypt_file(
    enc_path: Path | str, password: str, keep_safety: bool = True
) -> Path | None:
    """Remove encryption from a SQLCipher DB in place.

    Exports the decrypted contents into a fresh plaintext file then
    atomically replaces the original. When ``keep_safety`` is True a
    timestamped copy of the still-encrypted original is kept as
    ``<stem>.pre-decrypt.<ts>.sqlite``.

    Raises :class:`WrongPassword` if ``password`` is wrong.
    """
    enc_path = Path(enc_path)
    if not looks_encrypted(enc_path):
        raise EncryptionError("database is not encrypted")
    conn = open_connection(enc_path, password)  # verifies the password
    tmp = Path(str(enc_path) + f".dec-tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        # Materialise committed WAL frames so the encrypted safety copy
        # below (a bare file copy) reflects the full database.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute(f"ATTACH DATABASE {_q(str(tmp))} AS plaintext KEY ''")
        conn.execute("SELECT sqlcipher_export('plaintext')")
        conn.execute("DETACH DATABASE plaintext")
    except Exception:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()

    safety: Path | None = None
    if keep_safety:
        safety = _safety_copy(enc_path, "pre-decrypt")
    os.replace(tmp, enc_path)
    _remove_wal_sidecars(enc_path)
    return safety


def change_password(db_path: Path | str, old_password: str, new_password: str) -> None:
    """Re-key an encrypted DB from ``old_password`` to ``new_password``.

    Raises :class:`WrongPassword` if ``old_password`` is wrong and
    :class:`PasswordRequired` if ``new_password`` is empty.
    """
    if not new_password:
        raise PasswordRequired("the new password cannot be empty")
    conn = open_connection(db_path, old_password)  # verifies old password
    try:
        conn.execute("PRAGMA rekey = " + _q(new_password))
        # Flush WAL so the re-keyed pages land in the main file before we
        # let go of the only connection holding the new key.
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        conn.close()


def export_encrypted_copy(
    src_path: Path | str,
    password: str,
    dest_path: Path | str,
    dest_password: str | None = None,
) -> Path:
    """Write an encrypted, self-contained SQLCipher copy of an encrypted DB.

    Used by the backup flow so an encrypted account's backup stays
    encrypted on disk. ``dest_password`` defaults to ``password`` (back up
    under the account's current key), so restoring the file later requires
    that same password.
    """
    dest_password = dest_password or password
    if not dest_password:
        raise PasswordRequired("the backup password cannot be empty")
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        dest_path.unlink()
    conn = open_connection(src_path, password)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute(
            f"ATTACH DATABASE {_q(str(dest_path))} AS backup KEY {_q(dest_password)}"
        )
        conn.execute("SELECT sqlcipher_export('backup')")
        conn.execute("DETACH DATABASE backup")
    except Exception:
        conn.close()
        dest_path.unlink(missing_ok=True)
        raise
    conn.close()
    return dest_path


def cipher_version() -> str | None:
    """Return the SQLCipher library version string, or None if unavailable."""
    try:
        drv = _driver()
    except EncryptionUnavailable:
        return None
    conn = drv.connect(":memory:")
    try:
        row = conn.execute("PRAGMA cipher_version").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    return row[0] if row else None
