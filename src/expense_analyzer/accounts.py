"""Account registry: track multiple accounts each with their own SQLite DB.

The package was originally single-account: one ``db.sqlite`` under
``~/.expense-analyzer/``. Multi-account support introduces a small
top-level registry (``accounts.yaml``) so the user can keep e.g.
*Personal* and *Business* expenses separate without juggling
``$EXPENSE_ANALYZER_HOME``.

Directory layout::

    ~/.expense-analyzer/
      config.yaml            # global: ML models, vendor_lookup, streamlit
      accounts.yaml          # registry: [{id, name, data_dir}]
      active_account         # plain text: slug of currently-active account
      accounts/
        personal/
          db.sqlite
        business/
          db.sqlite

Migration from the legacy single-account layout is transparent (see
:func:`migrate_legacy_if_needed`): if ``db.sqlite`` exists at the
top-level but ``accounts.yaml`` does not, a ``"Default"`` account is
auto-registered pointing at the global home itself. No files are moved.

This module is dependency-free apart from stdlib + PyYAML so it can
be imported from anywhere in the codebase without pulling in heavy ML
dependencies.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_GLOBAL_HOME = Path("~/.expense-analyzer").expanduser()
ACCOUNTS_FILENAME = "accounts.yaml"
ACTIVE_ACCOUNT_FILENAME = "active_account"
LEGACY_DB_FILENAME = "db.sqlite"
ACCOUNTS_SUBDIR = "accounts"

# Slug rules: lowercase, 1–40 chars, [a-z0-9-], no leading/trailing dash.
_SLUG_RE = re.compile(r"[a-z0-9-]+")
_SLUG_MAX_LEN = 40


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountInfo:
    """One row in the registry."""

    id: str          # slug; primary key. e.g. "personal".
    name: str        # human-friendly display name. e.g. "Personal".
    data_dir: Path   # absolute path to this account's directory.

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db.sqlite"


class AccountNotFoundError(KeyError):
    """Raised when an account id / name doesn't match any registered row."""


# ---------------------------------------------------------------------------
# Slugification
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe slug.

    Lowercase, replaces non-[a-z0-9] with dashes, collapses runs, trims
    leading/trailing dashes, caps at 40 characters. Pure function so
    tests can pin its behaviour."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > _SLUG_MAX_LEN:
        s = s[:_SLUG_MAX_LEN].rstrip("-")
    return s


def _unique_slug(base: str, taken: set[str]) -> str:
    """Append ``-2``, ``-3``, … until the slug is free in ``taken``."""
    if not base:
        base = "account"
    if base not in taken:
        return base
    i = 2
    while True:
        candidate = f"{base}-{i}"
        if candidate not in taken:
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Atomic YAML I/O
# ---------------------------------------------------------------------------


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` atomically (tmp file + rename).

    Avoids leaving a half-written ``accounts.yaml`` behind on crash, which
    would brick the next startup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the half-written tmp file on any failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# AccountRegistry
# ---------------------------------------------------------------------------


class AccountRegistry:
    """In-memory view over ``<global_home>/accounts.yaml``.

    The on-disk format::

        accounts:
          - id: default
            name: Default
            data_dir: /home/u/.expense-analyzer
          - id: business
            name: Business
            data_dir: /home/u/.expense-analyzer/accounts/business

    ``data_dir`` is stored as-is; absolute paths preserve cross-machine
    portability if the file is copied (the user can hand-edit). The
    ``active_account`` file is a separate one-line text file holding
    just the active slug.
    """

    def __init__(self, global_home: Path, accounts: list[AccountInfo]) -> None:
        self._global_home = global_home
        self._accounts: list[AccountInfo] = list(accounts)

    # ---- I/O ----------------------------------------------------------

    @classmethod
    def load(cls, global_home: Path) -> AccountRegistry:
        """Read ``accounts.yaml`` from ``global_home`` (empty if missing)."""
        raw = _load_yaml(global_home / ACCOUNTS_FILENAME)
        rows = raw.get("accounts") or []
        accounts: list[AccountInfo] = []
        for r in rows:
            try:
                accounts.append(
                    AccountInfo(
                        id=str(r["id"]),
                        name=str(r["name"]),
                        data_dir=Path(str(r["data_dir"])).expanduser(),
                    )
                )
            except (KeyError, TypeError):
                # Skip malformed entries -- don't crash the whole UI on
                # one bad row. Saving the registry rewrites cleanly.
                continue
        return cls(global_home, accounts)

    def save(self) -> None:
        """Write ``accounts.yaml`` atomically."""
        path = self._global_home / ACCOUNTS_FILENAME
        payload = {
            "accounts": [
                {
                    "id": a.id,
                    "name": a.name,
                    "data_dir": str(a.data_dir),
                }
                for a in self._accounts
            ],
        }
        _atomic_write_yaml(path, payload)

    # ---- Read ---------------------------------------------------------

    def all(self) -> list[AccountInfo]:
        return list(self._accounts)

    def __len__(self) -> int:
        return len(self._accounts)

    def __iter__(self):
        return iter(self._accounts)

    def get(self, account_id: str) -> AccountInfo | None:
        """Look up by id (slug). Returns None if not found."""
        for a in self._accounts:
            if a.id == account_id:
                return a
        return None

    def get_by_name_or_id(self, key: str) -> AccountInfo | None:
        """Match by id first, then by case-insensitive name. CLI-friendly."""
        if not key:
            return None
        direct = self.get(key)
        if direct is not None:
            return direct
        lower = key.lower()
        for a in self._accounts:
            if a.name.lower() == lower:
                return a
        return None

    @property
    def global_home(self) -> Path:
        return self._global_home

    # ---- Mutate -------------------------------------------------------

    def add(
        self, name: str, data_dir: Path | None = None
    ) -> AccountInfo:
        """Register a new account. Auto-slugifies ``name``; if the slug
        collides, suffixes ``-2`` / ``-3`` / ... ``data_dir`` defaults
        to ``<global_home>/accounts/<slug>``.

        Raises ``ValueError`` if ``name`` slugifies to the empty string
        (e.g. an all-punctuation name). The directory itself is **not**
        created here; call :func:`init_account_db` to bootstrap it.
        """
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValueError("account name cannot be empty")
        base_slug = slugify(cleaned)
        if not base_slug:
            raise ValueError(
                f"account name {name!r} produces an empty slug -- pick a "
                "name with letters or digits"
            )
        taken = {a.id for a in self._accounts}
        slug = _unique_slug(base_slug, taken)
        if data_dir is None:
            data_dir = self._global_home / ACCOUNTS_SUBDIR / slug
        info = AccountInfo(id=slug, name=cleaned, data_dir=Path(data_dir))
        self._accounts.append(info)
        return info

    def remove(self, account_id: str) -> AccountInfo:
        """Drop a row from the registry. Does **not** delete files on
        disk (the user might want to keep them; the CLI prints the
        path). Raises :class:`AccountNotFoundError` if missing."""
        for i, a in enumerate(self._accounts):
            if a.id == account_id:
                return self._accounts.pop(i)
        raise AccountNotFoundError(account_id)

    def rename(self, account_id: str, new_name: str) -> AccountInfo:
        """Update the display name. Slug / data_dir stay put (a rename
        is purely cosmetic). Raises :class:`AccountNotFoundError`."""
        cleaned = (new_name or "").strip()
        if not cleaned:
            raise ValueError("new account name cannot be empty")
        for i, a in enumerate(self._accounts):
            if a.id == account_id:
                updated = AccountInfo(id=a.id, name=cleaned, data_dir=a.data_dir)
                self._accounts[i] = updated
                return updated
        raise AccountNotFoundError(account_id)

    # ---- Active-account I/O ------------------------------------------

    def _active_path(self) -> Path:
        return self._global_home / ACTIVE_ACCOUNT_FILENAME

    def get_active_id(self) -> str | None:
        """Return the slug from ``active_account`` if it refers to a
        registered account; otherwise None."""
        p = self._active_path()
        if not p.is_file():
            return None
        try:
            slug = p.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not slug:
            return None
        # Resolve only against the current registry -- a stale slug
        # silently falls back to "no active".
        return slug if self.get(slug) is not None else None

    def set_active_id(self, account_id: str) -> None:
        """Persist ``account_id`` as the active account. Raises
        :class:`AccountNotFoundError` if the slug isn't registered."""
        if self.get(account_id) is None:
            raise AccountNotFoundError(account_id)
        self._global_home.mkdir(parents=True, exist_ok=True)
        self._active_path().write_text(account_id + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_legacy_if_needed(global_home: Path) -> AccountRegistry:
    """Auto-register a ``Default`` account on first run if needed.

    Idempotent. The migration logic:

    * If ``accounts.yaml`` exists -> load it, no-op.
    * Else if the legacy ``db.sqlite`` exists at ``global_home`` (the
      single-account install) -> register one row pointing at
      ``global_home`` itself and save. No files are moved.
    * Otherwise -> return an empty registry (a brand-new install).

    Called at process start by both the CLI and the Streamlit UI.
    """
    global_home = Path(global_home).expanduser()
    accounts_path = global_home / ACCOUNTS_FILENAME
    if accounts_path.is_file():
        return AccountRegistry.load(global_home)

    legacy_db = global_home / LEGACY_DB_FILENAME
    registry = AccountRegistry(global_home, accounts=[])
    if legacy_db.is_file():
        # Zero-data-movement migration: point at the existing location.
        default = AccountInfo(id="default", name="Default", data_dir=global_home)
        registry = AccountRegistry(global_home, accounts=[default])
        registry.save()
        try:
            registry.set_active_id("default")
        except AccountNotFoundError:
            # Shouldn't happen: just registered the row above.
            pass
    return registry


# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------


def init_account_db(account: AccountInfo, with_defaults: bool = True) -> sqlite3.Connection:
    """Create the account's directory + DB, optionally seeding default
    categories. Returns the open connection so the caller can do
    follow-up writes (e.g. flip the active-account file).

    Safe to call repeatedly: directory creation is idempotent;
    ``get_or_create_database`` applies the schema only if needed; the
    default-category seed is upserted, not duplicated.
    """
    # Local imports keep this module free of heavy storage/ deps for
    # callers that just want to read the registry.
    from expense_analyzer.config import packaged_default_categories
    from expense_analyzer.storage.categories import import_categories_from_yaml
    from expense_analyzer.storage.database import get_or_create_database

    account.data_dir.mkdir(parents=True, exist_ok=True)
    conn = get_or_create_database(account.db_path)
    if with_defaults:
        import_categories_from_yaml(conn, packaged_default_categories())
    return conn
