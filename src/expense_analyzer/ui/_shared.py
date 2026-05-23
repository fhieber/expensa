"""Shared resources for every Streamlit tab module.

Streamlit re-runs the entire script on each interaction. The two heavy
objects (Config load, DB connection, sentence-transformer load) are
cached via ``@st.cache_resource`` so the cost is paid exactly once per
session. Tab modules import the accessors here -- they don't construct
their own.

Multi-account: the active account lives in
``st.session_state["active_account_id"]``. The boot sequence
(``ensure_active_account``) seeds it on first render by reading the
registry and the persisted ``active_account`` file. Switching accounts
just rewrites the session-state key, clears the connection cache, and
triggers a rerun -- the embedder cache stays warm because the model is
global, not per-account.

This module deliberately does NOT do any UI rendering or schema work
(``init_schema`` is run inside ``connect()`` via
``get_or_create_database``).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import streamlit as st

from expense_analyzer.accounts import (
    AccountInfo,
    AccountRegistry,
    init_account_db,
    migrate_legacy_if_needed,
)
from expense_analyzer.config import (
    Config,
    GlobalConfig,
    load_config_for_account,
    load_global_config,
)
from expense_analyzer.features.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
)
from expense_analyzer.storage.database import get_or_create_database

_ACTIVE_KEY = "active_account_id"


def _global_home() -> Path:
    """Resolve the global home directory; honours ``$EXPENSE_ANALYZER_HOME``."""
    return Path(
        os.environ.get("EXPENSE_ANALYZER_HOME", "~/.expense-analyzer")
    ).expanduser()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_global_config_cached() -> GlobalConfig:
    return load_global_config()


@st.cache_resource
def _load_registry_cached(global_home_str: str) -> AccountRegistry:
    """Loaded once per session. Mutations from the UI go through
    :func:`reload_registry` so the next render sees the new state."""
    return migrate_legacy_if_needed(Path(global_home_str))


@st.cache_resource
def _connect_cached(db_path_str: str) -> sqlite3.Connection:
    return get_or_create_database(Path(db_path_str))


@st.cache_resource
def _real_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=model_name, device=device, batch_size=batch_size, verbose=False
    )


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_global_config() -> GlobalConfig:
    return _load_global_config_cached()


def get_global_home() -> Path:
    return _global_home()


def get_registry() -> AccountRegistry:
    """Account registry for the current session. Mutations elsewhere
    must call :func:`reload_registry` to invalidate this cache."""
    return _load_registry_cached(str(_global_home()))


def reload_registry() -> AccountRegistry:
    """Drop the cached registry + return the freshly-loaded copy. Call
    after ``registry.save()`` so the UI picks up the new state on the
    next render."""
    _load_registry_cached.clear()
    return get_registry()


def get_active_account() -> AccountInfo:
    """The account the UI is currently looking at.

    Order: session-state key (set by the picker) > ``active_account``
    file on disk > first registered account > legacy fallback to
    ``global_home`` itself (preserves single-account behaviour on
    brand-new installs).
    """
    registry = get_registry()
    chosen_id = st.session_state.get(_ACTIVE_KEY)
    if chosen_id:
        info = registry.get(chosen_id)
        if info is not None:
            return info
    persisted = registry.get_active_id()
    if persisted is not None:
        info = registry.get(persisted)
        if info is not None:
            st.session_state[_ACTIVE_KEY] = info.id
            return info
    rows = registry.all()
    if rows:
        st.session_state[_ACTIVE_KEY] = rows[0].id
        return rows[0]
    # Brand-new install + no legacy DB: synthesize a default pointing
    # at the global home so the rest of the UI keeps working.
    return AccountInfo(id="default", name="Default", data_dir=_global_home())


def set_active_account(account_id: str) -> None:
    """Switch the UI's active account. Persists to disk + clears the
    DB connection cache so the next ``get_conn()`` opens the right
    file. The embedder cache stays warm (model is global, not
    per-account)."""
    registry = get_registry()
    info = registry.get(account_id)
    if info is None:
        raise KeyError(account_id)
    st.session_state[_ACTIVE_KEY] = info.id
    try:
        registry.set_active_id(info.id)
    except Exception:
        # File write failed (read-only mount?) -- the session-state key
        # still wins for this session.
        pass
    _connect_cached.clear()


def get_config() -> Config:
    """Per-account resolved config (GlobalConfig + active data_dir)."""
    return load_config_for_account(get_active_account(), get_global_config())


def get_conn() -> sqlite3.Connection:
    return _connect_cached(str(get_active_account().db_path))


def get_embedder() -> Embedder:
    """The configured local sentence-transformer (no cloud calls)."""
    g = get_global_config()
    return _real_embedder(g.embedding_model, g.device, g.embedding_batch_size)


def invalidate_connection() -> None:
    """Drop the cached DB connection. Call after closing it manually
    (e.g. before a backup/restore swaps the file on disk on Windows)."""
    _connect_cached.clear()


def invalidate_global_config() -> None:
    """Drop the cached GlobalConfig. Call after writing through
    ``save_user_config`` so the next render reads the updated YAML
    without forcing a UI restart."""
    _load_global_config_cached.clear()


# ---------------------------------------------------------------------------
# Account picker support
# ---------------------------------------------------------------------------


# Session-state key prefixes the picker wipes when the user switches
# accounts. Each tab lazily re-derives its own state on the next render
# from the new DB.
_TAB_STATE_PREFIXES = (
    "dashboard_",
    "cat_",
    "data_",
    "review_",
    "own_iban_",
    "new_own_iban_",
    "confirm_",
    "new_cat_",
    "inspect_",
)


def clear_tab_state() -> None:
    """Drop every tab-scoped session_state key. Used by the account
    picker so a stale auto-label stash from account A doesn't try to
    save itself against account B's DB."""
    keys_to_drop = [
        k for k in list(st.session_state.keys())
        if any(k.startswith(p) for p in _TAB_STATE_PREFIXES)
    ]
    for k in keys_to_drop:
        st.session_state.pop(k, None)


def add_account_via_ui(name: str, with_defaults: bool = True) -> AccountInfo:
    """Create + register + bootstrap an account. Used by the Add dialog.

    On success the new account is left UNACTIVATED -- the caller is
    expected to flip it active explicitly so the rerun is the only
    place state changes."""
    registry = get_registry()
    info = registry.add(name)
    registry.save()
    conn = init_account_db(info, with_defaults=with_defaults)
    conn.close()
    reload_registry()
    return info


def remove_account_via_ui(account_id: str) -> AccountInfo:
    """Remove the registry row (files on disk stay). Returns the
    removed AccountInfo so the caller can show the data_dir path."""
    registry = get_registry()
    info = registry.remove(account_id)
    registry.save()
    reload_registry()
    return info


def rename_account_via_ui(account_id: str, new_name: str) -> AccountInfo:
    registry = get_registry()
    info = registry.rename(account_id, new_name)
    registry.save()
    reload_registry()
    return info
