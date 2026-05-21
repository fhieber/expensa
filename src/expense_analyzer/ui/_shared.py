"""Shared resources for every Streamlit tab module.

Streamlit re-runs the entire script on each interaction. The two heavy
objects (Config load, DB connection, sentence-transformer load) are
cached via ``@st.cache_resource`` so the cost is paid exactly once per
session. Tab modules import the accessors here -- they don't construct
their own.

This module deliberately does NOT do any UI rendering or schema work
(``init_schema`` is run inside ``connect()`` via
``get_or_create_database``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from expense_analyzer.config import Config, load_config
from expense_analyzer.features.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
)
from expense_analyzer.storage.database import get_or_create_database


@st.cache_resource
def _load_config_cached() -> Config:
    return load_config()


@st.cache_resource
def _connect_cached(db_path_str: str) -> sqlite3.Connection:
    return get_or_create_database(Path(db_path_str))


@st.cache_resource
def _real_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=model_name, device=device, batch_size=batch_size, verbose=False
    )


def get_config() -> Config:
    return _load_config_cached()


def get_conn() -> sqlite3.Connection:
    cfg = get_config()
    return _connect_cached(str(cfg.db_path))


def get_embedder() -> Embedder:
    """The configured local sentence-transformer (no cloud calls)."""
    cfg = get_config()
    return _real_embedder(cfg.embedding_model, cfg.device, cfg.embedding_batch_size)


def invalidate_connection() -> None:
    """Drop the cached DB connection. Call after closing it manually
    (e.g. before a backup/restore swaps the file on disk on Windows)."""
    _connect_cached.clear()
