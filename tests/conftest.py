"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from expense_analyzer.config import Config
from expense_analyzer.storage.database import get_or_create_database


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A Config pointing at a temp data dir. Uses defaults otherwise."""
    return Config(data_dir=tmp_path)


@pytest.fixture
def tmp_db(tmp_config: Config):
    """A freshly-initialized SQLite DB at tmp_config.db_path."""
    conn = get_or_create_database(tmp_config.db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
