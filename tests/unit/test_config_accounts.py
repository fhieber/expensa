"""Tests for the GlobalConfig / Config split and the multi-account
loader paths in :mod:`expensa.config`.

These pin the *contract* of the split: `GlobalConfig` is the cross-
account ML / device / vendor_lookup / streamlit settings;
`Config` adds the per-account `data_dir` / `db_filename` on top.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expensa.accounts import (
    ACCOUNTS_FILENAME,
    LEGACY_DB_FILENAME,
    AccountInfo,
    AccountRegistry,
)
from expensa.config import (
    Config,
    GlobalConfig,
    load_config,
    load_config_for_account,
    load_global_config,
    save_user_config,
)

# ---------------------------------------------------------------------------
# GlobalConfig field shape
# ---------------------------------------------------------------------------


def test_global_config_has_no_data_dir_field() -> None:
    """The whole point of the split: GlobalConfig must not own
    `data_dir` (it's per-account)."""
    assert "data_dir" not in GlobalConfig.model_fields
    assert "db_filename" not in GlobalConfig.model_fields


def test_config_inherits_from_global_config() -> None:
    """Config picks up every GlobalConfig field and adds the
    per-account ones. Existing callers that type-hint `Config` keep
    working unchanged."""
    assert issubclass(Config, GlobalConfig)
    for f in (
        "embedding_model",
        "zeroshot_model",
        "embedding_batch_size",
        "device",
        "classifier",
        "vendor_exact_match",
        "knn",
        "zeroshot",
        "category_similarity",
        "active_learning",
        "vendor_lookup",
        "streamlit",
    ):
        assert f in Config.model_fields, f"Config missing inherited field {f!r}"
    assert "data_dir" in Config.model_fields
    assert "db_filename" in Config.model_fields


def test_global_config_rejects_data_dir(monkeypatch) -> None:
    """If a stray `data_dir` survives in a YAML, GlobalConfig must
    reject it (Pydantic strict-extra would silently accept by default;
    we don't enable strict-extra so it's actually ignored). Spot-check
    that the field doesn't bleed onto the resulting instance."""
    g = GlobalConfig(**{"data_dir": "/tmp/should-not-stick"})  # type: ignore[arg-type]
    assert not hasattr(g, "data_dir")


# ---------------------------------------------------------------------------
# load_global_config
# ---------------------------------------------------------------------------


def test_load_global_config_returns_packaged_defaults_in_clean_env(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    g = load_global_config()
    # Packaged default model name -- pinning this catches a config drift
    # where the YAML loader silently fails and returns an empty dict.
    assert g.embedding_model.startswith("T-Systems-onsite/")
    assert g.streamlit.host == "127.0.0.1"


def test_load_global_config_user_overrides_apply(
    monkeypatch, tmp_path: Path
) -> None:
    """A user config.yaml override should reach GlobalConfig."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "embedding_model: my-custom/embedder\n", encoding="utf-8"
    )
    g = load_global_config()
    assert g.embedding_model == "my-custom/embedder"


def test_load_global_config_drops_per_account_keys(
    monkeypatch, tmp_path: Path
) -> None:
    """The legacy single-account config.yaml may have data_dir in it.
    GlobalConfig must drop those keys cleanly rather than blowing up."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "data_dir: /tmp/legacy-stray\n"
        "db_filename: legacy.sqlite\n"
        "embedding_model: my-custom/embedder\n",
        encoding="utf-8",
    )
    g = load_global_config()
    assert g.embedding_model == "my-custom/embedder"
    assert not hasattr(g, "data_dir")
    assert not hasattr(g, "db_filename")


# ---------------------------------------------------------------------------
# load_config_for_account
# ---------------------------------------------------------------------------


def test_load_config_for_account_uses_account_data_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    account = AccountInfo(
        id="business", name="Business", data_dir=tmp_path / "biz",
    )
    cfg = load_config_for_account(account)
    assert isinstance(cfg, Config)
    assert cfg.data_dir == tmp_path / "biz"
    assert cfg.db_path == tmp_path / "biz" / "db.sqlite"


def test_load_config_for_account_inherits_ml_settings(
    monkeypatch, tmp_path: Path
) -> None:
    """Per-account configs must inherit the global ML model settings
    (the whole reason for the split is that those are shared)."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "embedding_model: shared/across-accounts\n", encoding="utf-8"
    )
    a1 = AccountInfo(id="a", name="A", data_dir=tmp_path / "a")
    a2 = AccountInfo(id="b", name="B", data_dir=tmp_path / "b")
    c1 = load_config_for_account(a1)
    c2 = load_config_for_account(a2)
    assert c1.embedding_model == "shared/across-accounts"
    assert c2.embedding_model == "shared/across-accounts"
    assert c1.data_dir != c2.data_dir


def test_load_config_for_account_with_explicit_global_cfg(
    monkeypatch, tmp_path: Path
) -> None:
    """Passing in a GlobalConfig avoids re-reading the YAML. The
    resulting Config should reflect THAT object, not whatever's on
    disk."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    # Disk says one thing...
    (tmp_path / "config.yaml").write_text(
        "embedding_model: on-disk/model\n", encoding="utf-8"
    )
    # ...but the explicit instance says another.
    explicit = GlobalConfig(embedding_model="in-memory/model")
    account = AccountInfo(id="a", name="A", data_dir=tmp_path / "a")
    cfg = load_config_for_account(account, global_cfg=explicit)
    assert cfg.embedding_model == "in-memory/model"


# ---------------------------------------------------------------------------
# load_config -- backward-compat path
# ---------------------------------------------------------------------------


def test_load_config_falls_back_to_legacy_layout_when_registry_empty(
    monkeypatch, tmp_path: Path
) -> None:
    """A fresh tmp_path has no accounts.yaml and no legacy db.sqlite, so
    every existing test that points EXPENSA_HOME at tmp_path
    must still get a sensible Config back (data_dir = global_home)."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "db.sqlite"


def test_load_config_resolves_via_active_account(
    monkeypatch, tmp_path: Path
) -> None:
    """If a registry exists and an active account is set, load_config
    returns that account's Config."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    reg = AccountRegistry(tmp_path, accounts=[])
    biz = reg.add("Business", data_dir=tmp_path / "biz")
    reg.add("Personal", data_dir=tmp_path / "pers")
    reg.save()
    reg.set_active_id(biz.id)
    cfg = load_config()
    assert cfg.data_dir == tmp_path / "biz"


def test_load_config_picks_first_when_no_active_set(
    monkeypatch, tmp_path: Path
) -> None:
    """Without an active_account file, load_config picks the first
    registered account. Deterministic enough for the CLI; the UI
    should always explicitly set active."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("First", data_dir=tmp_path / "first")
    reg.add("Second", data_dir=tmp_path / "second")
    reg.save()
    cfg = load_config()
    assert cfg.data_dir == tmp_path / "first"


def test_load_config_migrates_legacy_db_to_default_account(
    monkeypatch, tmp_path: Path
) -> None:
    """Smoke test the end-to-end migration: an existing db.sqlite at
    the root + no accounts.yaml -> first load_config call registers the
    Default account, flips it active, and returns a Config rooted at
    the global home (no data movement)."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    (tmp_path / LEGACY_DB_FILENAME).write_bytes(b"not really a db, just present")

    cfg = load_config()
    assert cfg.data_dir == tmp_path
    # accounts.yaml was created as a side effect.
    assert (tmp_path / ACCOUNTS_FILENAME).is_file()
    reg = AccountRegistry.load(tmp_path)
    assert [a.id for a in reg.all()] == ["default"]
    assert reg.get_active_id() == "default"


def test_save_user_config_then_load_global(
    monkeypatch, tmp_path: Path
) -> None:
    """save_user_config + load_global_config roundtrip: writing the
    global YAML must propagate to the next load_global_config call.
    Pins the contract that the Settings model picker can rely on."""
    monkeypatch.setenv("EXPENSA_HOME", str(tmp_path))
    monkeypatch.delenv("EXPENSA_CONFIG", raising=False)
    save_user_config({"embedding_model": "from/picker"}, data_dir=tmp_path)
    g = load_global_config()
    assert g.embedding_model == "from/picker"


# ---------------------------------------------------------------------------
# Privacy invariant -- spot-check that the streamlit loopback validator
# still fires on the new GlobalConfig type.
# ---------------------------------------------------------------------------


def test_streamlit_host_must_be_loopback() -> None:
    with pytest.raises(ValueError):
        GlobalConfig(streamlit={"host": "0.0.0.0"})  # type: ignore[arg-type]
