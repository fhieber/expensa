"""Config loader tests: defaults load, user overrides merge, privacy invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from expense_analyzer.config import (
    Config,
    StreamlitConfig,
    load_config,
    packaged_default_categories,
    packaged_default_config,
)


def test_packaged_default_config_has_required_keys() -> None:
    cfg = packaged_default_config()
    for k in ("data_dir", "embedding_model", "zeroshot_model", "classifier", "vendor_lookup"):
        assert k in cfg


def test_packaged_default_categories_is_nonempty_german() -> None:
    cats = packaged_default_categories()
    assert cats, "default categories should not be empty"
    names = {c["name"] for c in cats}
    # At least a few canonical German categories should be present.
    assert {"Lebensmittel", "Miete", "Einkommen"}.issubset(names)


def test_load_config_returns_validated_object() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.embedding_model
    assert cfg.streamlit.host in {"127.0.0.1", "localhost", "::1"}


def test_streamlit_host_must_be_loopback() -> None:
    with pytest.raises(ValueError):
        StreamlitConfig(host="0.0.0.0")
    with pytest.raises(ValueError):
        StreamlitConfig(host="192.168.1.1")


def test_vendor_lookup_disabled_by_default() -> None:
    cfg = load_config()
    assert cfg.vendor_lookup.enabled is False


def test_user_config_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_cfg = tmp_path / "config.yaml"
    user_cfg.write_text(
        "embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n"
        "classifier:\n"
        "  confidence_threshold: 0.9\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXPENSE_ANALYZER_CONFIG", str(user_cfg))
    cfg = load_config()
    assert cfg.embedding_model.endswith("MiniLM-L12-v2")
    assert cfg.classifier.confidence_threshold == 0.9
    # Untouched values keep defaults.
    assert cfg.knn.k == 5


def test_data_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPENSE_ANALYZER_HOME", str(tmp_path / "elsewhere"))
    cfg = load_config()
    assert cfg.data_dir == (tmp_path / "elsewhere")
    assert cfg.db_path == cfg.data_dir / cfg.db_filename
