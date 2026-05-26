"""save_user_config: persisting Settings choices to <data_dir>/config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from expense_analyzer.config import load_config, save_user_config


def test_save_creates_file_with_updates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXPENSE_ANALYZER_HOME", str(tmp_path))
    path = save_user_config({"embedding_model": "test/model"}, data_dir=tmp_path)
    assert path == tmp_path / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["embedding_model"] == "test/model"


def test_save_merges_into_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXPENSE_ANALYZER_HOME", str(tmp_path))
    save_user_config({"embedding_model": "first/model"}, data_dir=tmp_path)
    save_user_config({"zeroshot_model": "second/model"}, data_dir=tmp_path)
    data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert data["embedding_model"] == "first/model"
    assert data["zeroshot_model"] == "second/model"


def test_load_config_picks_up_saved_choice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXPENSE_ANALYZER_HOME", str(tmp_path))
    save_user_config({"embedding_model": "test/preferred-model"}, data_dir=tmp_path)
    cfg = load_config()
    assert cfg.embedding_model == "test/preferred-model"


def test_save_deep_merges_nested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXPENSE_ANALYZER_HOME", str(tmp_path))
    save_user_config({"classifier": {"confidence_threshold": 0.9}}, data_dir=tmp_path)
    save_user_config({"classifier": {"retrain_after_n_new_labels": 5}}, data_dir=tmp_path)
    cfg = load_config()
    assert cfg.classifier.confidence_threshold == 0.9
    assert cfg.classifier.retrain_after_n_new_labels == 5
