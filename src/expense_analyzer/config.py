"""Configuration loading. Reads YAML, validates with pydantic."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ClassifierConfig(BaseModel):
    type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    rf_switch_threshold: int = 200
    confidence_threshold: float = 0.7
    retrain_after_n_new_labels: int = 10


class VendorExactMatchConfig(BaseModel):
    enabled: bool = True
    agreement_min: float = 0.8


class KnnConfig(BaseModel):
    enabled: bool = True
    k: int = 5
    agreement_min: int = 4


class ZeroshotConfig(BaseModel):
    enabled: bool = True
    use_when_confidence_below: float = 0.5


class ClusteringConfig(BaseModel):
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    hdbscan_min_cluster_size: int = 5


class ActiveLearningConfig(BaseModel):
    default_batch_size: int = 10
    default_strategy: Literal["uncertainty", "diverse", "outliers", "mixed"] = "uncertainty"


class VendorLookupConfig(BaseModel):
    enabled: bool = False
    backend: Literal["duckduckgo", "searxng"] = "duckduckgo"
    searxng_url: str = ""
    cache_ttl_days: int = 90


class StreamlitConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501
    headless: bool = True

    @field_validator("host")
    @classmethod
    def _enforce_loopback(cls, v: str) -> str:
        # Privacy invariant: never bind Streamlit to a non-loopback address.
        if v not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                f"streamlit.host must be a loopback address (127.0.0.1/localhost/::1), got {v!r}"
            )
        return v


class Config(BaseModel):
    data_dir: Path = Path("~/.expense-analyzer").expanduser()
    db_filename: str = "db.sqlite"

    embedding_model: str = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    zeroshot_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    embedding_batch_size: int = 32
    device: Literal["auto", "cpu", "cuda"] = "auto"

    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    vendor_exact_match: VendorExactMatchConfig = Field(default_factory=VendorExactMatchConfig)
    knn: KnnConfig = Field(default_factory=KnnConfig)
    zeroshot: ZeroshotConfig = Field(default_factory=ZeroshotConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    vendor_lookup: VendorLookupConfig = Field(default_factory=VendorLookupConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: Path | str) -> Path:
        return Path(str(v)).expanduser()

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def packaged_default_config() -> dict:
    return _load_yaml_resource("default_config.yaml")


def packaged_default_categories() -> list[dict]:
    data = _load_yaml_resource("default_categories.yaml")
    return data.get("categories", [])


def _load_yaml_resource(name: str) -> dict:
    """Read a packaged YAML file from the `config/` directory at repo root.

    We look for it relative to the installed package; in source checkouts that's
    ``<repo>/config/<name>``.
    """
    # Walk up from this file: src/expense_analyzer/config.py -> repo/config/<name>
    repo_config = Path(__file__).resolve().parent.parent.parent / "config" / name
    if repo_config.is_file():
        return _load_yaml(repo_config)
    # Fallback: package data (if shipped via setuptools package_data).
    try:
        text = resources.files("expense_analyzer").joinpath(f"../../config/{name}").read_text(
            encoding="utf-8"
        )
        return yaml.safe_load(text) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def resolve_config_path(explicit: Path | None = None) -> Path | None:
    """Find a user config file. Order: --config, $EXPENSE_ANALYZER_CONFIG,
    <data_dir>/config.yaml. Returns None if no user file exists."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env = os.environ.get("EXPENSE_ANALYZER_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    home = Path(os.environ.get("EXPENSE_ANALYZER_HOME", "~/.expense-analyzer")).expanduser()
    candidate = home / "config.yaml"
    return candidate if candidate.is_file() else None


def load_config(explicit: Path | None = None) -> Config:
    """Merge: packaged defaults <- user config (if any). Pydantic validates."""
    merged = packaged_default_config()
    user_path = resolve_config_path(explicit)
    if user_path is not None:
        user = _load_yaml(user_path)
        merged = _deep_merge(merged, user)
    # data_dir env override
    env_home = os.environ.get("EXPENSE_ANALYZER_HOME")
    if env_home:
        merged["data_dir"] = env_home
    return Config(**merged)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_user_config(updates: dict, data_dir: Path | None = None) -> Path:
    """Merge `updates` into the user config file at <data_dir>/config.yaml.

    Used by the Settings UI to persist choices (e.g. selected model) across
    sessions. Only the keys in `updates` are touched; everything else in the
    existing user config stays put.
    """
    base = (
        Path(data_dir)
        if data_dir is not None
        else Path(os.environ.get("EXPENSE_ANALYZER_HOME", "~/.expense-analyzer")).expanduser()
    )
    base.mkdir(parents=True, exist_ok=True)
    path = base / "config.yaml"
    existing = _load_yaml(path) if path.is_file() else {}
    merged = _deep_merge(existing, updates)
    path.write_text(
        yaml.safe_dump(merged, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
