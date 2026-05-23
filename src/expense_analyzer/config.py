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


class CategorySimilarityConfig(BaseModel):
    """Zero-shot via embedding similarity between expense and category text.

    Embeds each category as ``"name: description"`` using the same
    sentence-transformer as expenses. Cosine similarity picks the top
    category. Faster and (in practice on German bank text) more accurate
    than the NLI-based zero-shot.
    """

    enabled: bool = True
    min_top1: float = 0.25        # require top-1 cosine >= this
    min_margin: float = 0.03      # require top1 - top2 >= this
    # When True, append the vendor_cache.industry tag (e.g. "supermarket"
    # for REWE) to the expense text before lexical scoring. Helps when
    # the industry tag matches a token in the target category's description.
    use_vendor_industry: bool = True


class ActiveLearningConfig(BaseModel):
    default_batch_size: int = 10
    default_strategy: Literal[
        "uncertainty", "low-confidence-first", "diverse", "mixed",
    ] = "uncertainty"


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


class GlobalConfig(BaseModel):
    """Settings that apply to every account.

    Identical field set to :class:`Config` minus ``data_dir`` /
    ``db_filename``. The two distinct types let the type checker (and a
    quick code reader) tell shared-across-accounts state apart from
    per-account state.
    """

    embedding_model: str = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    zeroshot_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    embedding_batch_size: int = 32
    # "auto" picks the best available: CUDA on NVIDIA, MPS on Apple
    # Silicon, otherwise CPU. Override explicitly per host if needed.
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    vendor_exact_match: VendorExactMatchConfig = Field(default_factory=VendorExactMatchConfig)
    knn: KnnConfig = Field(default_factory=KnnConfig)
    zeroshot: ZeroshotConfig = Field(default_factory=ZeroshotConfig)
    category_similarity: CategorySimilarityConfig = Field(default_factory=CategorySimilarityConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    vendor_lookup: VendorLookupConfig = Field(default_factory=VendorLookupConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)


class Config(GlobalConfig):
    """Per-account resolved config = GlobalConfig + a data directory.

    Inheritance keeps :class:`Config` source-compatible with every
    pre-multi-account caller: a single resolved object with both ML
    settings and the data directory.
    """

    data_dir: Path = Path("~/.expense-analyzer").expanduser()
    db_filename: str = "db.sqlite"

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


def _merged_global_dict(explicit: Path | None = None) -> dict:
    """Read packaged defaults + the user's global config.yaml and merge.

    Returns a raw dict (not validated yet) holding ML / device /
    vendor_lookup / streamlit keys. Used by both ``load_global_config``
    and the legacy ``load_config`` path.
    """
    merged = packaged_default_config()
    user_path = resolve_config_path(explicit)
    if user_path is not None:
        user = _load_yaml(user_path)
        merged = _deep_merge(merged, user)
    return merged


def load_global_config(explicit: Path | None = None) -> GlobalConfig:
    """Read ``config.yaml`` and validate as :class:`GlobalConfig`.

    Per-account fields (``data_dir`` / ``db_filename``) are silently
    dropped if present in the user file -- they're per-account now and
    belong in the account registry, not the global config.
    """
    raw = _merged_global_dict(explicit)
    # Drop per-account keys if they survived from the legacy single-
    # account config.yaml; GlobalConfig doesn't accept them.
    raw.pop("data_dir", None)
    raw.pop("db_filename", None)
    return GlobalConfig(**raw)


def load_config_for_account(account, global_cfg: GlobalConfig | None = None) -> Config:
    """Build a per-account :class:`Config` from a registry entry.

    ``account`` is duck-typed: anything with ``.data_dir`` works (the
    real type is :class:`expense_analyzer.accounts.AccountInfo` but
    that's a heavier import we keep out of this module).

    If ``global_cfg`` is None we load it fresh from disk; pass an
    existing instance to avoid re-reading the YAML in tight loops.
    """
    if global_cfg is None:
        global_cfg = load_global_config()
    payload = global_cfg.model_dump()
    payload["data_dir"] = Path(account.data_dir)
    return Config(**payload)


def load_config(explicit: Path | None = None) -> Config:
    """Backward-compatible single-account loader.

    Resolves the active account via the registry under ``$EXPENSE_ANALYZER_HOME``
    (or ``~/.expense-analyzer``), defaulting to the first registered account
    if no active slug is set. Falls back to the legacy single-account
    layout (``data_dir = global_home``) when the registry is empty -- so
    a never-opened install or a test environment with neither
    ``accounts.yaml`` nor a legacy ``db.sqlite`` still gets a sensible
    Config back rather than crashing.
    """
    # Local import to avoid a circular dependency: accounts.py imports
    # storage / config helpers lazily.
    from expense_analyzer.accounts import migrate_legacy_if_needed

    global_cfg = load_global_config(explicit)
    global_home = Path(
        os.environ.get("EXPENSE_ANALYZER_HOME", "~/.expense-analyzer")
    ).expanduser()
    registry = migrate_legacy_if_needed(global_home)

    active_id = registry.get_active_id()
    account = None
    if active_id is not None:
        account = registry.get(active_id)
    if account is None and len(registry) > 0:
        account = registry.all()[0]
    if account is not None:
        return load_config_for_account(account, global_cfg)

    # No accounts and no legacy DB: fall back to the legacy single-
    # account layout. Tests rely on this when EXPENSE_ANALYZER_HOME
    # points at a fresh tmp_path.
    payload = global_cfg.model_dump()
    payload["data_dir"] = global_home
    return Config(**payload)


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
