"""Configuration loading. Reads YAML, validates with pydantic."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ClassifierConfig(BaseModel):
    enabled: bool = True
    type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    rf_switch_threshold: int = 200
    confidence_threshold: float = 0.7
    retrain_after_n_new_labels: int = 10
    # Probability calibration. A RandomForest / LogReg on an imbalanced
    # label set emits over-confident scores for rare classes (0.95 for a
    # category with two examples), which ``confidence_threshold`` then
    # trusts blindly. Wrapping the estimator in
    # ``CalibratedClassifierCV`` maps the scores back to real accuracy.
    # Only kicks in once there's enough data to cross-validate the
    # calibrator (``calibrate_min_train`` labels AND every class has at
    # least ``calibrate_cv`` members), so tiny training sets keep the
    # raw, fast estimator.
    calibrate_probas: bool = True
    calibrate_min_train: int = 50
    calibrate_cv: int = 3


class SignGuardrailConfig(BaseModel):
    """Post-prediction guardrail on the income/expense sign.

    Some categories are sign-consistent in practice: salary / refunds are
    always income, groceries are always an expense. After the cascade
    picks a category we check whether the expense's sign matches what that
    category's training labels overwhelmingly show; a clear violation
    (e.g. a positive amount predicted as ``Lebensmittel``) is demoted to an
    abstention rather than served as a confident-but-wrong answer.

    Only categories with at least ``min_support`` user labels and a sign
    that holds for at least ``min_consistency`` of them are policed; the
    rest are left untouched. User labels are never second-guessed.
    """

    enabled: bool = True
    min_consistency: float = 0.95
    min_support: int = 4


class VendorExactMatchConfig(BaseModel):
    enabled: bool = True
    agreement_min: float = 0.8
    # IBAN-based merchant identity. The same merchant often files under
    # several counterparty-name variants (REWE, REWE MARKT, REWE-BONUS)
    # but a stable IBAN. When the name-based exact match abstains we fall
    # back to the label distribution for the expense's IBAN, using the
    # same ``agreement_min`` threshold. Bridges the cold-start gap for
    # variable-name vendors that name matching and kNN both miss.
    use_iban: bool = True
    # SEPA creditor-id (Gläubiger-ID) fallback, tried BEFORE the IBAN when
    # the name match abstains. The creditor id is globally unique and
    # stable across both name and IBAN changes, so it's the strongest
    # identity key for recurring direct-debit merchants.
    use_glaeubiger: bool = True


class KnnConfig(BaseModel):
    enabled: bool = True
    k: int = 5
    agreement_min: int = 4


class ZeroshotConfig(BaseModel):
    enabled: bool = True
    use_when_confidence_below: float = 0.5
    # German hypothesis template: multilingual NLI models perform far
    # better on German bank text when the hypothesis is in the same
    # language as the premise. The default English template
    # "This example is {}." forces the model to mentally code-switch
    # for every call and bleeds 5-10 accuracy points on DE data.
    hypothesis_template: str = "In diesem Text geht es um {}."
    # When True, enrich the NLI premise with the cached vendor industry
    # tag + a short slice of the cached web summary (vendor_lookup must
    # also be enabled and populated). Off by default because the signal
    # is noisy on cryptic bank text -- recommended to A/B via the
    # Quality tab before flipping on for real.
    use_vendor_context: bool = False
    # Cap on how many characters of the cached vendor summary get
    # appended to the premise. Keeps the NLI input within model
    # context and limits exposure to long marketing snippets.
    vendor_summary_max_chars: int = 240
    # Batch size passed to the transformers zero-shot pipeline. The
    # pipeline groups <batch_size> (text × candidate-label) pairs per
    # GPU forward pass. 16 is a safe CPU default; bump to 32/64 on a
    # GPU to amortise kernel-launch overhead. Only kicks in when the
    # cascade has >1 row reaching the zero-shot stage (we batch
    # them rather than calling the pipeline once per row).
    batch_size: int = 16


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
    # Lexical-overlap bonus added on top of the cosine score, per shared
    # >=4-char token between the expense text and a category's name +
    # description. ``lexical_weight`` is the per-token bonus and
    # ``lexical_max`` caps the total so the semantic (cosine) signal still
    # dominates. Tunable so users with terse, keyword-y bank text can lean
    # harder on lexical hits. Defaults reproduce the historic hard-coded
    # 0.10 / 0.30 behaviour.
    lexical_weight: float = 0.10
    lexical_max: float = 0.30
    # When True, append the vendor_cache.industry tag (e.g. "supermarket"
    # for REWE) to the expense text before lexical scoring. Helps when
    # the industry tag matches a token in the target category's description.
    use_vendor_industry: bool = True


class ActiveLearningConfig(BaseModel):
    default_batch_size: int = 10
    default_strategy: Literal[
        "uncertainty", "low-confidence-first", "diverse", "mixed",
    ] = "uncertainty"
    # Stratified diversity: when picking diverse candidates, prefer rows
    # whose nearest labelled category is under-represented in the training
    # set, so a greedy max-min sweep doesn't return eight diverse-but-all-
    # grocery rows while rent stays unlabelled. Falls back to plain
    # geometric diversity when there are no labels yet (true cold start).
    stratified_diversity: bool = True
    # A category counts as "covered" once it has at least this many user
    # labels; diversity sampling deprioritises rows that map to covered
    # categories.
    diversity_min_label_per_category: int = 5


class VendorLookupConfig(BaseModel):
    enabled: bool = False
    backend: Literal["duckduckgo", "searxng"] = "duckduckgo"
    searxng_url: str = ""
    cache_ttl_days: int = 90


class EnrichmentConfig(BaseModel):
    """Secondary-source enrichment (e.g. a PayPal activity CSV matched to
    bank Lastschrift lines). ``date_window_days`` is how far apart the
    secondary record's date and the bank booking date may be and still be
    considered the same transaction (a debit settles a few days after the
    purchase)."""

    date_window_days: int = 4


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
    sign_guardrail: SignGuardrailConfig = Field(default_factory=SignGuardrailConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    vendor_lookup: VendorLookupConfig = Field(default_factory=VendorLookupConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)


class Config(GlobalConfig):
    """Per-account resolved config = GlobalConfig + a data directory.

    Inheritance keeps :class:`Config` source-compatible with every
    pre-multi-account caller: a single resolved object with both ML
    settings and the data directory.
    """

    data_dir: Path = Path("~/.expensa").expanduser()
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
    # Walk up from this file: src/expensa/config.py -> repo/config/<name>
    repo_config = Path(__file__).resolve().parent.parent.parent / "config" / name
    if repo_config.is_file():
        return _load_yaml(repo_config)
    # Fallback: package data (if shipped via setuptools package_data).
    try:
        text = resources.files("expensa").joinpath(f"../../config/{name}").read_text(
            encoding="utf-8"
        )
        return yaml.safe_load(text) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def resolve_config_path(explicit: Path | None = None) -> Path | None:
    """Find a user config file. Order: --config, $EXPENSA_CONFIG,
    <data_dir>/config.yaml. Returns None if no user file exists."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env = os.environ.get("EXPENSA_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    home = Path(os.environ.get("EXPENSA_HOME", "~/.expensa")).expanduser()
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
    real type is :class:`expensa.accounts.AccountInfo` but
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

    Resolves the active account via the registry under ``$EXPENSA_HOME``
    (or ``~/.expensa``), defaulting to the first registered account
    if no active slug is set. Falls back to the legacy single-account
    layout (``data_dir = global_home``) when the registry is empty -- so
    a never-opened install or a test environment with neither
    ``accounts.yaml`` nor a legacy ``db.sqlite`` still gets a sensible
    Config back rather than crashing.
    """
    # Local import to avoid a circular dependency: accounts.py imports
    # storage / config helpers lazily.
    from expensa.accounts import migrate_legacy_if_needed

    global_cfg = load_global_config(explicit)
    global_home = Path(
        os.environ.get("EXPENSA_HOME", "~/.expensa")
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
    # account layout. Tests rely on this when EXPENSA_HOME
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
        else Path(os.environ.get("EXPENSA_HOME", "~/.expensa")).expanduser()
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
