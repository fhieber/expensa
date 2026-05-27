"""Curated list of locally-runnable Hugging Face models the UI can offer.

Each entry includes enough metadata for the Settings table:
  * model_id  -- the HF hub repo id, used by sentence-transformers / transformers
  * role      -- "embedding" or "zeroshot"
  * dim       -- output dimensionality (None for NLI models)
  * languages -- short tag, e.g. "de", "de+en", "multi"
  * approx_size_mb -- rough on-disk footprint after download
  * notes     -- one-line description shown in the table
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    role: str  # "embedding" | "zeroshot"
    dim: int | None
    languages: str
    approx_size_mb: int
    notes: str


EMBEDDING_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        role="embedding",
        dim=768,
        languages="de+en",
        approx_size_mb=1100,
        notes="Balanced quality/speed; current default.",
    ),
    ModelInfo(
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        role="embedding",
        dim=384,
        languages="multi",
        approx_size_mb=480,
        notes="Smallest, fastest; lower quality but solid for short text.",
    ),
    ModelInfo(
        model_id="aari1995/German_Semantic_V3",
        role="embedding",
        dim=1024,
        languages="de",
        approx_size_mb=2000,
        notes="German-specialized, 8K context, knowledge-rich. Best on GPU.",
    ),
    ModelInfo(
        model_id="intfloat/multilingual-e5-large",
        role="embedding",
        dim=1024,
        languages="multi",
        approx_size_mb=2200,
        notes="Large multilingual; strong general-purpose embeddings.",
    ),
]


ZEROSHOT_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        role="zeroshot",
        dim=None,
        languages="multi",
        approx_size_mb=580,
        notes="Multilingual NLI for zero-shot classification; current default.",
    ),
    ModelInfo(
        model_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        role="zeroshot",
        dim=None,
        languages="multi",
        approx_size_mb=580,
        notes="Same family, trained on a larger multilingual NLI mix.",
    ),
    # --- v2 family: Moritz Laurer's 2024 retraining with cleaner data ---
    ModelInfo(
        model_id="MoritzLaurer/bge-m3-zeroshot-v2.0",
        role="zeroshot",
        dim=None,
        languages="multi",
        approx_size_mb=2270,
        notes=(
            "BGE-M3 fine-tuned for zero-shot. Often the strongest "
            "multilingual zero-shot model on the HF leaderboard; "
            "single-pass scoring (faster than NLI for many labels). "
            "Heavier disk footprint."
        ),
    ),
    ModelInfo(
        model_id="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        role="zeroshot",
        dim=None,
        languages="en",
        approx_size_mb=850,
        notes=(
            "Best-in-class for English zero-shot. German support is "
            "weaker than the multilingual options here — pick only "
            "if your data is mostly English."
        ),
    ),
    ModelInfo(
        model_id="MoritzLaurer/ernie-m-large-mnli-xnli",
        role="zeroshot",
        dim=None,
        languages="multi",
        approx_size_mb=2160,
        notes=(
            "Baidu's ERNIE-M (large) on XNLI — strong on low-resource "
            "language transfer, including German bank text."
        ),
    ),
]


ALL_MODELS: list[ModelInfo] = EMBEDDING_MODELS + ZEROSHOT_MODELS


def is_downloaded(model_id: str) -> tuple[bool, float]:
    """Return (present, size_gb) for `model_id` in the local HF cache.

    Uses huggingface_hub.scan_cache_dir() so it picks up wherever the user
    has HF_HOME / HF_HUB_CACHE pointed.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False, 0.0
    try:
        info = scan_cache_dir()
    except Exception:
        return False, 0.0
    for repo in info.repos:
        if repo.repo_id == model_id:
            return True, repo.size_on_disk / 1e9
    return False, 0.0


def trigger_download(model_id: str, role: str = "embedding") -> None:
    """Pull `model_id` into the local cache. Blocking call -- caller is
    responsible for showing a spinner / status.
    """
    if role == "embedding":
        from sentence_transformers import SentenceTransformer

        # Constructing the model triggers the download.
        SentenceTransformer(model_id)
    elif role == "zeroshot":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        raise ValueError(f"unknown role {role!r}")


def hf_cache_dir() -> Path:
    """Return the configured HF cache root (or its default location)."""
    import os

    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser()
    return Path("~/.cache/huggingface/hub").expanduser()
