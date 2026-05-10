"""Sentence-transformer embeddings with on-disk caching.

Two implementations:
  * :class:`SentenceTransformerEmbedder` — wraps a real HF model. Lazy-loaded.
  * :class:`HashEmbedder` — deterministic dummy used by tests so they don't
    download a 1 GB model. Same interface as the real one.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class Embedder(ABC):
    """Abstract embedder. Embeddings are float32 numpy arrays."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array. May internally batch."""


class HashEmbedder(Embedder):
    """Deterministic dummy embedder for tests. ``encode`` hashes each
    input string and projects it into the target dimension."""

    def __init__(self, dim: int = 384, model_name: str = "hash-test-embedder") -> None:
        self._dim = dim
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256((t or "").encode("utf-8")).digest()
            # Tile the 32-byte digest to fill `dim`, normalize to unit length.
            raw = np.frombuffer(
                (digest * ((self._dim // 32) + 1))[: self._dim], dtype=np.uint8
            ).astype(np.float32)
            raw -= 127.5
            n = np.linalg.norm(raw)
            if n > 0:
                raw /= n
            out[i] = raw
        return out


class SentenceTransformerEmbedder(Embedder):
    """Wraps `sentence_transformers.SentenceTransformer`. Lazy-loaded so
    that just *importing* this module doesn't pull torch into memory."""

    def __init__(
        self,
        model_name: str = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        device: str = "auto",
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device_pref = device
        self._verbose = verbose
        self._model = None
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _resolve_device(self) -> str:
        if self._device_pref != "auto":
            return self._device_pref
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        self._model = SentenceTransformer(self._model_name, device=self._resolve_device())
        # Dim isn't always exposed; encode a test string to discover it.
        v = self._model.encode(["test"], show_progress_bar=False)
        self._dim = int(v.shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        assert self._model is not None
        # Show tqdm only when there's enough work to be worth a bar.
        show_bar = self._verbose and len(texts) >= max(self._batch_size, 16)
        v = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return v.astype(np.float32, copy=False)


# ---- Persistence helpers ---------------------------------------------------


def store_embeddings(
    conn: sqlite3.Connection,
    embedder: Embedder,
    rows: Iterable[tuple[int, str]],
) -> int:
    """Encode and persist embeddings for the given (expense_id, text) pairs.

    Skips ids that already have an embedding from the same model.
    Returns the number of new rows written.
    """
    rows = list(rows)
    if not rows:
        return 0
    placeholder = ",".join("?" * len(rows))
    existing = conn.execute(
        f"SELECT expense_id FROM embeddings WHERE model_name = ? AND expense_id IN ({placeholder})",
        (embedder.model_name, *[r[0] for r in rows]),
    ).fetchall()
    skip = {r["expense_id"] for r in existing}
    pending = [(eid, txt) for eid, txt in rows if eid not in skip]
    if not pending:
        return 0
    vectors = embedder.encode([t for _, t in pending])
    payloads = [
        (eid, embedder.model_name, embedder.dim, vec.tobytes())
        for (eid, _), vec in zip(pending, vectors, strict=True)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(expense_id, model_name, dim, vector) VALUES (?, ?, ?, ?)",
        payloads,
    )
    return len(payloads)


def load_embeddings(
    conn: sqlite3.Connection, model_name: str, expense_ids: list[int] | None = None
) -> tuple[list[int], np.ndarray]:
    """Load (ids, matrix) for a given model. If `expense_ids` is provided
    the result is filtered to those ids."""
    if expense_ids is not None:
        if not expense_ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        placeholder = ",".join("?" * len(expense_ids))
        rows = conn.execute(
            f"SELECT expense_id, dim, vector FROM embeddings "
            f"WHERE model_name = ? AND expense_id IN ({placeholder})",
            (model_name, *expense_ids),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT expense_id, dim, vector FROM embeddings WHERE model_name = ?",
            (model_name,),
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = int(rows[0]["dim"])
    ids = [int(r["expense_id"]) for r in rows]
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        matrix[i] = np.frombuffer(r["vector"], dtype=np.float32)
    return ids, matrix
