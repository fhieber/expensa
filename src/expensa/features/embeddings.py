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

            if torch.cuda.is_available():
                return "cuda"
            # Apple Silicon (M1/M2/M3/...): MPS gives ~5-10x speedup over
            # CPU for sentence-transformer encode. Available on macOS 12.3+
            # with PyTorch 1.12+. Guarded with hasattr so we don't crash on
            # older torch builds where the attribute doesn't exist.
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
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
    ids: list[int] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        # Guard against a corrupt / mixed-dimension store (e.g. a vector
        # written under a different model that somehow shares this
        # model_name). Skip mismatched rows rather than letting a ragged
        # stack raise deep inside the cascade.
        if int(r["dim"]) != dim:
            continue
        vec = np.frombuffer(r["vector"], dtype=np.float32)
        if vec.shape[0] != dim:
            continue
        ids.append(int(r["expense_id"]))
        vecs.append(vec)
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.vstack(vecs).astype(np.float32, copy=False)
    return ids, matrix


def embedding_model_inventory(conn: sqlite3.Connection) -> dict[str, int]:
    """Return ``{model_name: row_count}`` over the embeddings table.

    Lets callers detect a model swap: if the configured embedding model
    isn't the one with the most stored vectors, existing rows are stale /
    only-lazily-recomputed and predictions will silently run on a subset.
    """
    rows = conn.execute(
        "SELECT model_name, COUNT(*) AS n FROM embeddings GROUP BY model_name"
    ).fetchall()
    return {str(r["model_name"]): int(r["n"]) for r in rows}


def purge_embeddings_except(conn: sqlite3.Connection, keep_model: str) -> int:
    """Delete every stored embedding whose model_name isn't ``keep_model``.

    Returns the number of rows removed. Used by ``train --force-reembed``
    to clear vectors left behind by a previous embedding model before the
    active model is recomputed from scratch.
    """
    cur = conn.execute(
        "DELETE FROM embeddings WHERE model_name <> ?", (keep_model,)
    )
    return int(cur.rowcount or 0)
