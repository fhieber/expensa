"""Active-learning candidate selection.

Strategies pick the next N expenses to ask the user about. They differ in
the *kind* of information gain they target:

  * uncertainty: lowest classifier confidence
  * diverse:     max-min distance in embedding space (kicks off well)
  * mixed:       round-robin across the two above
"""

from __future__ import annotations

import sqlite3
from typing import Literal

import numpy as np

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import Embedder, load_embeddings
from expense_analyzer.ml.classifier import CategorizationCascade

Strategy = Literal["uncertainty", "diverse", "mixed"]


def _unlabeled_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT id FROM expenses
        WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'user')
        ORDER BY id
        """
    ).fetchall()
    return [int(r["id"]) for r in rows]


def select_uncertain(
    conn: sqlite3.Connection,
    cascade: CategorizationCascade,
    n: int,
) -> list[int]:
    candidates = _unlabeled_ids(conn)
    if not candidates:
        return []
    preds = cascade.predict_batch(candidates)
    # Score = confidence; we want the lowest confidences.
    preds.sort(key=lambda p: p.confidence)
    return [p.expense_id for p in preds[:n]]


def select_diverse(
    conn: sqlite3.Connection, embedder: Embedder, n: int
) -> list[int]:
    """Greedy max-min diversity sample over unlabeled embeddings."""
    candidates = _unlabeled_ids(conn)
    if not candidates:
        return []
    ids, vecs = load_embeddings(conn, embedder.model_name, candidates)
    if not ids:
        return candidates[:n]
    # Seed with the row whose vector is farthest from the global mean.
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    chosen_pos = [int(np.argmax(dists))]
    while len(chosen_pos) < min(n, len(ids)):
        chosen_vecs = vecs[chosen_pos]
        # Min distance from each candidate to any chosen point.
        dmat = np.linalg.norm(vecs[:, None, :] - chosen_vecs[None, :, :], axis=2)
        min_d = dmat.min(axis=1)
        # Force already-chosen points to -inf so they don't get picked again.
        min_d[chosen_pos] = -np.inf
        chosen_pos.append(int(np.argmax(min_d)))
    return [ids[p] for p in chosen_pos]


def pick_candidates(
    conn: sqlite3.Connection,
    config: Config,
    embedder: Embedder,
    cascade: CategorizationCascade,
    n: int | None = None,
    strategy: Strategy | None = None,
) -> list[int]:
    n = n or config.active_learning.default_batch_size
    strategy = strategy or config.active_learning.default_strategy
    if strategy == "uncertainty":
        return select_uncertain(conn, cascade, n)
    if strategy == "diverse":
        return select_diverse(conn, embedder, n)
    if strategy == "mixed":
        per = max(1, n // 2)
        out: list[int] = []
        seen: set[int] = set()
        for src in (
            select_uncertain(conn, cascade, per),
            select_diverse(conn, embedder, per),
        ):
            for x in src:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
                if len(out) >= n:
                    return out
        return out[:n]
    raise ValueError(f"unknown strategy {strategy!r}")
