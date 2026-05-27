"""Active-learning candidate selection.

Strategies pick the next N expenses to ask the user about. They differ in
the *kind* of information gain they target:

  * uncertainty:           re-predict every candidate, pick lowest confidence
  * low-confidence-first:  read STORED low-confidence model labels and pick
                           by ascending confidence -- cheaper than uncertainty
                           because no re-prediction; surfaces the rows the
                           last cascade run was already unsure about
  * diverse:               max-min distance in embedding space (cold start)
  * mixed:                 round-robin between uncertainty and diverse
"""

from __future__ import annotations

import sqlite3
from typing import Literal

import numpy as np

from expensa.config import Config
from expensa.features.embeddings import Embedder, load_embeddings
from expensa.ml.classifier import CategorizationCascade

Strategy = Literal["uncertainty", "low-confidence-first", "diverse", "mixed"]

# Mirror of `review_tab._CONF_LOW`. Kept in sync by convention; if the
# review-tab threshold ever changes, change this too.
_LOW_CONF_THRESHOLD: float = 0.40


def get_neighbor_context(
    conn: sqlite3.Connection,
    embedder: Embedder,
    expense_id: int,
    n: int = 2,
) -> list[dict]:
    """Return the n nearest user-labeled expenses by cosine similarity.

    Each result dict has keys: expense_id, buchungsdatum, counterparty,
    betrag_cents, category_name, similarity (float 0-1).
    Returns an empty list when embeddings are unavailable or no labeled
    expenses exist.
    """
    target_ids, target_vecs = load_embeddings(conn, embedder.model_name, [expense_id])
    if not target_ids:
        return []
    target_vec = target_vecs[0]

    labeled_rows = conn.execute(
        """
        SELECT DISTINCT expense_id FROM labels
        WHERE source = 'user' AND expense_id != ?
        """,
        (expense_id,),
    ).fetchall()
    labeled_ids = [int(r["expense_id"]) for r in labeled_rows]
    if not labeled_ids:
        return []

    lab_ids, lab_vecs = load_embeddings(conn, embedder.model_name, labeled_ids)
    if not lab_ids:
        return []

    sims = lab_vecs @ target_vec
    top_pos = sims.argsort()[::-1][:n]

    results = []
    for pos in top_pos:
        eid = lab_ids[int(pos)]
        sim = float(sims[int(pos)])
        row = conn.execute(
            """
            SELECT e.buchungsdatum, e.counterparty, e.betrag_cents,
                   c.name AS category_name
            FROM expenses e
            JOIN labels l ON l.expense_id = e.id
            JOIN categories c ON c.id = l.category_id
            WHERE e.id = ? AND l.source = 'user'
            ORDER BY l.id DESC LIMIT 1
            """,
            (eid,),
        ).fetchone()
        if row:
            results.append({
                "expense_id": eid,
                "buchungsdatum": str(row["buchungsdatum"])[:10],
                "counterparty": row["counterparty"],
                "betrag_cents": row["betrag_cents"],
                "category_name": row["category_name"],
                "similarity": sim,
            })
    return results


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


def select_low_confidence(
    conn: sqlite3.Connection,
    cascade: CategorizationCascade,
    n: int,
) -> list[int]:
    """Pick rows whose **stored** model label has the lowest confidence.

    Two-stage strategy so an empty / sparse low-confidence pool falls
    back gracefully:

      1. Pull every expense whose latest label is a model prediction
         with confidence below :data:`_LOW_CONF_THRESHOLD`, sorted by
         confidence ascending (most uncertain first).
      2. If the request asks for more than that, pad from
         :func:`select_uncertain` over rows that have NO stored model
         label at all -- which the cascade will fresh-predict and rank
         by confidence too.

    Compared to ``select_uncertain``, this skips re-prediction for the
    rows that already have a stored low-confidence label, which is the
    common case after a `Predict-all` pass.
    """
    rows = conn.execute(
        """
        SELECT expense_id, confidence FROM latest_label
        WHERE label_source = 'model'
          AND (confidence IS NULL OR confidence < ?)
          AND expense_id NOT IN (
            SELECT DISTINCT expense_id FROM labels WHERE source = 'user'
          )
        ORDER BY COALESCE(confidence, 0) ASC, expense_id ASC
        LIMIT ?
        """,
        (_LOW_CONF_THRESHOLD, n),
    ).fetchall()
    low_ids = [int(r["expense_id"]) for r in rows]
    if len(low_ids) >= n:
        return low_ids
    # Padding: rows with no model label at all -- delegate to the
    # uncertainty selector (which fresh-predicts via the cascade).
    seen = set(low_ids)
    extras = [x for x in select_uncertain(conn, cascade, n) if x not in seen]
    return (low_ids + extras)[:n]


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
    if strategy == "low-confidence-first":
        return select_low_confidence(conn, cascade, n)
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
