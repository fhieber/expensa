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
    # Most uncertain first: lowest top-1 confidence, then smallest margin
    # to the runner-up (a 0.55-vs-0.50 call is more informative to label
    # than a 0.55-vs-0.05 one). ``runner_up_confidence`` is populated by
    # the classifier stage and is 0.0 elsewhere, so the margin tiebreak
    # gracefully degrades to plain confidence ordering for other stages.
    preds.sort(key=lambda p: (p.confidence, p.confidence - p.runner_up_confidence))
    return [p.expense_id for p in preds[:n]]


def _undercovered_categories(
    conn: sqlite3.Connection, min_per_category: int
) -> set[int]:
    """Category ids with fewer than ``min_per_category`` user labels.

    A category absent from the labels table has zero labels and so is
    under-covered by definition; we include every defined category and
    subtract the well-covered ones.
    """
    covered = {
        int(r["category_id"])
        for r in conn.execute(
            """
            SELECT category_id, COUNT(*) AS n
            FROM labels WHERE source = 'user'
            GROUP BY category_id
            HAVING n >= ?
            """,
            (min_per_category,),
        ).fetchall()
    }
    all_cats = {
        int(r["id"]) for r in conn.execute("SELECT id FROM categories").fetchall()
    }
    return all_cats - covered


def select_diverse(
    conn: sqlite3.Connection,
    embedder: Embedder,
    n: int,
    config: Config | None = None,
) -> list[int]:
    """Greedy max-min diversity sample over unlabeled embeddings.

    When ``config.active_learning.stratified_diversity`` is on (the
    default) and there are already some user labels, candidates whose
    nearest labelled neighbour belongs to a well-covered category are
    deprioritised so the sweep spends its budget on under-represented
    categories instead of returning eight diverse-but-all-grocery rows.
    Falls back to plain geometric diversity at true cold start (no labels).
    """
    candidates = _unlabeled_ids(conn)
    if not candidates:
        return []
    ids, vecs = load_embeddings(conn, embedder.model_name, candidates)
    if not ids:
        return candidates[:n]

    # Optional stratification: split candidates into a preferred pool
    # (nearest labelled neighbour is an under-covered category) and the
    # rest, then run the diversity sweep over the preferred pool first.
    pref_mask: np.ndarray | None = None
    if config is not None and config.active_learning.stratified_diversity:
        undercovered = _undercovered_categories(
            conn, config.active_learning.diversity_min_label_per_category
        )
        # Only stratify once at least one category is well-covered; before
        # that everything is under-covered and stratification is a no-op.
        all_cats = {
            int(r["id"]) for r in conn.execute("SELECT id FROM categories").fetchall()
        }
        if undercovered and undercovered != all_cats:
            nn_cat = _nearest_labelled_category(conn, embedder, ids, vecs)
            pref_mask = np.array(
                [nn_cat.get(eid) in undercovered for eid in ids], dtype=bool
            )

    def _greedy(pos_pool: list[int], want: int) -> list[int]:
        if not pos_pool or want <= 0:
            return []
        sub = vecs[pos_pool]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid, axis=1)
        chosen_local = [int(np.argmax(dists))]
        while len(chosen_local) < min(want, len(pos_pool)):
            chosen_vecs = sub[chosen_local]
            dmat = np.linalg.norm(sub[:, None, :] - chosen_vecs[None, :, :], axis=2)
            min_d = dmat.min(axis=1)
            min_d[chosen_local] = -np.inf
            chosen_local.append(int(np.argmax(min_d)))
        return [pos_pool[c] for c in chosen_local]

    if pref_mask is not None:
        pref_pos = [i for i in range(len(ids)) if pref_mask[i]]
        rest_pos = [i for i in range(len(ids)) if not pref_mask[i]]
        chosen_pos = _greedy(pref_pos, n)
        if len(chosen_pos) < n:
            chosen_pos += _greedy(rest_pos, n - len(chosen_pos))
    else:
        chosen_pos = _greedy(list(range(len(ids))), n)
    return [ids[p] for p in chosen_pos]


def _nearest_labelled_category(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cand_ids: list[int],
    cand_vecs: np.ndarray,
) -> dict[int, int]:
    """Map each candidate id to the category of its nearest user-labelled
    expense (cosine). Empty when there are no labelled embeddings."""
    rows = conn.execute(
        """
        SELECT l.expense_id, l.category_id
        FROM labels l
        JOIN (
            SELECT expense_id, MAX(id) AS max_id
            FROM labels WHERE source = 'user'
            GROUP BY expense_id
        ) m ON l.id = m.max_id
        """
    ).fetchall()
    if not rows:
        return {}
    lab_cat = {int(r["expense_id"]): int(r["category_id"]) for r in rows}
    lab_ids, lab_vecs = load_embeddings(conn, embedder.model_name, list(lab_cat))
    if not lab_ids:
        return {}
    sims = cand_vecs @ lab_vecs.T  # (n_cand, n_lab)
    best = sims.argmax(axis=1)
    return {
        cand_ids[i]: lab_cat[lab_ids[int(best[i])]] for i in range(len(cand_ids))
    }


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


def evaluate_label_batch_impact(
    conn: sqlite3.Connection,
    cfg: Config,
    embedder: Embedder,
    batch_ids: list[int],
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float | int]:
    """Measure how a freshly-labelled batch moved cross-validated accuracy.

    Closes the active-learning loop: re-runs leak-free CV twice -- once
    with the ``batch_ids`` labels masked out (the "before" state, as if the
    user hadn't labelled them yet) and once with everything (the "after"
    state) -- and reports the delta. Read-only: it temporarily restricts
    the *training* view via the cascade's ``train_ids`` whitelist rather
    than mutating any labels.

    Returns a dict with ``before_accuracy``, ``after_accuracy``,
    ``delta``, ``before_macro_f1``, ``after_macro_f1`` and
    ``n_batch`` (batch rows that were actually labelled & usable).

    NaNs come back when there isn't enough labelled data per category to
    cross-validate (mirrors :func:`cross_validate`).
    """
    # Local import: evaluation pulls sklearn, which we keep out of this
    # module's import-time cost for the lightweight selection paths.
    from expensa.ml.evaluation import cross_validate
    from expensa.storage.categories import labeled_ids_with_categories

    all_labeled = {eid for eid, _ in labeled_ids_with_categories(conn, source="user")}
    batch = {int(x) for x in batch_ids} & all_labeled
    before_ids = all_labeled - batch

    after = cross_validate(conn, cfg, embedder, n_folds=n_folds, seed=seed)
    if before_ids:
        before = cross_validate(
            conn, cfg, embedder, n_folds=n_folds, seed=seed,
            train_ids_filter=before_ids,
        )
        before_acc, before_f1 = before.accuracy, before.macro_f1
    else:
        # Everything was in the batch -> there's no "before" to compare.
        before_acc = before_f1 = float("nan")

    delta = (
        after.accuracy - before_acc
        if not (np.isnan(after.accuracy) or np.isnan(before_acc))
        else float("nan")
    )
    return {
        "before_accuracy": before_acc,
        "after_accuracy": after.accuracy,
        "delta": delta,
        "before_macro_f1": before_f1,
        "after_macro_f1": after.macro_f1,
        "n_batch": len(batch),
    }


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
        return select_diverse(conn, embedder, n, config=config)
    if strategy == "mixed":
        per = max(1, n // 2)
        out: list[int] = []
        seen: set[int] = set()
        for src in (
            select_uncertain(conn, cascade, per),
            select_diverse(conn, embedder, per, config=config),
        ):
            for x in src:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
                if len(out) >= n:
                    return out
        return out[:n]
    raise ValueError(f"unknown strategy {strategy!r}")
