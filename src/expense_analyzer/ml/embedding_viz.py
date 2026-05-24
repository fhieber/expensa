"""2D projection of labeled-expense embeddings, coloured by category.

Headline: how cleanly do your hand-labeled expenses separate in the
embedding space? Visible cluster overlap predicts which categories
the cascade will confuse downstream -- which is exactly what the
Quality tab's confusion matrix surfaces *after* the fact. This module
makes the same diagnosis visible *before* a costly cross-validation
run.

Pure / Streamlit-free so it can be exercised by both the Categories
tab and unit tests with the ``HashEmbedder``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

import numpy as np

from expense_analyzer.features.embeddings import load_embeddings
from expense_analyzer.storage.categories import labeled_ids_with_categories

Method = Literal["pca", "tsne"]


@dataclass
class ProjectionResult:
    """2D projection ready to plot.

    ``xy`` is N×2; ``category_ids`` and ``expense_ids`` are length-N
    aligned. ``method`` records which projector ran (so the chart can
    label its axes); ``notes`` carries any "we fell back" caveats so
    the UI can show them.
    """

    xy: np.ndarray
    category_ids: list[int]
    expense_ids: list[int]
    method: str
    n_categories: int
    n_dropped_singletons: int
    notes: str = ""


def project_labeled_embeddings(
    conn: sqlite3.Connection,
    model_name: str,
    method: Method = "pca",
    seed: int = 0,
    tsne_perplexity: float = 30.0,
) -> ProjectionResult | None:
    """Return a 2D projection of every user-labeled expense's embedding.

    Returns ``None`` when there isn't enough data to project (no
    user-labeled rows, no embeddings stored for the model). The UI
    treats None as "show an info message, no chart".

    PCA is preferred for the default because:
      * deterministic (no perplexity hyperparam to tune)
      * fast (O(n*d^2) eigendecomp instead of t-SNE's O(n^2))
      * preserves global structure -- meaningful "distance between
        category centroids" claims, which is what the user is after.

    t-SNE is offered as a non-default because it often reveals
    fine-grained sub-clusters at the cost of distorting global
    distances. Both fall back to whichever is installed if one is
    missing.
    """
    labels = labeled_ids_with_categories(conn, source="user")
    if not labels:
        return None
    expense_ids = [eid for eid, _ in labels]
    cat_ids = np.array([cid for _, cid in labels], dtype=np.int64)

    loaded_ids, vecs = load_embeddings(conn, model_name, expense_ids)
    if not loaded_ids or vecs.shape[0] == 0:
        return None

    # Re-align: load_embeddings may return a subset (rows where the
    # embedding hasn't been computed are silently dropped).
    pos_for = {eid: i for i, eid in enumerate(loaded_ids)}
    keep_pos = [pos_for[eid] for eid in expense_ids if eid in pos_for]
    keep_mask = np.array(
        [eid in pos_for for eid in expense_ids], dtype=bool
    )
    if not keep_pos:
        return None
    cat_ids = cat_ids[keep_mask]
    aligned_expense_ids = [eid for eid in expense_ids if eid in pos_for]
    X = vecs[keep_pos]

    # Drop singleton classes: a single point can't form a separation
    # signal and clutters the legend.
    uniq, counts = np.unique(cat_ids, return_counts=True)
    keep_cats = set(uniq[counts >= 2].tolist())
    keep = np.array([cid in keep_cats for cid in cat_ids], dtype=bool)
    n_dropped = int((~keep).sum())
    X = X[keep]
    cat_ids = cat_ids[keep]
    aligned_expense_ids = [
        eid for eid, k in zip(aligned_expense_ids, keep, strict=True) if k
    ]
    if X.shape[0] < 3:
        return None

    notes = ""
    xy: np.ndarray
    used_method: str = method

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            # t-SNE perplexity must be < n_samples. Clamp + warn.
            perp = min(tsne_perplexity, max(5.0, (X.shape[0] - 1) / 3.0))
            tsne = TSNE(
                n_components=2, init="pca", random_state=seed,
                perplexity=perp, learning_rate="auto",
            )
            xy = tsne.fit_transform(X.astype(np.float64))
            if perp != tsne_perplexity:
                notes = (
                    f"t-SNE perplexity clamped to {perp:.0f} "
                    f"(requested {tsne_perplexity:.0f}, "
                    f"need < n_samples = {X.shape[0]})."
                )
        except ImportError:
            method = "pca"
            notes = "scikit-learn manifold unavailable; fell back to PCA."

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=seed)
        xy = pca.fit_transform(X)
        var = float(pca.explained_variance_ratio_.sum())
        notes = (notes + " " if notes else "") + (
            f"PCA explains {var:.1%} of variance in the first 2 components."
        )
        used_method = "pca"

    return ProjectionResult(
        xy=xy.astype(np.float32),
        category_ids=cat_ids.tolist(),
        expense_ids=aligned_expense_ids,
        method=used_method,
        n_categories=len(set(cat_ids.tolist())),
        n_dropped_singletons=n_dropped,
        notes=notes.strip(),
    )


__all__ = ("Method", "ProjectionResult", "project_labeled_embeddings")
