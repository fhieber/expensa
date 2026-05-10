"""UMAP + HDBSCAN clustering. Persists the cluster_id back to expenses."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import Embedder
from expense_analyzer.features.pipeline import build_full_features
from expense_analyzer.ml.classifier import _numeric_block


@dataclass
class ClusterReport:
    n_points: int
    n_clusters: int
    n_outliers: int


def cluster_all(
    conn: sqlite3.Connection, config: Config, embedder: Embedder
) -> ClusterReport:
    """Recompute clusters over every expense and store cluster_id back."""
    df, emb = build_full_features(conn, embedder=embedder)
    if df.empty or emb is None or emb.shape[0] == 0:
        return ClusterReport(0, 0, 0)

    # Compose feature matrix: embedding | z-scored numerics
    num = _numeric_block(df).astype(np.float32)
    if num.shape[1] > 0:
        std = num.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        num = (num - num.mean(axis=0, keepdims=True)) / std
    X = np.hstack([emb, num])

    # UMAP needs at least n_neighbors+1 points to be meaningful.
    n = len(df)
    n_components = min(config.clustering.umap_n_components, max(2, n - 2))
    n_neighbors = min(config.clustering.umap_n_neighbors, max(2, n - 1))
    if n < 4:
        # Tiny dataset: skip dim reduction, cluster on raw features.
        reduced = X
    else:
        from umap import UMAP

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=0,
        )
        reduced = reducer.fit_transform(X)

    import hdbscan

    min_cluster_size = max(2, config.clustering.hdbscan_min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=False)
    labels = clusterer.fit_predict(reduced)

    # Persist back to expenses
    payload = [(int(label), int(eid)) for eid, label in zip(df.index, labels, strict=True)]
    conn.executemany("UPDATE expenses SET cluster_id = ? WHERE id = ?", payload)
    n_clusters = int(len({lbl for lbl in labels if lbl != -1}))
    n_outliers = int((labels == -1).sum())
    return ClusterReport(n_points=len(labels), n_clusters=n_clusters, n_outliers=n_outliers)
