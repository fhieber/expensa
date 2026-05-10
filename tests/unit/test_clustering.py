"""Clustering tests using HashEmbedder. Hash embeddings give unique
vectors per text, so HDBSCAN may classify most rows as outliers — but we
care that the function runs end-to-end and writes cluster_id back."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import HashEmbedder
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.clustering import cluster_all


def test_cluster_all_runs_and_persists(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cfg = Config(data_dir=tmp_path)
    cfg.clustering.hdbscan_min_cluster_size = 2
    report = cluster_all(tmp_db, cfg, HashEmbedder(dim=64))
    assert report.n_points == 50
    # Every row got a cluster_id assigned (could be -1).
    n = tmp_db.execute("SELECT COUNT(*) AS n FROM expenses WHERE cluster_id IS NOT NULL").fetchone()["n"]
    assert n == 50


def test_cluster_empty_db(tmp_db: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    report = cluster_all(tmp_db, cfg, HashEmbedder(dim=64))
    assert report.n_points == 0
