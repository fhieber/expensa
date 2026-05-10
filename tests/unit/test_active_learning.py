"""Active-learning candidate selection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import HashEmbedder
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.active_learning import (
    pick_candidates,
    select_diverse,
    select_outliers,
    select_uncertain,
)
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.ml.clustering import cluster_all
from expense_analyzer.storage.categories import add_label, upsert_category


def _cfg(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.enabled = False
    cfg.clustering.hdbscan_min_cluster_size = 2
    return cfg


def test_select_uncertain_excludes_labeled(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    labeled_ids = []
    for r in tmp_db.execute("SELECT id FROM expenses LIMIT 5").fetchall():
        add_label(tmp_db, int(r["id"]), cid, "user")
        labeled_ids.append(int(r["id"]))
    cascade = CategorizationCascade(tmp_db, _cfg(tmp_path), HashEmbedder(dim=64))
    cands = select_uncertain(tmp_db, cascade, n=10)
    assert all(eid not in labeled_ids for eid in cands)
    assert len(cands) == 10


def test_select_diverse_returns_unique(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    e = HashEmbedder(dim=64)
    # Embeddings need to exist for select_diverse to work.
    from expense_analyzer.features.embeddings import store_embeddings

    rows = tmp_db.execute("SELECT id, combined_text FROM expenses").fetchall()
    store_embeddings(tmp_db, e, [(r["id"], r["combined_text"]) for r in rows])
    cands = select_diverse(tmp_db, e, n=8)
    assert len(set(cands)) == 8


def test_select_outliers_uses_cluster_id_minus_one(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cluster_all(tmp_db, _cfg(tmp_path), HashEmbedder(dim=64))
    cands = select_outliers(tmp_db, n=5)
    # All returned should be -1 cluster.
    for eid in cands:
        cid = tmp_db.execute("SELECT cluster_id FROM expenses WHERE id=?", (eid,)).fetchone()["cluster_id"]
        assert cid == -1


def test_pick_candidates_dispatch(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    e = HashEmbedder(dim=64)
    from expense_analyzer.features.embeddings import store_embeddings

    rows = tmp_db.execute("SELECT id, combined_text FROM expenses").fetchall()
    store_embeddings(tmp_db, e, [(r["id"], r["combined_text"]) for r in rows])
    cluster_all(tmp_db, _cfg(tmp_path), e)
    cascade = CategorizationCascade(tmp_db, _cfg(tmp_path), e)
    for s in ("uncertainty", "diverse", "outliers", "mixed"):
        out = pick_candidates(tmp_db, _cfg(tmp_path), e, cascade, n=5, strategy=s)
        assert len(out) <= 5
