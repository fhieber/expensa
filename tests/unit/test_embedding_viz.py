"""Tests for the category embedding viz module + its disk cache."""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np

from expensa.features.embeddings import HashEmbedder, store_embeddings
from expensa.ingestion import ingest_csv
from expensa.ml import embedding_viz_cache
from expensa.ml.embedding_viz import (
    ProjectionResult,
    project_labeled_embeddings,
)
from expensa.storage.categories import add_label, upsert_category


def _seed_labels_and_embeddings(
    conn: sqlite3.Connection, embedder: HashEmbedder
) -> tuple[int, int]:
    """Label expenses across two categories and pre-compute embeddings.
    Returns (food_cat_id, rent_cat_id)."""
    food = upsert_category(conn, "Lebensmittel")
    rent = upsert_category(conn, "Miete")
    rows = conn.execute(
        "SELECT id, counterparty_normalized, combined_text FROM expenses"
    ).fetchall()
    for r in rows:
        if r["counterparty_normalized"] in {"markt alpha", "markt beta", "markt gamma"}:
            add_label(conn, int(r["id"]), food, "user")
        elif r["counterparty_normalized"] == "vermieter":
            add_label(conn, int(r["id"]), rent, "user")
    store_embeddings(
        conn, embedder, [(r["id"], r["combined_text"]) for r in rows]
    )
    return food, rent


def test_project_returns_aligned_2d(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    embedder = HashEmbedder(dim=64)
    _seed_labels_and_embeddings(tmp_db, embedder)
    proj = project_labeled_embeddings(tmp_db, model_name=embedder.model_name, method="pca")
    assert proj is not None
    # Shape contract: xy is N×2, category_ids/expense_ids length N.
    assert proj.xy.shape[1] == 2
    assert proj.xy.shape[0] == len(proj.category_ids)
    assert proj.xy.shape[0] == len(proj.expense_ids)
    assert proj.method == "pca"
    # PCA notes always mention variance explained.
    assert "variance" in proj.notes.lower()


def test_project_returns_none_with_no_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """No user labels means nothing to colour by -> nothing to render."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    embedder = HashEmbedder(dim=64)
    rows = tmp_db.execute("SELECT id, combined_text FROM expenses").fetchall()
    store_embeddings(tmp_db, embedder, [(r["id"], r["combined_text"]) for r in rows])
    assert project_labeled_embeddings(tmp_db, model_name=embedder.model_name) is None


def test_project_drops_singleton_classes(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """A category with only one labeled example contributes no
    separation signal and should be dropped."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    embedder = HashEmbedder(dim=64)
    food = upsert_category(tmp_db, "Lebensmittel")
    sport = upsert_category(tmp_db, "Sport")  # singleton
    rows = tmp_db.execute(
        "SELECT id, counterparty_normalized, combined_text FROM expenses"
    ).fetchall()
    seeded_sport = False
    for r in rows:
        if r["counterparty_normalized"] in {"markt alpha", "markt beta", "markt gamma"}:
            add_label(tmp_db, int(r["id"]), food, "user")
        elif not seeded_sport:
            add_label(tmp_db, int(r["id"]), sport, "user")
            seeded_sport = True
    store_embeddings(tmp_db, embedder, [(r["id"], r["combined_text"]) for r in rows])
    proj = project_labeled_embeddings(tmp_db, model_name=embedder.model_name)
    assert proj is not None
    assert proj.n_dropped_singletons >= 1
    # The remaining points are only from the food category.
    assert sport not in proj.category_ids
    assert all(cid == food for cid in proj.category_ids)


def test_project_handles_missing_embeddings_for_some_ids(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """If some labeled expenses don't have stored embeddings, they're
    dropped silently (rather than crashing the projection)."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    embedder = HashEmbedder(dim=64)
    food, _ = _seed_labels_and_embeddings(tmp_db, embedder)
    # Wipe half the embedding rows to simulate a partial backfill.
    n_before = tmp_db.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
    tmp_db.execute("DELETE FROM embeddings WHERE expense_id % 2 = 0")
    n_after = tmp_db.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
    assert n_after < n_before  # deletion was effective
    proj = project_labeled_embeddings(tmp_db, model_name=embedder.model_name)
    assert proj is not None
    # Every returned id must have a stored embedding.
    remaining = {
        int(r["expense_id"])
        for r in tmp_db.execute("SELECT expense_id FROM embeddings").fetchall()
    }
    assert all(eid in remaining for eid in proj.expense_ids)


# --- cache --------------------------------------------------------------


def _fake_projection() -> ProjectionResult:
    return ProjectionResult(
        xy=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        category_ids=[1, 2],
        expense_ids=[10, 11],
        method="pca",
        n_categories=2,
        n_dropped_singletons=0,
        notes="PCA explains 80% of variance.",
    )


def test_cache_roundtrip(tmp_path: Path) -> None:
    saved = embedding_viz_cache.save(
        tmp_path, _fake_projection(), {"method": "pca", "model_name": "hash"}
    )
    assert saved.parent.name == "cache"
    loaded = embedding_viz_cache.load(tmp_path)
    assert loaded is not None
    assert np.array_equal(loaded.projection.xy, _fake_projection().xy)
    assert loaded.meta["method"] == "pca"


def test_cache_load_returns_none_on_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "cache" / "embedding_viz_latest.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a pickle")
    assert embedding_viz_cache.load(tmp_path) is None


def test_cache_load_returns_none_on_schema_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "cache" / "embedding_viz_latest.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump({"schema_version": 999, "saved_at": "x"}, fh)
    assert embedding_viz_cache.load(tmp_path) is None


def test_cache_atomic_overwrite_leaves_no_tmp(tmp_path: Path) -> None:
    """A second save replaces the first; the .tmp staging file must not linger."""
    embedding_viz_cache.save(tmp_path, _fake_projection(), {"method": "pca"})
    embedding_viz_cache.save(tmp_path, _fake_projection(), {"method": "tsne"})
    loaded = embedding_viz_cache.load(tmp_path)
    assert loaded is not None
    assert loaded.meta["method"] == "tsne"
    assert not (tmp_path / "cache" / "embedding_viz_latest.pkl.tmp").exists()
