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
    select_low_confidence,
    select_uncertain,
)
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.categories import add_label, upsert_category


def _cfg(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.enabled = False
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


def test_pick_candidates_dispatch(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    e = HashEmbedder(dim=64)
    from expense_analyzer.features.embeddings import store_embeddings

    rows = tmp_db.execute("SELECT id, combined_text FROM expenses").fetchall()
    store_embeddings(tmp_db, e, [(r["id"], r["combined_text"]) for r in rows])
    cascade = CategorizationCascade(tmp_db, _cfg(tmp_path), e)
    for s in ("uncertainty", "low-confidence-first", "diverse", "mixed"):
        out = pick_candidates(tmp_db, _cfg(tmp_path), e, cascade, n=5, strategy=s)
        assert len(out) <= 5


def test_select_low_confidence_orders_by_conf_asc(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Stored low-confidence model labels should come first, sorted
    by ascending confidence (most uncertain first)."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cat = upsert_category(tmp_db, "Lebensmittel")
    # Seed three model labels with deliberately ordered low confidences.
    exp_rows = tmp_db.execute(
        "SELECT id FROM expenses ORDER BY id LIMIT 4"
    ).fetchall()
    confidences = {
        int(exp_rows[0]["id"]): 0.10,
        int(exp_rows[1]["id"]): 0.05,
        int(exp_rows[2]["id"]): 0.30,
        int(exp_rows[3]["id"]): 0.85,   # high-conf: must NOT be picked.
    }
    for eid, conf in confidences.items():
        add_label(tmp_db, eid, cat, "model", confidence=conf)

    cascade = CategorizationCascade(tmp_db, _cfg(tmp_path), HashEmbedder(dim=64))
    picks = select_low_confidence(tmp_db, cascade, n=3)
    # First three slots are the three low-conf rows in ascending order.
    low_only_expected = sorted(
        [eid for eid, c in confidences.items() if c < 0.40],
        key=lambda eid: confidences[eid],
    )
    assert picks[:3] == low_only_expected


def test_select_low_confidence_excludes_user_labeled(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Rows the user has already labelled shouldn't reappear in the
    low-confidence queue, even if they also carry a stored model label."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cat = upsert_category(tmp_db, "Lebensmittel")
    rows = tmp_db.execute("SELECT id FROM expenses ORDER BY id LIMIT 5").fetchall()
    ids = [int(r["id"]) for r in rows]
    # Stamp all five with a low-conf model label, then user-confirm two.
    for eid in ids:
        add_label(tmp_db, eid, cat, "model", confidence=0.10)
    for eid in ids[:2]:
        add_label(tmp_db, eid, cat, "user")

    cascade = CategorizationCascade(tmp_db, _cfg(tmp_path), HashEmbedder(dim=64))
    picks = select_low_confidence(tmp_db, cascade, n=10)
    assert all(eid not in ids[:2] for eid in picks)
