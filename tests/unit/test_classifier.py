"""Cascaded classifier tests using the HashEmbedder."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import HashEmbedder
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.classifier import CategorizationCascade, _knn_vote, _vendor_exact_match
from expense_analyzer.storage.categories import (
    add_label,
    upsert_category,
)


def _config_no_zeroshot(data_dir: Path) -> Config:
    """A config with the zeroshot stage disabled (so tests don't try HF download)."""
    cfg = Config(data_dir=data_dir)
    cfg.zeroshot.enabled = False
    return cfg


def _label_one(conn: sqlite3.Connection, counterparty_norm: str, category_id: int) -> int:
    """Label the first expense matching the counterparty. Returns expense_id."""
    row = conn.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = ? ORDER BY id LIMIT 1",
        (counterparty_norm,),
    ).fetchone()
    assert row is not None, f"no expense with counterparty {counterparty_norm!r}"
    add_label(conn, int(row["id"]), category_id, "user")
    return int(row["id"])


def test_vendor_exact_match_majority(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    # Label every REWE row as Lebensmittel.
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt'"
    ).fetchall()
    for r in rows:
        add_label(tmp_db, int(r["id"]), cid, "user")
    hit = _vendor_exact_match(tmp_db, "rewe markt", agreement_min=0.8)
    assert hit is not None
    assert hit[0] == cid
    assert hit[1] == 1.0


def test_vendor_exact_match_below_agreement_threshold(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    a = upsert_category(tmp_db, "Lebensmittel")
    b = upsert_category(tmp_db, "Sonstiges")
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt'"
    ).fetchall()
    # Half-half disagreement -> below 0.8 threshold.
    for i, r in enumerate(rows):
        add_label(tmp_db, int(r["id"]), a if i % 2 == 0 else b, "user")
    assert _vendor_exact_match(tmp_db, "rewe markt", agreement_min=0.8) is None


def test_knn_vote_unanimous() -> None:
    train_vecs = np.eye(5, dtype=np.float32)
    train_labels = np.array([0, 0, 0, 0, 0])
    target = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    hit = _knn_vote(target, train_vecs, train_labels, k=5, agreement_min=4)
    assert hit == (0, 1.0)


def test_knn_vote_below_threshold() -> None:
    train_vecs = np.eye(4, dtype=np.float32)
    train_labels = np.array([0, 1, 2, 3])
    target = np.array([1, 0, 0, 0], dtype=np.float32)
    # Each neighbor votes for a different class -> no agreement.
    assert _knn_vote(target, train_vecs, train_labels, k=4, agreement_min=2) is None


def test_cascade_vendor_exact_match_predicts(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    # Label all REWE rows except the most recent.
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt' "
        "ORDER BY buchungsdatum"
    ).fetchall()
    for r in rows[:-1]:
        add_label(tmp_db, int(r["id"]), cid, "user")
    target = int(rows[-1]["id"])

    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=64)
    )
    pred = cascade.predict(target)
    assert pred.category_id == cid
    assert pred.stage == "vendor_exact_match"


def test_cascade_falls_through_to_unknown_with_no_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "Lebensmittel")  # category exists but no labels
    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=64)
    )
    target = tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"]
    pred = cascade.predict(int(target))
    assert pred.category_id is None
    assert pred.stage == "unknown"


def test_cascade_fit_and_classifier_predict(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """With a few diverse labels the trained classifier should be confident
    on the same training rows (overfit). Confirms fit/predict path runs."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    income = upsert_category(tmp_db, "Einkommen")
    _label_one(tmp_db, "rewe markt", food)
    _label_one(tmp_db, "edeka sued", food)
    _label_one(tmp_db, "aldi sued", food)
    _label_one(tmp_db, "vermieter schmidt", rent)
    arbeitgeber = tmp_db.execute(
        "SELECT id FROM expenses WHERE zahlungspflichtiger='Arbeitgeber AG' LIMIT 1"
    ).fetchone()
    add_label(tmp_db, int(arbeitgeber["id"]), income, "user")

    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=128)
    )
    report = cascade.fit()
    assert report.n_train >= 5
    assert report.n_classes == 3
    assert report.classifier_type == "logistic_regression"
    assert report.train_score >= 0.99  # overfit on 5 rows
