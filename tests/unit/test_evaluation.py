"""Cross-validation / ablation tests for the cascade quality evaluator.

Uses the HashEmbedder + sample fixture so no HF model is downloaded; the
zero-shot stage is disabled throughout.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import HashEmbedder
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.ml.evaluation import (
    STAGE_ORDER,
    ablation,
    cross_validate,
)
from expense_analyzer.storage.categories import (
    add_label,
    upsert_category,
    vendor_label_distribution,
)


def _cfg(data_dir: Path) -> Config:
    cfg = Config(data_dir=data_dir)
    cfg.zeroshot.enabled = False
    return cfg


def _seed_labels(conn: sqlite3.Connection) -> dict[str, int]:
    """Label rows across three categories so every class has >=2 members."""
    food = upsert_category(conn, "Lebensmittel")
    rent = upsert_category(conn, "Miete")
    income = upsert_category(conn, "Einkommen")
    rows = conn.execute(
        "SELECT id, counterparty_normalized, is_income FROM expenses"
    ).fetchall()
    for r in rows:
        if r["is_income"]:
            add_label(conn, int(r["id"]), income, "user")
        elif r["counterparty_normalized"] in {"rewe markt", "edeka sued", "aldi sued"}:
            add_label(conn, int(r["id"]), food, "user")
        elif r["counterparty_normalized"] == "vermieter schmidt":
            add_label(conn, int(r["id"]), rent, "user")
    return {"food": food, "rent": rent, "income": income}


def test_cross_validate_populates_result(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _seed_labels(tmp_db)

    result = cross_validate(
        tmp_db, _cfg(tmp_path), HashEmbedder(dim=64), n_folds=2, seed=0
    )

    assert result.n_folds == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert 0.0 <= result.weighted_f1 <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    # Accuracy-among-covered is >= overall accuracy (abstentions only hurt
    # the latter), and NaN only when nothing was predicted.
    if result.coverage > 0:
        assert 0.0 <= result.accuracy_covered <= 1.0
        assert result.accuracy_covered + 1e-9 >= result.accuracy
    assert result.per_category
    assert result.confusion.shape == (
        len(result.confusion_labels),
        len(result.confusion_labels),
    )
    # Every test row appears once in records and once across the stage
    # breakdown coverage counts.
    n_records = len(result.records)
    assert n_records > 0
    assert sum(s.n_predicted for s in result.stage_breakdown) == n_records


def test_cross_validate_too_few_labels_returns_empty(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Only one category with one label -> nothing stratifiable.
    cid = upsert_category(tmp_db, "Solo")
    rid = int(tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"])
    add_label(tmp_db, rid, cid, "user")

    result = cross_validate(
        tmp_db, _cfg(tmp_path), HashEmbedder(dim=64), n_folds=5, seed=0
    )
    assert result.n_folds == 0
    assert result.notes


def test_vendor_label_distribution_restrict_ids(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt' ORDER BY id"
    ).fetchall()
    ids = [int(r["id"]) for r in rows]
    for eid in ids:
        add_label(tmp_db, eid, cid, "user")

    full = vendor_label_distribution(tmp_db, "rewe markt")
    assert full == {cid: len(ids)}

    # Restricting to a subset reduces the count; excluding all returns {}.
    subset = set(ids[:-1])
    restricted = vendor_label_distribution(tmp_db, "rewe markt", restrict_ids=subset)
    assert restricted == {cid: len(subset)}
    assert vendor_label_distribution(tmp_db, "rewe markt", restrict_ids=set()) == {}


def test_cascade_train_ids_prevents_vendor_leak(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """A held-out row whose vendor has no other labeled examples in the
    train set must NOT be vendor-matched (no self-leak)."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    other = upsert_category(tmp_db, "Miete")
    rewe = [
        int(r["id"])
        for r in tmp_db.execute(
            "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt' ORDER BY id"
        ).fetchall()
    ]
    # Need >=2 classes so fit() doesn't bail; give the other category a row.
    rent_row = int(
        tmp_db.execute(
            "SELECT id FROM expenses WHERE counterparty_normalized = 'vermieter schmidt' LIMIT 1"
        ).fetchone()["id"]
    )
    for eid in rewe:
        add_label(tmp_db, eid, cid, "user")
    add_label(tmp_db, rent_row, other, "user")

    held_out = rewe[0]
    train_ids = set(rewe[1:]) | {rent_row}
    cascade = CategorizationCascade(
        tmp_db, _cfg(tmp_path), HashEmbedder(dim=64), train_ids=train_ids
    )
    cascade.fit()
    # Vendor match should still fire for the held-out REWE row because OTHER
    # REWE rows are in the train set -- that's legitimate, not a leak.
    pred = cascade.predict_batch([held_out])[0]
    assert pred.stage == "vendor_exact_match"
    assert pred.category_id == cid

    # But if NO REWE rows are in the train set, vendor match must not fire.
    cascade2 = CategorizationCascade(
        tmp_db, _cfg(tmp_path), HashEmbedder(dim=64), train_ids={rent_row}
    )
    cascade2.fit()
    pred2 = cascade2.predict_batch([held_out])[0]
    assert pred2.stage != "vendor_exact_match"


def test_classifier_enabled_flag_skips_stage(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _seed_labels(tmp_db)
    cfg = _cfg(tmp_path)
    cfg.classifier.enabled = False
    # Disable the cheaper stages so the classifier would otherwise be the
    # one to fire.
    cfg.vendor_exact_match.enabled = False
    cfg.knn.enabled = False
    cfg.category_similarity.enabled = False

    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=64))
    cascade.fit()
    ids = [int(r["id"]) for r in tmp_db.execute("SELECT id FROM expenses LIMIT 10").fetchall()]
    preds = cascade.predict_batch(ids)
    assert all(p.stage != "classifier" for p in preds)


def test_ablation_shapes(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _seed_labels(tmp_db)
    cfg = _cfg(tmp_path)  # zeroshot off -> 4 participating stages

    abl = ablation(tmp_db, cfg, HashEmbedder(dim=64), n_folds=2, seed=0)
    expected_stages = [s for s in STAGE_ORDER if s != "zeroshot"]
    assert len(abl.cumulative) == len(expected_stages)
    assert len(abl.leave_one_out) == len(expected_stages)
    assert 0.0 <= abl.full_accuracy <= 1.0
    # Each leave-one-out entry carries a delta vs the full run.
    for _stage, acc, _f1, delta in abl.leave_one_out:
        assert abs(delta - (acc - abl.full_accuracy)) < 1e-9
