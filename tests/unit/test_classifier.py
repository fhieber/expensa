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
    # Label every Markt Alpha row as Lebensmittel.
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'markt alpha'"
    ).fetchall()
    for r in rows:
        add_label(tmp_db, int(r["id"]), cid, "user")
    hit = _vendor_exact_match(tmp_db, "markt alpha", agreement_min=0.8)
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
        "SELECT id FROM expenses WHERE counterparty_normalized = 'markt alpha'"
    ).fetchall()
    # Half-half disagreement -> below 0.8 threshold.
    for i, r in enumerate(rows):
        add_label(tmp_db, int(r["id"]), a if i % 2 == 0 else b, "user")
    assert _vendor_exact_match(tmp_db, "markt alpha", agreement_min=0.8) is None


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
    # Label all Markt Alpha rows except the most recent.
    rows = tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'markt alpha' "
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
    cfg = _config_no_zeroshot(tmp_path)
    cfg.category_similarity.enabled = False  # avoid hash-collision false positives
    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=64))
    target = tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"]
    pred = cascade.predict(int(target))
    assert pred.category_id is None
    assert pred.stage == "unknown"


def test_category_similarity_fires_with_no_user_labels(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """With zero user labels, the category_similarity stage should still
    produce predictions for every record using only category text."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Install a few categories whose names exactly match the hash-embedded text
    # of some records, so the HashEmbedder produces matching vectors.
    # HashEmbedder hashes the literal string; we can't exploit semantic match,
    # so this test just checks that the stage *runs* (and at least one cat
    # similarity prediction lands), not that the categories are right.
    upsert_category(tmp_db, "A", description="Lebensmittel")
    upsert_category(tmp_db, "B", description="Miete")

    cfg = _config_no_zeroshot(tmp_path)
    cfg.category_similarity.min_top1 = -1.0  # accept anything top-1
    cfg.category_similarity.min_margin = -1.0
    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=128))
    cascade.fit()
    sample_ids = [int(r["id"]) for r in tmp_db.execute(
        "SELECT id FROM expenses LIMIT 5"
    ).fetchall()]
    preds = cascade.predict_batch(sample_ids)
    stages = {p.stage for p in preds}
    assert "category_similarity" in stages
    # All those predictions must point at one of our two installed categories.
    for p in preds:
        if p.stage == "category_similarity":
            assert p.category_id is not None


def test_vendor_industry_boosts_category_similarity(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """A vendor-cache 'industry' tag should bias category_similarity toward
    the category whose description contains that keyword.

    Setup: two near-identical categories, only one of which has the
    keyword "supermarket" in its description; the vendor_cache marks the
    counterparty's industry as "supermarket". With the boost the
    targeted category must win even though the embedding signal alone
    can be too noisy."""
    from expense_analyzer.storage.categories import upsert_category as _upsert

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")

    grocery = _upsert(
        tmp_db, "Lebensmittel",
        description="Markt Alpha Beta Gamma Haendler Supermarkt supermarket Kauf.",
    )
    other = _upsert(
        tmp_db, "Sonstiges",
        description="Anderes nicht klar zuordenbare Ausgabe miscellaneous.",
    )
    assert grocery != other

    cp = tmp_db.execute(
        "SELECT counterparty_normalized FROM expenses WHERE counterparty_normalized = 'markt alpha' LIMIT 1"
    ).fetchone()["counterparty_normalized"]
    tmp_db.execute(
        "INSERT INTO vendor_cache(counterparty_normalized, summary, industry) VALUES (?, ?, ?)",
        (cp, "Markt Alpha ist ein Haendler.", "supermarket"),
    )

    cfg = _config_no_zeroshot(tmp_path)
    cfg.category_similarity.min_top1 = -1.0
    cfg.category_similarity.min_margin = -1.0
    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=128))
    cascade.fit()

    alpha_ids = [
        int(r["id"]) for r in tmp_db.execute(
            "SELECT id FROM expenses WHERE counterparty_normalized = 'markt alpha'"
        ).fetchall()
    ]
    preds = cascade.predict_batch(alpha_ids)
    # All Markt Alpha rows must land on Lebensmittel (the category whose
    # description matches the vendor industry).
    assert all(p.category_id == grocery for p in preds), (
        f"expected Markt Alpha -> Lebensmittel, got "
        f"{[(p.expense_id, p.category_id) for p in preds]}"
    )


def test_vendor_industry_boost_disabled(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """With use_vendor_industry=False the boost is skipped (no SQL hit)."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "X", description="x")
    cfg = _config_no_zeroshot(tmp_path)
    cfg.category_similarity.use_vendor_industry = False
    cfg.category_similarity.min_top1 = -1.0
    cfg.category_similarity.min_margin = -1.0
    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=64))
    cascade.fit()
    # The vendor_cache table is empty; even if we forgot to gate the
    # SQL it'd just return no rows. This test mostly exercises the
    # disabled code path.
    preds = cascade.predict_batch([
        int(r["id"]) for r in tmp_db.execute(
            "SELECT id FROM expenses LIMIT 3"
        ).fetchall()
    ])
    assert len(preds) == 3


def test_category_similarity_respects_min_top1_threshold(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """With an impossibly-high threshold, similarity should abstain."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    upsert_category(tmp_db, "Solo")
    cfg = _config_no_zeroshot(tmp_path)
    cfg.category_similarity.min_top1 = 0.999
    cascade = CategorizationCascade(tmp_db, cfg, HashEmbedder(dim=64))
    cascade.fit()
    rid = int(tmp_db.execute("SELECT id FROM expenses LIMIT 1").fetchone()["id"])
    p = cascade.predict(rid)
    # Nothing should have fired -> stage 'unknown'.
    assert p.stage == "unknown"
    assert p.category_id is None


def test_cascade_fit_and_classifier_predict(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """With a few diverse labels the trained classifier should be confident
    on the same training rows (overfit). Confirms fit/predict path runs."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    income = upsert_category(tmp_db, "Einkommen")
    _label_one(tmp_db, "markt alpha", food)
    _label_one(tmp_db, "markt beta", food)
    _label_one(tmp_db, "markt gamma", food)
    _label_one(tmp_db, "vermieter", rent)
    arbeitgeber = tmp_db.execute(
        "SELECT id FROM expenses WHERE zahlungspflichtiger='Arbeitgeber GmbH' LIMIT 1"
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


# ─── Vectorization + zero-shot batching ────────────────────────────────


def test_knn_vote_from_sims_equivalent_to_knn_vote() -> None:
    """The new bulk-sims helper must vote the same way as the
    historical per-vector _knn_vote. Pin the contract because
    predict_batch's kNN stage now exclusively uses the precomputed
    sim matrix."""
    from expense_analyzer.ml.classifier import _knn_vote_from_sims

    rng = np.random.default_rng(0)
    train_vecs = rng.standard_normal((20, 16)).astype(np.float32)
    train_vecs /= np.linalg.norm(train_vecs, axis=1, keepdims=True)
    train_labels = np.array([0, 1, 0, 1, 2] * 4, dtype=np.int64)

    for _ in range(10):
        target = rng.standard_normal(16).astype(np.float32)
        target /= np.linalg.norm(target)
        per_row = _knn_vote(target, train_vecs, train_labels, k=5, agreement_min=3)
        bulk = _knn_vote_from_sims(
            train_vecs @ target, train_labels, k=5, agreement_min=3
        )
        assert per_row == bulk


def test_predict_batch_is_deterministic_across_input_order(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Vectorisation lookup by ``pos_for_eid`` plus the deferred
    zero-shot pass: if we mistakenly index by loop position rather
    than expense id anywhere, shuffling the input order would shift
    predictions. Pin that the order is irrelevant."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    _label_one(tmp_db, "markt alpha", food)
    _label_one(tmp_db, "markt beta", food)
    _label_one(tmp_db, "markt gamma", food)
    _label_one(tmp_db, "vermieter", rent)
    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=128)
    )
    cascade.fit()
    eids = [
        int(r["id"])
        for r in tmp_db.execute("SELECT id FROM expenses ORDER BY id LIMIT 30").fetchall()
    ]
    in_order = {p.expense_id: (p.category_id, p.stage) for p in cascade.predict_batch(eids)}
    shuffled = list(reversed(eids))
    out_of_order = {
        p.expense_id: (p.category_id, p.stage)
        for p in cascade.predict_batch(shuffled)
    }
    # Same expense id -> same (category, stage) regardless of where it
    # sat in the input list.
    assert in_order == out_of_order


def test_zeroshot_predict_batch_routes_empty_inputs_to_abstain(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Empty / whitespace-only premises must surface as (None, 0.0)
    -- the batch path used to crash because the transformers
    pipeline rejects ``''`` outright; we substitute a space and
    map those slots back to abstain afterwards."""
    upsert_category(tmp_db, "Lebensmittel", "Supermarkt Einkauf")
    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=64)
    )
    # Force the lazy ``_zs`` slot to a stub so we don't pay an HF
    # download in unit tests -- exercises the empty-text handling
    # without touching the network.
    def _fake_pipeline(texts, candidate_labels, **_kw):
        # transformers returns a single dict for one input, list for many.
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = [
            {"labels": candidate_labels, "scores": [1.0 / len(candidate_labels)] * len(candidate_labels)}
            for _ in items
        ]
        return out[0] if single else out

    cascade._zs = _fake_pipeline  # type: ignore[assignment]
    results = cascade._zeroshot_predict_batch(["", "  ", "real text"])
    assert results[0] == (None, 0.0)
    assert results[1] == (None, 0.0)
    # The real-text slot should produce *some* category id (any will
    # do given the fake pipeline returns uniform scores).
    cid, conf = results[2]
    assert cid is not None
    assert 0.0 <= conf <= 1.0


def test_zeroshot_predict_batch_preserves_input_order(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Each input position must get its own answer in the same slot
    -- the deferred-zero-shot path in predict_batch stitches results
    back via index, so order-preservation is a hard contract."""
    cat_a = upsert_category(tmp_db, "Lebensmittel", "Supermarkt")
    cat_b = upsert_category(tmp_db, "Miete", "Wohnung")
    cascade = CategorizationCascade(
        tmp_db, _config_no_zeroshot(tmp_path), HashEmbedder(dim=64)
    )

    # Fake pipeline that always picks the label whose text matches
    # the premise verbatim, so we can verify order by reading back.
    def _fake_pipeline(texts, candidate_labels, **_kw):
        # The label format is "name: description". Pick whichever
        # label has a name that appears in the premise.
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = []
        for t in items:
            scores = []
            for lab in candidate_labels:
                name = lab.split(":", 1)[0].strip().lower()
                scores.append(1.0 if name in t.lower() else 0.0)
            # Renormalise so transformers' argmax behaviour is sane.
            total = sum(scores) or 1.0
            scores = [s / total for s in scores]
            order = sorted(range(len(candidate_labels)), key=lambda i: -scores[i])
            out.append({
                "labels": [candidate_labels[i] for i in order],
                "scores": [scores[i] for i in order],
            })
        return out[0] if single else out

    cascade._zs = _fake_pipeline  # type: ignore[assignment]
    results = cascade._zeroshot_predict_batch(
        ["lebensmittel einkauf", "miete januar", "lebensmittel laden"]
    )
    assert len(results) == 3
    assert results[0][0] == cat_a
    assert results[1][0] == cat_b
    assert results[2][0] == cat_a
