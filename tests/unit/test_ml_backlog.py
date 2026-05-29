"""Tests for the ML backlog batch.

Covers, with the HashEmbedder (no model download):

  * IBAN-based merchant-identity fallback in the vendor-exact-match stage.
  * New temporal features: recurring_months_12, recurring_is_exact_amount,
    iban_count_before.
  * Classifier probability calibration gate (engages only with enough data).
  * Sign-consistency guardrail demoting sign-violating predictions.
  * kNN runner-up surfacing on Prediction.
  * Active-learning stratified diversity + the label-batch feedback loop.
  * Embedding model-swap inventory / purge helpers.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from expensa.config import Config
from expensa.features.embeddings import (
    HashEmbedder,
    embedding_model_inventory,
    load_embeddings,
    purge_embeddings_except,
    store_embeddings,
)
from expensa.features.temporal import compute_temporal_features_bulk
from expensa.ingestion import ingest_csv
from expensa.ml.classifier import (
    CategorizationCascade,
    _knn_tally_from_sims,
    _vendor_exact_match,
)
from expensa.storage.categories import (
    add_label,
    category_sign_consistency,
    iban_label_distribution,
    upsert_category,
)


def _cfg(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.enabled = False  # never download an HF model in unit tests
    return cfg


def _insert_expense(
    conn: sqlite3.Connection,
    *,
    eid: int,
    cents: int,
    cpn: str,
    iban: str,
    d: date,
) -> None:
    conn.execute(
        """
        INSERT INTO expenses(
            id, buchungsdatum, betrag_cents, iban,
            counterparty, counterparty_normalized,
            verwendungszweck_normalized, combined_text, dedup_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (eid, d.isoformat(), cents, iban, cpn, cpn, cpn, cpn, f"hash-{eid}"),
    )


# ── IBAN-based merchant identity ──────────────────────────────────────


def test_iban_label_distribution_counts_user_labels(tmp_db: sqlite3.Connection) -> None:
    cat = upsert_category(tmp_db, "Lebensmittel")
    _insert_expense(tmp_db, eid=1, cents=-1000, cpn="rewe markt", iban="DE111", d=date(2026, 1, 1))
    _insert_expense(tmp_db, eid=2, cents=-2000, cpn="rewe bonus", iban="DE111", d=date(2026, 2, 1))
    add_label(tmp_db, 1, cat, "user")
    add_label(tmp_db, 2, cat, "user")
    dist = iban_label_distribution(tmp_db, "DE111")
    assert dist == {cat: 2}
    assert iban_label_distribution(tmp_db, "") == {}


def test_vendor_exact_match_falls_back_to_iban(tmp_db: sqlite3.Connection) -> None:
    """A new name variant with no name-labels still matches via its IBAN."""
    cat = upsert_category(tmp_db, "Lebensmittel")
    # Two labelled rows under one name + IBAN.
    _insert_expense(tmp_db, eid=1, cents=-1000, cpn="rewe markt", iban="DE111", d=date(2026, 1, 1))
    _insert_expense(tmp_db, eid=2, cents=-1100, cpn="rewe markt", iban="DE111", d=date(2026, 2, 1))
    add_label(tmp_db, 1, cat, "user")
    add_label(tmp_db, 2, cat, "user")
    # A new, never-name-labelled variant sharing the IBAN.
    new_name = "rewe sagt danke"
    # Name match abstains; IBAN fallback fires.
    assert _vendor_exact_match(tmp_db, new_name, 0.8) is None
    hit = _vendor_exact_match(tmp_db, new_name, 0.8, iban="DE111")
    assert hit is not None and hit[0] == cat


# ── New temporal features ─────────────────────────────────────────────


def test_temporal_exposes_recurrence_and_iban_features(tmp_db: sqlite3.Connection) -> None:
    iban = "DE999"
    base = date(2026, 1, 15)
    # Six monthly identical charges to the same vendor + IBAN.
    for i in range(6):
        _insert_expense(
            tmp_db, eid=i + 1, cents=-999, cpn="netflix", iban=iban,
            d=base + timedelta(days=30 * i),
        )
    feats = compute_temporal_features_bulk(tmp_db)
    last = feats[6]
    assert last["recurring_months_12"] >= 3
    assert last["recurring_is_exact_amount"] == 1
    assert last["iban_count_before"] == 5  # five prior rows share the IBAN

    # A variable-amount vendor is recurring but not exact-amount.
    for i in range(4):
        _insert_expense(
            tmp_db, eid=100 + i, cents=-(1000 + i * 200), cpn="stadtwerke", iban="DE222",
            d=base + timedelta(days=30 * i),
        )
    feats = compute_temporal_features_bulk(tmp_db)
    assert feats[103]["recurring_is_exact_amount"] == 0


# ── Sign guardrail ────────────────────────────────────────────────────


def test_category_sign_consistency(tmp_db: sqlite3.Connection) -> None:
    income = upsert_category(tmp_db, "Gehalt")
    _insert_expense(tmp_db, eid=1, cents=320000, cpn="arbeitgeber", iban="DE1", d=date(2026, 1, 1))
    _insert_expense(tmp_db, eid=2, cents=325000, cpn="arbeitgeber", iban="DE1", d=date(2026, 2, 1))
    add_label(tmp_db, 1, income, "user")
    add_label(tmp_db, 2, income, "user")
    cons = category_sign_consistency(tmp_db)
    sign, consistency, support = cons[income]
    assert sign == 1 and consistency == 1.0 and support == 2


def test_sign_guardrail_demotes_violating_prediction(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A vendor-exact match that yields an income category for a clearly
    negative (expense) row is demoted to abstention by the guardrail."""
    income = upsert_category(tmp_db, "Gehalt")
    # Build a strong, sign-consistent income history for "arbeitgeber".
    for i in range(5):
        _insert_expense(
            tmp_db, eid=i + 1, cents=320000, cpn="arbeitgeber", iban="DE1",
            d=date(2026, 1 + i, 1),
        )
        add_label(tmp_db, i + 1, income, "user")
    # A NEGATIVE amount to the same vendor: vendor-exact-match would tag it
    # Gehalt, but the sign contradicts the category.
    _insert_expense(tmp_db, eid=99, cents=-5000, cpn="arbeitgeber", iban="DE1", d=date(2026, 7, 1))

    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    cascade = CategorizationCascade(tmp_db, cfg, emb)
    cascade.fit()
    pred = cascade.predict(99)
    assert pred.category_id is None
    assert pred.stage == "unknown"
    assert "sign guardrail" in pred.notes

    # With the guardrail disabled the (wrong-sign) match is served.
    cfg2 = _cfg(tmp_path)
    cfg2.sign_guardrail.enabled = False
    cascade2 = CategorizationCascade(tmp_db, cfg2, emb)
    cascade2.fit()
    assert cascade2.predict(99).category_id == income


# ── kNN runner-up ─────────────────────────────────────────────────────


def test_knn_tally_reports_runner_up() -> None:
    # 3 votes for label 0, 2 for label 1 in the top-5.
    sims = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    tally = _knn_tally_from_sims(sims, labels, k=5)
    assert tally is not None
    top, top_n, ru, ru_n = tally
    assert (top, top_n) == (0, 3)
    assert (ru, ru_n) == (1, 2)


def test_knn_prediction_carries_runner_up(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    food = upsert_category(tmp_db, "Lebensmittel")
    other = upsert_category(tmp_db, "Sonstiges")
    # Several labelled neighbours, mixed so the winner has a runner-up.
    for i in range(4):
        _insert_expense(tmp_db, eid=i + 1, cents=-500, cpn=f"v{i}", iban="DE1", d=date(2026, 1, i + 1))
    add_label(tmp_db, 1, food, "user")
    add_label(tmp_db, 2, food, "user")
    add_label(tmp_db, 3, food, "user")
    add_label(tmp_db, 4, other, "user")
    _insert_expense(tmp_db, eid=99, cents=-500, cpn="v0", iban="DE1", d=date(2026, 2, 1))

    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    cfg.vendor_exact_match.enabled = False  # force the kNN stage
    cfg.knn.agreement_min = 1               # accept whatever the vote is
    cfg.classifier.enabled = False
    cfg.category_similarity.enabled = False
    cascade = CategorizationCascade(tmp_db, cfg, emb)
    cascade.fit()
    pred = cascade.predict(99)
    assert pred.stage == "knn"
    # Runner-up is populated when the top-k spans more than one class.
    assert pred.runner_up is not None
    assert pred.runner_up_confidence > 0.0


# ── Calibration gate ──────────────────────────────────────────────────


def test_calibration_skipped_on_tiny_training_set(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Below calibrate_min_train the raw estimator is used (type has no
    '+calibrated' suffix)."""
    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    for i in range(4):
        _insert_expense(tmp_db, eid=i + 1, cents=-500, cpn=f"v{i}", iban="DE1", d=date(2026, 1, i + 1))
    add_label(tmp_db, 1, food, "user")
    add_label(tmp_db, 2, food, "user")
    add_label(tmp_db, 3, rent, "user")
    add_label(tmp_db, 4, rent, "user")
    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    report = CategorizationCascade(tmp_db, cfg, emb).fit()
    assert "+calibrated" not in report.classifier_type


def test_calibration_engages_with_enough_data(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    # 60 labels, both classes well-populated -> calibrator can CV.
    for i in range(60):
        cat = food if i % 2 == 0 else rent
        cents = -500 if i % 2 == 0 else -90000
        _insert_expense(
            tmp_db, eid=i + 1, cents=cents, cpn=f"v{i % 7}", iban="DE1",
            d=date(2026, 1, 1) + timedelta(days=i),
        )
        add_label(tmp_db, i + 1, cat, "user")
    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    cfg.classifier.calibrate_min_train = 50
    report = CategorizationCascade(tmp_db, cfg, emb).fit()
    assert "+calibrated" in report.classifier_type


# ── Embedding model-swap helpers ──────────────────────────────────────


def test_embedding_inventory_and_purge(tmp_db: sqlite3.Connection) -> None:
    _insert_expense(tmp_db, eid=1, cents=-100, cpn="a", iban="DE1", d=date(2026, 1, 1))
    _insert_expense(tmp_db, eid=2, cents=-200, cpn="b", iban="DE1", d=date(2026, 1, 2))
    old = HashEmbedder(dim=32, model_name="old-model")
    new = HashEmbedder(dim=64, model_name="new-model")
    store_embeddings(tmp_db, old, [(1, "a")])
    store_embeddings(tmp_db, new, [(2, "b")])
    inv = embedding_model_inventory(tmp_db)
    assert inv == {"old-model": 1, "new-model": 1}
    removed = purge_embeddings_except(tmp_db, "new-model")
    assert removed == 1
    assert embedding_model_inventory(tmp_db) == {"new-model": 1}


def test_load_embeddings_skips_mismatched_dim(tmp_db: sqlite3.Connection) -> None:
    """A stray wrong-dim row under the same model_name is skipped, not
    stacked into a ragged matrix."""
    _insert_expense(tmp_db, eid=1, cents=-100, cpn="a", iban="DE1", d=date(2026, 1, 1))
    _insert_expense(tmp_db, eid=2, cents=-200, cpn="b", iban="DE1", d=date(2026, 1, 2))
    emb = HashEmbedder(dim=64, model_name="m")
    store_embeddings(tmp_db, emb, [(1, "a")])
    # Hand-write a corrupt 8-float vector under the same model_name.
    tmp_db.execute(
        "INSERT INTO embeddings(expense_id, model_name, dim, vector) VALUES (?, ?, ?, ?)",
        (2, "m", 64, np.zeros(8, dtype=np.float32).tobytes()),
    )
    ids, matrix = load_embeddings(tmp_db, "m")
    assert ids == [1]
    assert matrix.shape == (1, 64)


# ── Active-learning: stratified diversity + feedback loop ─────────────


def test_stratified_diversity_prefers_undercovered(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    from expensa.ml.active_learning import select_diverse

    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    # Make "food" well-covered (>=5 labels) and "rent" under-covered.
    for i in range(6):
        _insert_expense(tmp_db, eid=i + 1, cents=-500, cpn="markt", iban="DE1", d=date(2026, 1, i + 1))
        add_label(tmp_db, i + 1, food, "user")
    _insert_expense(tmp_db, eid=20, cents=-90000, cpn="vermieter", iban="DE2", d=date(2026, 2, 1))
    add_label(tmp_db, 20, rent, "user")
    # Unlabelled candidates: some look like rent, some like food.
    _insert_expense(tmp_db, eid=30, cents=-90500, cpn="vermieter", iban="DE2", d=date(2026, 3, 1))
    _insert_expense(tmp_db, eid=31, cents=-510, cpn="markt", iban="DE1", d=date(2026, 3, 2))

    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    cfg.active_learning.diversity_min_label_per_category = 5
    picks = select_diverse(tmp_db, emb, n=1, config=cfg)
    # The single pick should be the rent-like row (under-covered category).
    assert picks == [30]


def test_evaluate_label_batch_impact_runs(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    from expensa.ml.active_learning import evaluate_label_batch_impact

    food = upsert_category(tmp_db, "Lebensmittel")
    rent = upsert_category(tmp_db, "Miete")
    for i in range(12):
        cat = food if i % 2 == 0 else rent
        cents = -500 if i % 2 == 0 else -90000
        _insert_expense(
            tmp_db, eid=i + 1, cents=cents, cpn=f"v{i % 4}", iban="DE1",
            d=date(2026, 1, 1) + timedelta(days=i),
        )
        add_label(tmp_db, i + 1, cat, "user")
    emb = HashEmbedder(dim=64)
    store_embeddings(
        tmp_db, emb,
        [(r["id"], r["combined_text"]) for r in tmp_db.execute(
            "SELECT id, combined_text FROM expenses"
        ).fetchall()],
    )
    cfg = _cfg(tmp_path)
    out = evaluate_label_batch_impact(
        tmp_db, cfg, emb, batch_ids=[11, 12], n_folds=2,
    )
    assert out["n_batch"] == 2
    assert "after_accuracy" in out and "delta" in out
