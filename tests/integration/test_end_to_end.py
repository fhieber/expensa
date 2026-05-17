"""End-to-end integration tests.

The fast variant uses HashEmbedder so CI doesn't download a 1 GB model.
The slow variant (``@pytest.mark.slow``) runs the real
T-Systems sentence-transformer; opt in with ``pytest -m slow``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
)
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.active_learning import pick_candidates
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.categories import (
    add_label,
    upsert_category,
)
from expense_analyzer.storage.database import get_or_create_database
from expense_analyzer.viz import (
    monthly_flow_by_category,
    pie_chart,
    save_figure,
    spend_by_category,
    trend_lines,
)


def _config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.enabled = False  # never reach for HF unless explicitly tested
    return cfg


def _seed_labels(conn: sqlite3.Connection) -> None:
    """Label one example per category to bootstrap the cascade."""
    food = upsert_category(conn, "Lebensmittel")
    rent = upsert_category(conn, "Miete")
    income = upsert_category(conn, "Einkommen")
    streaming = upsert_category(conn, "Abos")
    transport = upsert_category(conn, "Transport")

    seeds = [
        ("rewe markt", food),
        ("edeka sued", food),
        ("aldi sued", food),
        ("vermieter schmidt", rent),
        ("spotify", streaming),
        ("netflix international", streaming),
        ("db bahn", transport),
        ("tankstelle aral", transport),
    ]
    for cp, cid in seeds:
        row = conn.execute(
            "SELECT id FROM expenses WHERE counterparty_normalized LIKE ? ORDER BY id LIMIT 1",
            (f"{cp}%",),
        ).fetchone()
        if row is not None:
            add_label(conn, int(row["id"]), cid, "user")
    arbeitgeber = conn.execute(
        "SELECT id FROM expenses WHERE zahlungspflichtiger='Arbeitgeber AG' LIMIT 1"
    ).fetchone()
    if arbeitgeber is not None:
        add_label(conn, int(arbeitgeber["id"]), income, "user")


def _full_pipeline(
    tmp_path: Path, fixtures_dir: Path, embedder: Embedder
) -> tuple[sqlite3.Connection, dict]:
    cfg = _config(tmp_path)
    conn = get_or_create_database(cfg.db_path)

    # Ingest the main CSV, then the overlap CSV. Verify dedup.
    r1 = ingest_csv(conn, fixtures_dir / "sample_de.csv")
    r2 = ingest_csv(conn, fixtures_dir / "sample_overlap.csv")
    assert r1.inserted == 50
    assert r2.inserted == 7 and r2.duplicates == 6

    # Re-ingesting the same file is a no-op.
    r3 = ingest_csv(conn, fixtures_dir / "sample_de.csv")
    assert r3.inserted == 0

    # Seed labels.
    _seed_labels(conn)

    # Train + predict.
    cascade = CategorizationCascade(conn, cfg, embedder)
    fit = cascade.fit()
    assert fit.train_score >= 0.99  # overfits seed labels

    unlabeled = [
        int(r["id"])
        for r in conn.execute(
            "SELECT id FROM expenses "
            "WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source='user') "
            "ORDER BY id"
        ).fetchall()
    ]
    preds = cascade.predict_batch(unlabeled)
    assert len(preds) == len(unlabeled)
    # At least the recurring REWE/Edeka rows should be hit by vendor_exact_match.
    rewe_pred = next(
        (p for p in preds
         if conn.execute(
            "SELECT counterparty_normalized FROM expenses WHERE id=?", (p.expense_id,)
        ).fetchone()["counterparty_normalized"] == "rewe markt"),
        None,
    )
    assert rewe_pred is not None
    assert rewe_pred.stage == "vendor_exact_match"

    # Active learning still has work to surface.
    candidates = pick_candidates(conn, cfg, embedder, cascade, n=5, strategy="uncertainty")
    assert len(candidates) > 0

    # Visualizations render and write to disk.
    out = tmp_path / "exports" / "pie.html"
    save_figure(pie_chart(spend_by_category(conn)), out)
    assert out.exists()

    out2 = tmp_path / "exports" / "trend.html"
    save_figure(trend_lines(monthly_flow_by_category(conn)), out2)
    assert out2.exists()

    return conn, {
        "n_total": 57,
        "fit_score": fit.train_score,
    }


def test_end_to_end_with_hash_embedder(tmp_path: Path, fixtures_dir: Path) -> None:
    """Fast end-to-end: ingest -> dedup -> seed labels -> train -> predict
    -> active learning -> viz. Uses the HashEmbedder."""
    conn, stats = _full_pipeline(tmp_path, fixtures_dir, HashEmbedder(dim=64))
    assert stats["n_total"] == 57


@pytest.mark.slow
def test_end_to_end_with_real_embedder(tmp_path: Path, fixtures_dir: Path) -> None:
    """Same end-to-end but with the real T-Systems sentence-transformer.

    First run downloads the model (~1.1 GB) into ~/.cache/huggingface; subsequent
    runs are fast. Skip with ``pytest -m 'not slow'``.
    """
    embedder = SentenceTransformerEmbedder(
        model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        device="auto",
    )
    conn, stats = _full_pipeline(tmp_path, fixtures_dir, embedder)
    assert stats["n_total"] == 57
