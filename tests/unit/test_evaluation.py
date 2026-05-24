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
    fold_sizes,
    planned_ablation_runs,
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


def test_fold_sizes_matches_actual_cv(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """`fold_sizes` is a UI helper used to display per-fold counts BEFORE
    cross-validation runs. The sums must match what the real CV sees so
    the status text doesn't lie to the user."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _seed_labels(tmp_db)

    n_kept, n_dropped, sizes = fold_sizes(tmp_db, n_folds=2, seed=0)
    assert n_kept > 0
    # Test halves sum to the kept-label total (each label tested exactly once).
    assert sum(test_n for _, test_n in sizes) == n_kept
    # Train + test per fold ≤ kept (some classes may be singletons within a fold).
    for train_n, test_n in sizes:
        assert train_n + test_n <= n_kept
    # Sanity: the same n_folds/seed used by cross_validate must produce
    # the same partition shape; this is the contract the UI relies on.
    result = cross_validate(
        tmp_db, _cfg(tmp_path), HashEmbedder(dim=64), n_folds=2, seed=0
    )
    assert result.n_folds == len(sizes)
    assert result.dropped_singletons == n_dropped


def test_planned_ablation_runs_matches_ablation_order(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """The planned label list must be the same length and order as the
    runs `ablation()` actually performs — the UI uses indices to label
    progress text."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _seed_labels(tmp_db)
    cfg = _cfg(tmp_path)
    planned = planned_ablation_runs(cfg)
    abl = ablation(tmp_db, cfg, HashEmbedder(dim=64), n_folds=2, seed=0)
    # Total runs = cumulative + 1 baseline + leave-one-out.
    assert len(planned) == len(abl.cumulative) + 1 + len(abl.leave_one_out)
    # First N entries are cumulative subsets in pipeline order.
    for i, (label, subset) in enumerate(planned[: len(abl.cumulative)]):
        assert label.startswith("cumulative:")
        assert len(subset) == i + 1
    # The baseline sits between cumulative and leave-one-out.
    baseline = planned[len(abl.cumulative)]
    assert baseline[0].startswith("full baseline:")
    # Tail entries label each leave-one-out drop.
    for label, _subset in planned[len(abl.cumulative) + 1 :]:
        assert label.startswith("leave-one-out: drop ")


def test_build_pdf_produces_pdf_bytes(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Smoke-test the PDF report builder end-to-end: real CV + ablation
    + reportlab render. Skips if reportlab or kaleido isn't installed."""
    pytest = __import__("pytest")
    try:
        import kaleido  # noqa: F401
        import reportlab  # noqa: F401
    except ImportError:
        pytest.skip("report-export extras not installed")

    from expense_analyzer.ml.eval_report import (
        ReportContext,
        build_pdf,
        default_filename,
    )

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cat_ids = _seed_labels(tmp_db)
    cfg = _cfg(tmp_path)
    embedder = HashEmbedder(dim=64)
    result = cross_validate(tmp_db, cfg, embedder, n_folds=2, seed=0)
    abl = ablation(tmp_db, cfg, embedder, n_folds=2, seed=0)

    id_to_name = {v: k.capitalize() for k, v in cat_ids.items()}
    ctx = ReportContext(
        account_name="Test Account",
        embedding_model="hash-test",
        n_folds=result.n_folds,
        seed=0,
        include_zeroshot=False,
        category_id_to_name=id_to_name,
    )
    pdf = build_pdf(result, abl, ctx)
    # %PDF header is the unambiguous "this is a valid PDF" signal.
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 2000  # non-trivial content (charts + tables)

    fname = default_filename("Test Account")
    assert fname.endswith(".pdf")
    assert "Test_Account" in fname  # space -> underscore by the sanitiser


def test_format_setting_value_renders_each_python_type() -> None:
    """``_format_setting_value`` is the appendix's one-stop value
    formatter; keep its contract pinned so future stage additions
    don't silently emit ``True``/``False``/``0.700000`` in the PDF."""
    from expense_analyzer.ml.eval_report import _format_setting_value

    assert _format_setting_value(True) == "yes"
    assert _format_setting_value(False) == "no"
    # Trailing zeros stripped so 0.70 -> 0.7 in the table.
    assert _format_setting_value(0.7) == "0.7"
    assert _format_setting_value(0.025) == "0.025"
    assert _format_setting_value(0.0) == "0"
    assert _format_setting_value(5) == "5"
    assert _format_setting_value("In diesem Text geht es um {}.") == (
        "In diesem Text geht es um {}."
    )
    # Empty string surfaces as a placeholder so the table never has a
    # truly blank cell (which reads as "missing", not as "empty").
    assert _format_setting_value("") == "(empty)"


def test_build_pdf_includes_cascade_settings_appendix(
    tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path
) -> None:
    """Two builds: one with the appendix data, one without. The
    appendix version must be larger AND end with a valid PDF marker."""
    pytest = __import__("pytest")
    try:
        import kaleido  # noqa: F401
        import reportlab  # noqa: F401
    except ImportError:
        pytest.skip("report-export extras not installed")

    from expense_analyzer.ml.eval_report import ReportContext, build_pdf

    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cat_ids = _seed_labels(tmp_db)
    cfg = _cfg(tmp_path)
    embedder = HashEmbedder(dim=64)
    result = cross_validate(tmp_db, cfg, embedder, n_folds=2, seed=0)
    id_to_name = {v: k.capitalize() for k, v in cat_ids.items()}

    base_ctx = ReportContext(
        account_name="Test",
        embedding_model="hash-test",
        n_folds=result.n_folds,
        seed=0,
        include_zeroshot=False,
        category_id_to_name=id_to_name,
    )
    settings_ctx = ReportContext(
        account_name="Test",
        embedding_model="hash-test",
        n_folds=result.n_folds,
        seed=0,
        include_zeroshot=False,
        category_id_to_name=id_to_name,
        cascade_settings={
            "vendor_exact_match": cfg.vendor_exact_match.model_dump(),
            "knn": cfg.knn.model_dump(),
            "classifier": cfg.classifier.model_dump(),
            "category_similarity": cfg.category_similarity.model_dump(),
            "zeroshot": cfg.zeroshot.model_dump(),
        },
        zeroshot_model="moritz/test-nli",
        device="cpu",
    )
    pdf_without = build_pdf(result, None, base_ctx)
    pdf_with = build_pdf(result, None, settings_ctx)
    # Both are valid PDFs.
    assert pdf_without[:4] == b"%PDF"
    assert pdf_with[:4] == b"%PDF"
    # Adding the appendix added real content (one PageBreak + ~7
    # tables -> hundreds of bytes minimum even after PDF compression).
    assert len(pdf_with) > len(pdf_without) + 500
