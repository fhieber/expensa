"""Cold-start cascade should produce reasonable predictions via
category_similarity, *without* any user labels.

Uses the real T-Systems sentence-transformer (marked @slow).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expensa.config import Config, packaged_default_categories
from expensa.features.embeddings import SentenceTransformerEmbedder
from expensa.ingestion import ingest_csv
from expensa.ml.classifier import CategorizationCascade
from expensa.storage.categories import (
    import_categories_from_yaml,
    list_categories,
)
from expensa.storage.database import get_or_create_database


@pytest.mark.slow
def test_cold_start_category_similarity_predicts_german_supermarkets(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """No user labels at all → category_similarity should still match the
    obvious supermarket rows (REWE / Edeka / Aldi) to 'Lebensmittel'."""
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.enabled = False  # don't fall through to NLI

    conn = get_or_create_database(cfg.db_path)
    import_categories_from_yaml(conn, packaged_default_categories())

    emb = SentenceTransformerEmbedder(
        model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        device="auto",
    )
    ingest_csv(conn, fixtures_dir / "sample_de.csv", embedder=emb)

    cascade = CategorizationCascade(conn, cfg, emb)
    cascade.fit()  # no user labels yet, but it embeds categories

    cats_by_id = {c.id: c.name for c in list_categories(conn)}

    # All supermarket rows -> Lebensmittel.
    super_rows = conn.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized IN "
        "('rewe markt', 'edeka sued', 'aldi sued') ORDER BY id"
    ).fetchall()
    super_ids = [int(r["id"]) for r in super_rows]
    preds = cascade.predict_batch(super_ids)
    super_predicted = [cats_by_id.get(p.category_id, "?") for p in preds]
    n_food = sum(1 for n in super_predicted if n == "Lebensmittel")
    assert n_food >= max(3, len(super_ids) - 2), (
        f"expected most supermarket rows -> Lebensmittel, got {super_predicted}"
    )

    # Vermieter Schmidt rows -> Miete.
    rent_rows = conn.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'vermieter schmidt'"
    ).fetchall()
    if rent_rows:
        preds = cascade.predict_batch([int(r["id"]) for r in rent_rows])
        names = [cats_by_id.get(p.category_id, "?") for p in preds]
        assert all(n == "Miete" for n in names), f"rent rows -> {names}"

    # All stages used should be category_similarity (no classifier or zeroshot here).
    all_ids = [int(r["id"]) for r in conn.execute("SELECT id FROM expenses").fetchall()]
    preds = cascade.predict_batch(all_ids)
    stages = {p.stage for p in preds}
    assert "category_similarity" in stages
    # NLI must NOT have fired since we disabled it.
    assert "zeroshot" not in stages

    conn.close()
