"""Tests for the eval disk cache + zeroshot premise enrichment.

Covers:
  * ``eval_cache.save``/``load``/``clear`` round-trip including atomic
    overwrite and schema-version invalidation.
  * ``_build_zeroshot_premise`` composition rules (empty pieces, long
    summary truncation, ordering).
  * The new ZeroshotConfig fields default sensibly and validate.

No HF model downloads; no transformer pipelines instantiated.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from expense_analyzer.config import Config, ZeroshotConfig
from expense_analyzer.ml import eval_cache
from expense_analyzer.ml.classifier import _build_zeroshot_premise
from expense_analyzer.ml.evaluation import (
    AblationResult,
    EvalResult,
    PerCategory,
    StageBreakdown,
)

# ─── fixtures ─────────────────────────────────────────────────────────


def _fake_result() -> EvalResult:
    return EvalResult(
        n_labeled=10,
        n_folds=2,
        accuracy=0.7,
        accuracy_covered=0.78,
        macro_f1=0.65,
        weighted_f1=0.71,
        coverage=0.9,
        per_category=[PerCategory(1, 0.8, 0.7, 0.74, 5)],
        confusion=np.array([[3, 1], [1, 5]], dtype=np.int64),
        confusion_labels=[1, 2],
        stage_breakdown=[StageBreakdown("classifier", 8, 6, 0.75)],
        records=[(1, 1, 1, "classifier", True)],
        dropped_singletons=0,
        notes="",
    )


def _fake_ablation() -> AblationResult:
    return AblationResult(
        cumulative=[("classifier", 0.7, 0.65)],
        leave_one_out=[("classifier", 0.5, 0.45, -0.20)],
        full_accuracy=0.7,
        full_macro_f1=0.65,
    )


# ─── eval_cache ───────────────────────────────────────────────────────


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    result = _fake_result()
    abl = _fake_ablation()
    meta = {"seed": 7, "include_zeroshot": False}

    saved_path = eval_cache.save(tmp_path, result, abl, meta)
    assert saved_path.is_file()
    # Stored under the cache subdir, not at the data_dir root.
    assert saved_path.parent.name == "cache"

    loaded = eval_cache.load(tmp_path)
    assert loaded is not None
    # Pickle round-trip preserves the numpy confusion matrix exactly.
    assert np.array_equal(loaded.result.confusion, result.confusion)
    assert loaded.result.accuracy == result.accuracy
    assert loaded.ablation is not None
    assert loaded.ablation.full_accuracy == abl.full_accuracy
    assert loaded.meta == meta
    assert isinstance(loaded.saved_at, datetime)


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    assert eval_cache.load(tmp_path) is None


def test_load_returns_none_on_schema_mismatch(tmp_path: Path) -> None:
    """A pickle from a future / past schema must not crash the tab --
    we just treat it as "no cache" and let the user re-run."""
    path = tmp_path / "cache" / "eval_latest.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"schema_version": 999, "saved_at": "x"}, fh)
    assert eval_cache.load(tmp_path) is None


def test_load_returns_none_on_corrupt_pickle(tmp_path: Path) -> None:
    """Garbage bytes shouldn't crash -- the eval tab must always render."""
    path = tmp_path / "cache" / "eval_latest.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a pickle, just bytes")
    assert eval_cache.load(tmp_path) is None


def test_save_is_atomic_overwrite(tmp_path: Path) -> None:
    """Second save replaces the first; no stray .tmp file lingers."""
    eval_cache.save(tmp_path, _fake_result(), None, {"seed": 0, "include_zeroshot": True})
    eval_cache.save(tmp_path, _fake_result(), None, {"seed": 1, "include_zeroshot": False})
    loaded = eval_cache.load(tmp_path)
    assert loaded is not None
    assert loaded.meta["seed"] == 1
    assert loaded.meta["include_zeroshot"] is False
    # No tmp file left behind by either write.
    assert not (tmp_path / "cache" / "eval_latest.pkl.tmp").exists()


def test_clear_removes_cache(tmp_path: Path) -> None:
    eval_cache.save(tmp_path, _fake_result(), None, {"seed": 0, "include_zeroshot": False})
    assert eval_cache.clear(tmp_path) is True
    assert eval_cache.load(tmp_path) is None
    # Second clear is a no-op, not an error.
    assert eval_cache.clear(tmp_path) is False


# ─── zeroshot premise composition ─────────────────────────────────────


def test_build_premise_text_only() -> None:
    """No vendor data -> the premise is the expense text, period-suffixed."""
    p = _build_zeroshot_premise("rewe markt einkauf", "", "", 240)
    assert p == "rewe markt einkauf."


def test_build_premise_preserves_existing_period() -> None:
    """If the text already ends with '.', don't double it up."""
    p = _build_zeroshot_premise("rewe markt.", "", "", 240)
    assert p == "rewe markt."


def test_build_premise_adds_industry_and_summary() -> None:
    p = _build_zeroshot_premise(
        "edeka filiale",
        "supermarket",
        "Edeka ist eine deutsche Supermarktkette mit Sitz in Hamburg.",
        240,
    )
    # Order: text → industry → summary, separated by single spaces.
    assert p.startswith("edeka filiale.")
    assert "Branche: supermarket." in p
    assert "Hamburg" in p
    # No double spaces, no stray separators.
    assert "  " not in p


def test_build_premise_truncates_long_summary() -> None:
    """Summary cap protects the NLI tokenizer from 600-char DDG dumps."""
    long_summary = (
        "Das ist ein sehr langer Werbetext der die NLI Eingabe komplett "
        "uebernehmen wuerde wenn wir ihn nicht abschneiden. " * 5
    )
    p = _build_zeroshot_premise("vendor x", "", long_summary, 80)
    # Truncated summary itself <= cap+1 (for the trailing ellipsis).
    truncated = p.split("vendor x. ", 1)[1]
    assert len(truncated) <= 81
    assert truncated.endswith("…")


def test_build_premise_skips_empty_pieces() -> None:
    """Empty industry/summary must not add stray "Branche: ." fragments."""
    p = _build_zeroshot_premise("vendor x", "", "", 240)
    assert "Branche" not in p
    p2 = _build_zeroshot_premise("vendor x", "shop", "", 240)
    assert "Branche: shop." in p2
    # No trailing "summary" gap.
    assert p2 == "vendor x. Branche: shop."


def test_build_premise_empty_text_returns_only_extras() -> None:
    """Even with no expense text, vendor pieces should still surface."""
    p = _build_zeroshot_premise("", "shop", "summary", 240)
    assert p == "Branche: shop. summary"


# ─── ZeroshotConfig defaults ──────────────────────────────────────────


def test_zeroshot_config_defaults_are_german() -> None:
    z = ZeroshotConfig()
    assert "{}" in z.hypothesis_template
    assert z.hypothesis_template.startswith("In diesem Text")
    assert z.use_vendor_context is False  # opt-in
    assert z.vendor_summary_max_chars > 0


def test_zeroshot_config_accepts_custom_template(tmp_path: Path) -> None:
    """User can override the template via the Settings tab."""
    cfg = Config(data_dir=tmp_path)
    cfg.zeroshot.hypothesis_template = "This text is about {}."
    cfg.zeroshot.use_vendor_context = True
    # Pydantic re-validation roundtrip mirrors what save_user_config does.
    payload = cfg.model_dump()
    rebuilt = Config(**payload)
    assert rebuilt.zeroshot.hypothesis_template == "This text is about {}."
    assert rebuilt.zeroshot.use_vendor_context is True
