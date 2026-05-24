"""Quality evaluation for the categorization cascade.

Leak-free k-fold cross-validation over the user's hand-labeled expenses,
plus stage ablation (cumulative and leave-one-out). Pure / Streamlit-free
so it can be driven from both the CLI and the UI and unit-tested with the
``HashEmbedder``.

The cascade reads its training labels straight from the DB, so we hand
each fold's ``CategorizationCascade`` a ``train_ids`` whitelist; the test
fold's own labels stay invisible to the vendor-exact-match and kNN stages.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import Embedder
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.categories import labeled_ids_with_categories

# Cascade stages in firing order. Each name matches a config section with
# an ``enabled`` flag, so any subset can be masked for ablation.
STAGE_ORDER: tuple[str, ...] = (
    "vendor_exact_match",
    "knn",
    "classifier",
    "category_similarity",
    "zeroshot",
)


@dataclass
class StageBreakdown:
    """How a single cascade stage contributed across a full CV run."""

    stage: str
    n_predicted: int  # rows this stage fired on (coverage)
    n_correct: int
    accuracy: float  # accuracy among the rows it fired on


@dataclass
class PerCategory:
    category_id: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalResult:
    n_labeled: int
    n_folds: int
    accuracy: float
    accuracy_covered: float  # accuracy over rows that got a concrete prediction
    macro_f1: float
    weighted_f1: float
    coverage: float  # fraction of test rows that got a concrete prediction
    per_category: list[PerCategory]
    confusion: np.ndarray  # (C, C) rows=true, cols=pred, over confusion_labels
    confusion_labels: list[int]
    stage_breakdown: list[StageBreakdown]
    records: list[tuple[int, int, int | None, str, bool]]  # (eid, true, pred, stage, correct)
    dropped_singletons: int = 0  # labels excluded because their class had <2 members
    notes: str = ""


@dataclass
class AblationResult:
    cumulative: list[tuple[str, float, float]] = field(default_factory=list)
    leave_one_out: list[tuple[str, float, float, float]] = field(default_factory=list)
    full_accuracy: float = float("nan")
    full_macro_f1: float = float("nan")


def _apply_stage_mask(cfg: Config, enabled: set[str]) -> Config:
    """Return a deep copy of ``cfg`` with each cascade stage's ``enabled``
    flag set to whether it's in ``enabled``."""
    masked = cfg.model_copy(deep=True)
    masked.vendor_exact_match.enabled = "vendor_exact_match" in enabled
    masked.knn.enabled = "knn" in enabled
    masked.classifier.enabled = "classifier" in enabled
    masked.category_similarity.enabled = "category_similarity" in enabled
    masked.zeroshot.enabled = "zeroshot" in enabled
    return masked


def _stratified_folds(
    ids: list[int], y: np.ndarray, n_folds: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Build stratified folds, dropping classes with <2 members (they can't
    appear in both a train and a test fold). Clamps ``n_folds`` to the
    smallest surviving class count. Returns (folds, n_dropped_rows)."""
    from sklearn.model_selection import StratifiedKFold

    values, counts = np.unique(y, return_counts=True)
    keepable = set(values[counts >= 2].tolist())
    keep_mask = np.array([cid in keepable for cid in y])
    dropped = int((~keep_mask).sum())

    ids_arr = np.array(ids)[keep_mask]
    y_kept = y[keep_mask]
    if len(y_kept) == 0:
        return [], dropped

    min_class = int(np.min(np.unique(y_kept, return_counts=True)[1]))
    k = max(2, min(n_folds, min_class))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [
        (ids_arr[train_idx], ids_arr[test_idx])
        for train_idx, test_idx in skf.split(ids_arr, y_kept)
    ]
    return folds, dropped


def cross_validate(
    conn: sqlite3.Connection,
    cfg: Config,
    embedder: Embedder,
    n_folds: int = 5,
    seed: int = 0,
    enabled_stages: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvalResult:
    """Leak-free stratified k-fold CV of the cascade on user labels.

    ``enabled_stages`` masks the cascade to a subset of stages (used by
    ablation); ``None`` means use ``cfg`` as-is (all stages per config).
    ``progress_callback(done_folds, total_folds)`` drives a UI bar.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if enabled_stages is not None:
        cfg = _apply_stage_mask(cfg, set(enabled_stages))

    labels = labeled_ids_with_categories(conn, source="user")
    ids = [eid for eid, _ in labels]
    y = np.array([cid for _, cid in labels], dtype=np.int64)
    n_labeled = len(ids)

    folds, dropped = _stratified_folds(ids, y, n_folds, seed)
    if not folds:
        return EvalResult(
            n_labeled=n_labeled,
            n_folds=0,
            accuracy=float("nan"),
            accuracy_covered=float("nan"),
            macro_f1=float("nan"),
            weighted_f1=float("nan"),
            coverage=0.0,
            per_category=[],
            confusion=np.zeros((0, 0), dtype=np.int64),
            confusion_labels=[],
            stage_breakdown=[],
            records=[],
            dropped_singletons=dropped,
            notes="not enough labeled data per category for cross-validation",
        )

    id_to_label = dict(labels)
    records: list[tuple[int, int, int | None, str, bool]] = []

    total = len(folds)
    for done, (train_ids_arr, test_ids_arr) in enumerate(folds, start=1):
        train_ids = {int(x) for x in train_ids_arr.tolist()}
        test_ids = [int(x) for x in test_ids_arr.tolist()]
        cascade = CategorizationCascade(conn, cfg, embedder, train_ids=train_ids)
        cascade.fit()
        preds = cascade.predict_batch(test_ids)
        for p in preds:
            true_cid = id_to_label[p.expense_id]
            correct = p.category_id == true_cid
            records.append((p.expense_id, true_cid, p.category_id, p.stage, correct))
        if progress_callback is not None:
            progress_callback(done, total)

    y_true = np.array([r[1] for r in records], dtype=np.int64)
    # Abstentions (pred None / stage 'unknown') count as wrong; map to a
    # sentinel that can never equal a real category id so accuracy/F1 treat
    # them as errors while coverage reports them separately.
    y_pred = np.array([(-1 if r[2] is None else r[2]) for r in records], dtype=np.int64)

    cat_labels = sorted(set(y_true.tolist()))
    accuracy = float(accuracy_score(y_true, y_pred))
    # F1 over the real categories only; abstentions (pred -1) lower a
    # category's recall but don't form their own spurious class. Macro
    # weights every category equally (surfaces weak rare classes); weighted
    # weights by support (closer to the overall hit rate).
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=cat_labels, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=cat_labels, average="weighted", zero_division=0)
    )
    covered = [r for r in records if r[2] is not None]
    coverage = (len(covered) / len(records)) if records else 0.0
    # Accuracy among rows the cascade actually predicted -- separates model
    # correctness from abstention so a low-coverage run isn't read as wrong.
    accuracy_covered = (
        sum(r[4] for r in covered) / len(covered) if covered else float("nan")
    )

    confusion = confusion_matrix(y_true, y_pred, labels=cat_labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=cat_labels, zero_division=0
    )
    per_category = [
        PerCategory(
            category_id=cid,
            precision=float(prec[i]),
            recall=float(rec[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i, cid in enumerate(cat_labels)
    ]

    stage_breakdown = _stage_breakdown(records)

    return EvalResult(
        n_labeled=n_labeled,
        n_folds=len(folds),
        accuracy=accuracy,
        accuracy_covered=accuracy_covered,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        coverage=coverage,
        per_category=per_category,
        confusion=confusion,
        confusion_labels=cat_labels,
        stage_breakdown=stage_breakdown,
        records=records,
        dropped_singletons=dropped,
    )


def _stage_breakdown(
    records: list[tuple[int, int, int | None, str, bool]],
) -> list[StageBreakdown]:
    """Coverage + accuracy per stage, ordered by STAGE_ORDER then 'unknown'."""
    by_stage: dict[str, list[bool]] = {}
    for _eid, _true, _pred, stage, correct in records:
        by_stage.setdefault(stage, []).append(correct)
    ordered = [s for s in STAGE_ORDER if s in by_stage]
    ordered += [s for s in by_stage if s not in STAGE_ORDER]
    out: list[StageBreakdown] = []
    for stage in ordered:
        flags = by_stage[stage]
        n = len(flags)
        n_correct = sum(flags)
        out.append(
            StageBreakdown(
                stage=stage,
                n_predicted=n,
                n_correct=n_correct,
                accuracy=(n_correct / n) if n else 0.0,
            )
        )
    return out


def ablation(
    conn: sqlite3.Connection,
    cfg: Config,
    embedder: Embedder,
    n_folds: int = 5,
    seed: int = 0,
    stages: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AblationResult:
    """Cumulative and leave-one-out stage ablation.

    Cumulative runs CV with the first 1, 2, ... stages enabled. Leave-one-
    out runs the full cascade, then full-minus-each-stage. By default only
    stages enabled in ``cfg`` participate, so a disabled zero-shot stage is
    not evaluated.
    """
    if stages is None:
        stages = [s for s in STAGE_ORDER if _stage_enabled(cfg, s)]
    stages = list(stages)

    # +1 for the full leave-one-out baseline run.
    total = len(stages) + len(stages) + 1
    done = 0

    def _bump() -> None:
        nonlocal done
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)

    result = AblationResult()

    # Cumulative.
    for i in range(len(stages)):
        subset = stages[: i + 1]
        res = cross_validate(conn, cfg, embedder, n_folds, seed, enabled_stages=subset)
        label = "+".join(subset) if len(subset) > 1 else subset[0]
        result.cumulative.append((label, res.accuracy, res.macro_f1))
        _bump()

    # Full baseline (all participating stages).
    full = cross_validate(conn, cfg, embedder, n_folds, seed, enabled_stages=stages)
    result.full_accuracy = full.accuracy
    result.full_macro_f1 = full.macro_f1
    _bump()

    # Leave-one-out.
    for stage in stages:
        subset = [s for s in stages if s != stage]
        if not subset:
            continue
        res = cross_validate(conn, cfg, embedder, n_folds, seed, enabled_stages=subset)
        delta = res.accuracy - full.accuracy
        result.leave_one_out.append((stage, res.accuracy, res.macro_f1, delta))
        _bump()

    return result


def _stage_enabled(cfg: Config, stage: str) -> bool:
    return bool(getattr(getattr(cfg, stage), "enabled", False))
