"""Cascaded categorizer.

Stages, in order:
  1. ``vendor_exact_match``: same counterparty already labeled, agreement >= threshold.
  2. ``knn``: k-nearest labeled embeddings; agreement >= threshold.
  3. ``classifier``: scikit-learn pipeline on combined features.
  4. ``category_similarity``: zero-shot via cosine similarity between the
     expense embedding and each category's ``name: description`` embedding.
     Works cold-start (no user labels needed).
  5. ``zeroshot``: mDeBERTa NLI fallback (lazy-loaded, slowest).

Each stage emits a Prediction or yields to the next.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from expense_analyzer.config import Config
from expense_analyzer.features.embeddings import Embedder, load_embeddings
from expense_analyzer.features.pipeline import build_full_features
from expense_analyzer.storage.categories import (
    labeled_ids_with_categories,
    vendor_label_distribution,
)


@dataclass
class Prediction:
    expense_id: int
    category_id: int | None
    confidence: float
    stage: str  # 'vendor_exact_match' | 'knn' | 'classifier' | 'category_similarity' | 'zeroshot' | 'unknown'
    runner_up: int | None = None
    runner_up_confidence: float = 0.0
    notes: str = ""


@dataclass
class FitReport:
    n_train: int
    n_classes: int
    classifier_type: str
    feature_dim: int
    train_score: float
    notes: str = ""


_NUMERIC_COLS = (
    "is_income",
    "is_round",
    "iban_is_foreign",
    "iban_is_known_self",
    "has_glaeubiger_id",
    "mandatsreferenz_present",
    "is_likely_recurring",
    "log_abs_amount",
    "year",
    "month",
    "quarter",
    "day_of_week",
    "is_weekend",
    "is_month_end",
    "days_since_prev_same_cp",
    "count_same_cp_30d",
    "count_same_cp_90d",
    "count_same_cp_365d",
    "amount_zscore_within_cp",
)


def _numeric_block(df) -> np.ndarray:
    out = np.zeros((len(df), len(_NUMERIC_COLS)), dtype=np.float32)
    for j, c in enumerate(_NUMERIC_COLS):
        if c in df.columns:
            col = df[c].fillna(0).astype("float32").to_numpy()
        else:
            col = np.zeros(len(df), dtype=np.float32)
        out[:, j] = col
    return out


def _build_x(df, embeddings: np.ndarray) -> np.ndarray:
    return np.hstack([embeddings.astype(np.float32, copy=False), _numeric_block(df)])


def _vendor_exact_match(
    conn: sqlite3.Connection, counterparty_normalized: str, agreement_min: float
) -> tuple[int, float] | None:
    if not counterparty_normalized:
        return None
    dist = vendor_label_distribution(conn, counterparty_normalized)
    total = sum(dist.values())
    if total == 0:
        return None
    cat_id, n = max(dist.items(), key=lambda kv: kv[1])
    agreement = n / total
    if agreement >= agreement_min:
        return cat_id, agreement
    return None


def _knn_vote(
    target_vec: np.ndarray,
    train_vecs: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    agreement_min: int,
) -> tuple[int, float] | None:
    if train_vecs.shape[0] == 0:
        return None
    # Cosine similarity (vectors are unit-normalized when produced by ST or HashEmbedder).
    sims = train_vecs @ target_vec
    k_eff = min(k, len(sims))
    top_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
    votes = train_labels[top_idx]
    values, counts = np.unique(votes, return_counts=True)
    best = int(counts.argmax())
    if counts[best] >= agreement_min:
        return int(values[best]), float(counts[best] / k_eff)
    return None


class CategorizationCascade:
    """Glue holding all stages plus the trained sklearn model."""

    def __init__(self, conn: sqlite3.Connection, config: Config, embedder: Embedder) -> None:
        self.conn = conn
        self.cfg = config
        self.embedder = embedder
        self._sk = None  # sklearn pipeline
        self._classes_: np.ndarray | None = None
        self._feature_dim: int | None = None
        self._zs = None  # zeroshot pipeline (lazy)
        # Category-similarity stage cache: (cat_ids, embedding_matrix)
        self._cat_emb_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ---- category similarity (zero-shot via embeddings) -----------------

    def _populate_category_embeddings(self) -> None:
        """Embed each category as multiple text variants and pre-compute
        a token set per category for the lexical-overlap bonus.

        At inference we combine:
          * per-category max cosine over the variants  (semantic signal)
          * a small bonus per shared >=4-char token    (lexical signal)
        This catches both fuzzy semantic matches AND obvious keyword hits
        like "Lebensmittel" appearing in both expense text and category
        description.
        """
        import re

        from expense_analyzer.storage.categories import list_categories

        cats = list_categories(self.conn)
        if not cats:
            self._cat_emb_cache = None
            return

        variants: list[str] = []
        variant_to_cat: list[int] = []
        cat_ids: list[int] = []
        cat_token_sets: dict[int, set[str]] = {}
        for c in cats:
            cat_ids.append(c.id)
            variants.append(c.name)
            variant_to_cat.append(c.id)
            if c.description:
                variants.append(f"{c.name}: {c.description}")
                variant_to_cat.append(c.id)
            text_for_tokens = f"{c.name} {c.description or ''}".lower()
            cat_token_sets[c.id] = {
                t for t in re.findall(r"[\wäöüß]+", text_for_tokens) if len(t) >= 4
            }

        matrix = self.embedder.encode(variants).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._cat_emb_cache = (
            np.array(cat_ids, dtype=np.int64),
            matrix,
            np.array(variant_to_cat, dtype=np.int64),
            cat_token_sets,
        )

    def _category_similarity(
        self, expense_vec: np.ndarray, expense_text: str = ""
    ) -> tuple[int | None, float]:
        import re

        if self._cat_emb_cache is None:
            self._populate_category_embeddings()
        if self._cat_emb_cache is None:
            return None, 0.0
        cat_ids, var_mat, var_to_cat, cat_token_sets = self._cat_emb_cache
        v = expense_vec.astype(np.float32, copy=False)
        n = float(np.linalg.norm(v))
        if n == 0:
            return None, 0.0
        var_sims = var_mat @ (v / n)

        # Per-category max cosine across that category's variants.
        embed_per_cat: dict[int, float] = {}
        for cid_int, sim in zip(var_to_cat.tolist(), var_sims.tolist(), strict=True):
            if sim > embed_per_cat.get(cid_int, float("-inf")):
                embed_per_cat[cid_int] = sim

        # Lexical bonus: tokens shared between expense_text and category text.
        bonus_per_cat: dict[int, float] = {}
        if expense_text:
            tokens = {
                t for t in re.findall(r"[\wäöüß]+", expense_text.lower())
                if len(t) >= 4
            }
            for cid_int, cat_tokens in cat_token_sets.items():
                overlap = tokens & cat_tokens
                # 0.10 per hit, capped at 0.30 so semantic always matters too.
                if overlap:
                    bonus_per_cat[cid_int] = min(0.30, 0.10 * len(overlap))

        combined: dict[int, float] = {
            cid: embed_per_cat[cid] + bonus_per_cat.get(cid, 0.0)
            for cid in embed_per_cat
        }
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top_cat, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            top_score >= self.cfg.category_similarity.min_top1
            and (top_score - second_score) >= self.cfg.category_similarity.min_margin
        ):
            return int(top_cat), float(top_score)
        return None, 0.0

    # ---- training -------------------------------------------------------

    def fit(self) -> FitReport:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Always embed categories -- the similarity stage works cold-start.
        if self.cfg.category_similarity.enabled:
            self._populate_category_embeddings()

        labels = labeled_ids_with_categories(self.conn, source="user")
        if len(labels) < 2:
            return FitReport(
                n_train=len(labels),
                n_classes=len({c for _, c in labels}),
                classifier_type="none",
                feature_dim=0,
                train_score=float("nan"),
                notes="need >= 2 labeled examples to train classifier (category_similarity still works)",
            )
        ids = [eid for eid, _ in labels]
        y = np.array([cid for _, cid in labels])

        df, emb = build_full_features(self.conn, embedder=self.embedder, expense_ids=ids)
        # Reorder labels to match df.index
        df = df.reindex([eid for eid in ids if eid in df.index])
        keep_mask = np.array([eid in df.index for eid in ids])
        y = y[keep_mask]
        assert emb is not None
        # build_full_features already aligned emb with df.index, so just use as-is
        X = _build_x(df, emb)

        n_classes = len(np.unique(y))
        n_train = len(y)
        if n_classes < 2:
            return FitReport(
                n_train=n_train,
                n_classes=n_classes,
                classifier_type="none",
                feature_dim=X.shape[1],
                train_score=float("nan"),
                notes="need labels covering >=2 categories to train",
            )

        if (
            self.cfg.classifier.type == "random_forest"
            or n_train >= self.cfg.classifier.rf_switch_threshold
        ):
            clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            classifier_type = "random_forest"
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])
        else:
            clf = LogisticRegression(max_iter=1000, n_jobs=None)
            classifier_type = "logistic_regression"
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])

        pipe.fit(X, y)
        self._sk = pipe
        self._classes_ = pipe.named_steps["clf"].classes_
        self._feature_dim = X.shape[1]
        train_score = float(pipe.score(X, y))
        return FitReport(
            n_train=n_train,
            n_classes=n_classes,
            classifier_type=classifier_type,
            feature_dim=X.shape[1],
            train_score=train_score,
        )

    # ---- prediction -----------------------------------------------------

    def predict(self, expense_id: int) -> Prediction:
        return self.predict_batch([expense_id])[0]

    def predict_batch(
        self,
        expense_ids: Sequence[int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Prediction]:
        """Predict categories for the given expense IDs.

        `progress_callback`, if given, is invoked as ``cb(done, total)``
        after each record so the UI can drive a progress bar. ``done``
        is 1-indexed.
        """
        if not expense_ids:
            return []
        df, emb = build_full_features(self.conn, embedder=self.embedder, expense_ids=list(expense_ids))
        # Build the labeled-set view for k-NN
        labels = labeled_ids_with_categories(self.conn, source="user")
        train_ids_arr = np.array([eid for eid, _ in labels], dtype=np.int64)
        train_y = np.array([cid for _, cid in labels], dtype=np.int64)
        if len(train_ids_arr) > 0:
            ids_loaded, train_vecs_all = load_embeddings(
                self.conn, self.embedder.model_name, list(train_ids_arr)
            )
            id_to_pos = {eid: i for i, eid in enumerate(ids_loaded)}
            order = [id_to_pos.get(int(eid)) for eid in train_ids_arr]
            keep = [(i, p) for i, p in enumerate(order) if p is not None]
            train_vecs = (
                train_vecs_all[[p for _, p in keep]] if keep else np.zeros((0, 0), dtype=np.float32)
            )
            train_y = train_y[[i for i, _ in keep]] if keep else train_y[:0]
        else:
            train_vecs = np.zeros((0, 0), dtype=np.float32)

        out: list[Prediction] = []
        # Lazy-build the X matrix only if we actually need the classifier.
        X_cache = None

        total = len(expense_ids)
        for done, eid in enumerate(expense_ids, start=1):
            try:
                if eid not in df.index:
                    out.append(
                        Prediction(eid, None, 0.0, "unknown", notes="not found / no features")
                    )
                    continue
                row = df.loc[eid]

                # Stage 1: vendor exact match
                if self.cfg.vendor_exact_match.enabled:
                    hit = _vendor_exact_match(
                        self.conn,
                        str(row.get("counterparty_normalized") or ""),
                        self.cfg.vendor_exact_match.agreement_min,
                    )
                    if hit:
                        cid, agreement = hit
                        out.append(Prediction(eid, cid, agreement, "vendor_exact_match"))
                        continue

                # Stage 2: k-NN over labeled embeddings
                if self.cfg.knn.enabled and train_vecs.shape[0] > 0:
                    target_pos = list(df.index).index(eid)
                    target_vec = emb[target_pos]
                    hit = _knn_vote(
                        target_vec,
                        train_vecs,
                        train_y,
                        k=self.cfg.knn.k,
                        agreement_min=self.cfg.knn.agreement_min,
                    )
                    if hit:
                        cid, conf = hit
                        out.append(Prediction(eid, cid, conf, "knn"))
                        continue

                # Stage 3: supervised classifier
                if self._sk is not None:
                    if X_cache is None:
                        X_cache = _build_x(df, emb)
                    target_pos = list(df.index).index(eid)
                    proba = self._sk.predict_proba(X_cache[target_pos : target_pos + 1])[0]
                    top = int(np.argmax(proba))
                    top_conf = float(proba[top])
                    runner = int(np.argsort(proba)[-2]) if len(proba) > 1 else top
                    if top_conf >= self.cfg.classifier.confidence_threshold:
                        out.append(
                            Prediction(
                                eid,
                                int(self._classes_[top]),
                                top_conf,
                                "classifier",
                                runner_up=int(self._classes_[runner]),
                                runner_up_confidence=float(proba[runner]),
                            )
                        )
                        continue
                    low_conf = top_conf < self.cfg.zeroshot.use_when_confidence_below
                else:
                    low_conf = True

                # Stage 4: category similarity (zero-shot via embeddings + lexical overlap)
                if self.cfg.category_similarity.enabled and low_conf:
                    target_pos = list(df.index).index(eid)
                    cid, conf = self._category_similarity(
                        emb[target_pos],
                        expense_text=str(row.get("combined_text") or ""),
                    )
                    if cid is not None:
                        out.append(Prediction(eid, cid, conf, "category_similarity"))
                        continue

                # Stage 5: zero-shot NLI (slowest fallback)
                if self.cfg.zeroshot.enabled and low_conf:
                    cid, conf = self._zeroshot_predict(str(row.get("combined_text") or ""))
                    if cid is not None:
                        out.append(Prediction(eid, cid, conf, "zeroshot"))
                        continue

                # Nothing fired: surface as unknown so the UI flags it.
                out.append(Prediction(eid, None, 0.0, "unknown"))
            finally:
                # Always fire the callback, even when a stage continue'd above.
                if progress_callback is not None:
                    progress_callback(done, total)

        return out

    # ---- zeroshot -------------------------------------------------------

    def _zeroshot_predict(self, text: str) -> tuple[int | None, float]:
        if not text.strip():
            return None, 0.0
        from expense_analyzer.storage.categories import list_categories

        cats = list_categories(self.conn)
        if not cats:
            return None, 0.0
        if self._zs is None:
            try:
                from transformers import pipeline

                self._zs = pipeline(
                    "zero-shot-classification",
                    model=self.cfg.zeroshot_model,
                )
            except Exception:
                # Network or HF cache unreachable; skip silently.
                self._zs = lambda *_a, **_kw: None  # type: ignore
                return None, 0.0
        # Use category names plus their descriptions as labels.
        label_map = {f"{c.name}: {c.description}": c.id for c in cats if c.name}
        labels = list(label_map.keys())
        try:
            res = self._zs(text, candidate_labels=labels, multi_label=False)
        except Exception:
            return None, 0.0
        if not res:
            return None, 0.0
        top_label = res["labels"][0]
        top_score = float(res["scores"][0])
        return label_map[top_label], top_score
