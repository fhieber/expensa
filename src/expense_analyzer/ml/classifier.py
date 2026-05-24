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
    conn: sqlite3.Connection,
    counterparty_normalized: str,
    agreement_min: float,
    restrict_ids: set[int] | None = None,
) -> tuple[int, float] | None:
    if not counterparty_normalized:
        return None
    dist = vendor_label_distribution(conn, counterparty_normalized, restrict_ids=restrict_ids)
    total = sum(dist.values())
    if total == 0:
        return None
    cat_id, n = max(dist.items(), key=lambda kv: kv[1])
    agreement = n / total
    if agreement >= agreement_min:
        return cat_id, agreement
    return None


def _knn_vote_from_sims(
    sims: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    agreement_min: int,
) -> tuple[int, float] | None:
    """kNN vote when the cosine-similarity vector is already computed.

    Splitting the vote logic out lets ``predict_batch`` precompute all
    test-vs-train sims with a single matmul (``emb @ train_vecs.T``)
    and then do a cheap lookup per row -- instead of N separate
    matmuls inside the cascade loop.
    """
    if len(sims) == 0:
        return None
    k_eff = min(k, len(sims))
    # argpartition is O(N) vs argsort O(N log N); we only need the top-k
    # so we use the cheaper version.
    top_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
    votes = train_labels[top_idx]
    values, counts = np.unique(votes, return_counts=True)
    best = int(counts.argmax())
    if counts[best] >= agreement_min:
        return int(values[best]), float(counts[best] / k_eff)
    return None


def _knn_vote(
    target_vec: np.ndarray,
    train_vecs: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    agreement_min: int,
) -> tuple[int, float] | None:
    """Per-row kNN vote. Kept for back-compat with direct callers
    (tests + any external user). The hot path in ``predict_batch``
    bypasses this and uses :func:`_knn_vote_from_sims` with a
    bulk-precomputed sim matrix."""
    if train_vecs.shape[0] == 0:
        return None
    # Cosine similarity (vectors are unit-normalized when produced by ST or HashEmbedder).
    sims = train_vecs @ target_vec
    return _knn_vote_from_sims(sims, train_labels, k, agreement_min)


class CategorizationCascade:
    """Glue holding all stages plus the trained sklearn model."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Config,
        embedder: Embedder,
        train_ids: set[int] | None = None,
    ) -> None:
        self.conn = conn
        self.cfg = config
        self.embedder = embedder
        # When set, every label read (classifier fit, kNN neighbours,
        # vendor-exact-match) is restricted to these expense ids. Used by
        # cross-validation to keep a held-out fold's own labels invisible.
        self._train_ids = train_ids
        self._sk = None  # sklearn pipeline
        self._classes_: np.ndarray | None = None
        self._feature_dim: int | None = None
        self._zs = None  # zeroshot pipeline (lazy)
        # Category-similarity stage cache: (cat_ids, embedding_matrix)
        self._cat_emb_cache: tuple[np.ndarray, np.ndarray] | None = None

    def _user_labels(self) -> list[tuple[int, int]]:
        """User labels (expense_id, category_id), restricted to
        ``self._train_ids`` when a training whitelist is set."""
        labels = labeled_ids_with_categories(self.conn, source="user")
        if self._train_ids is not None:
            labels = [(eid, cid) for eid, cid in labels if eid in self._train_ids]
        return labels

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

        labels = self._user_labels()
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
        labels = self._user_labels()
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

        # Vendor-lookup data (one SQL roundtrip for all expenses, then
        # split into per-expense dicts). Two consumers:
        #   * category_similarity uses the *industry* tag as an extra
        #     token in the lexical-overlap bonus.
        #   * zeroshot (when use_vendor_context is on) uses both the
        #     industry tag AND a truncated slice of the cached summary
        #     to enrich the NLI premise.
        # Fetched eagerly only if at least one consumer wants it.
        vendor_industry: dict[int, str] = {}
        vendor_summary: dict[int, str] = {}
        want_industry = self.cfg.category_similarity.use_vendor_industry
        want_summary = (
            self.cfg.zeroshot.enabled and self.cfg.zeroshot.use_vendor_context
        )
        if want_industry or want_summary:
            try:
                ids_list = list(expense_ids)
                ph_v = ",".join("?" * len(ids_list))
                rows = self.conn.execute(
                    f"""
                    SELECT e.id, vc.industry, vc.summary
                    FROM expenses e
                    LEFT JOIN vendor_cache vc
                      ON vc.counterparty_normalized = e.counterparty_normalized
                    WHERE e.id IN ({ph_v})
                    """,
                    ids_list,
                ).fetchall()
                # Populate both dicts whenever the row carries data,
                # regardless of which flag triggered the fetch. Cheaper
                # than a second roundtrip if both consumers want it, and
                # avoids a subtle bug where zeroshot's premise would
                # lose the industry when only use_vendor_context (not
                # use_vendor_industry) is enabled.
                # Industry tags are migrated to German + filtered to
                # exclude the "Sonstige" (no-signal) sentinel so they
                # don't pollute the cascade with dead tokens.
                from expense_analyzer.enrichment.vendor_web import (
                    is_meaningful_industry,
                    normalize_industry,
                )
                for r in rows:
                    eid_int = int(r["id"])
                    ind = normalize_industry(r["industry"])
                    summ = (r["summary"] or "")
                    if ind and is_meaningful_industry(ind):
                        vendor_industry[eid_int] = ind
                    if summ:
                        vendor_summary[eid_int] = summ
            except Exception:
                # vendor_cache table missing or other transient issue -- skip
                # the boost rather than crash prediction.
                vendor_industry = {}
                vendor_summary = {}

        # ── Pre-loop vectorization ───────────────────────────────────
        # Three perf bugs in the per-row loop are fixed here once:
        #   (1) ``list(df.index).index(eid)`` was O(N²) over the run --
        #       hoist a hash lookup outside the loop.
        #   (2) ``self._sk.predict_proba`` was called once per row even
        #       though sklearn does the whole matrix in one shot at
        #       essentially the same cost.
        #   (3) ``train_vecs @ target_vec`` ran N times in the kNN
        #       stage. ``emb @ train_vecs.T`` does the whole pairwise
        #       similarity matrix in one BLAS matmul.
        # Plus zero-shot is now deferred and batched (see second pass
        # below) instead of called once per row via the transformers
        # pipeline.
        pos_for_eid: dict[int, int] = {int(eid): i for i, eid in enumerate(df.index)}

        proba_all: np.ndarray | None = None
        if self.cfg.classifier.enabled and self._sk is not None:
            X_cache = _build_x(df, emb)
            # One predict_proba on the whole matrix; per-row code below
            # just looks up ``proba_all[target_pos]``.
            proba_all = self._sk.predict_proba(X_cache)

        knn_sim_matrix: np.ndarray | None = None
        if self.cfg.knn.enabled and train_vecs.shape[0] > 0 and emb.shape[0] > 0:
            # Shape (N_test, N_train). For 340×1015×float32 = ~1.3 MB --
            # comfortable even on giant-DB predict-all runs (100k rows
            # × 1k train × 4 bytes ≈ 400 MB worst case; revisit chunking
            # if that ever becomes the typical predict size).
            knn_sim_matrix = emb @ train_vecs.T

        # First pass: collect predictions through stages 1-4. Rows that
        # would have called zero-shot get a placeholder slot and we
        # record their premise for the batched second pass.
        out: list[Prediction | None] = [None] * len(expense_ids)
        zs_pending: list[tuple[int, int, str]] = []  # (out_idx, eid, premise)
        done = 0  # rows fully predicted; progress only ticks on these
        total = len(expense_ids)

        def _tick() -> None:
            if progress_callback is not None:
                progress_callback(done, total)

        for i, eid in enumerate(expense_ids):
            if eid not in df.index:
                out[i] = Prediction(eid, None, 0.0, "unknown", notes="not found / no features")
                done += 1
                _tick()
                continue

            target_pos = pos_for_eid[int(eid)]
            row = df.iloc[target_pos]

            # Stage 1: vendor exact match
            if self.cfg.vendor_exact_match.enabled:
                hit = _vendor_exact_match(
                    self.conn,
                    str(row.get("counterparty_normalized") or ""),
                    self.cfg.vendor_exact_match.agreement_min,
                    restrict_ids=self._train_ids,
                )
                if hit:
                    cid, agreement = hit
                    out[i] = Prediction(eid, cid, agreement, "vendor_exact_match")
                    done += 1
                    _tick()
                    continue

            # Stage 2: k-NN -- use the precomputed sim row.
            if knn_sim_matrix is not None:
                hit = _knn_vote_from_sims(
                    knn_sim_matrix[target_pos],
                    train_y,
                    k=self.cfg.knn.k,
                    agreement_min=self.cfg.knn.agreement_min,
                )
                if hit:
                    cid, conf = hit
                    out[i] = Prediction(eid, cid, conf, "knn")
                    done += 1
                    _tick()
                    continue

            # Stage 3: classifier -- look up the precomputed row.
            if proba_all is not None:
                proba = proba_all[target_pos]
                top = int(np.argmax(proba))
                top_conf = float(proba[top])
                if top_conf >= self.cfg.classifier.confidence_threshold:
                    runner = int(np.argsort(proba)[-2]) if len(proba) > 1 else top
                    out[i] = Prediction(
                        eid,
                        int(self._classes_[top]),
                        top_conf,
                        "classifier",
                        runner_up=int(self._classes_[runner]),
                        runner_up_confidence=float(proba[runner]),
                    )
                    done += 1
                    _tick()
                    continue
                low_conf = top_conf < self.cfg.zeroshot.use_when_confidence_below
            else:
                low_conf = True

            # Stage 4: category similarity (still per-row -- lexical
            # overlap bonus uses the expense text, so vectorising the
            # cosine part alone wouldn't change much).
            if self.cfg.category_similarity.enabled and low_conf:
                expense_text = str(row.get("combined_text") or "")
                industry = vendor_industry.get(eid, "")
                if industry:
                    expense_text = f"{expense_text} {industry}".strip()
                cid, conf = self._category_similarity(
                    emb[target_pos], expense_text=expense_text
                )
                if cid is not None:
                    out[i] = Prediction(eid, cid, conf, "category_similarity")
                    done += 1
                    _tick()
                    continue

            # Stage 5: zero-shot -- DEFER. Real call happens in the
            # batched second pass below so the transformers pipeline
            # gets <batch_size> rows per forward pass instead of 1.
            if self.cfg.zeroshot.enabled and low_conf:
                premise = _build_zeroshot_premise(
                    text=str(row.get("combined_text") or ""),
                    industry=vendor_industry.get(eid, "") if want_summary else "",
                    summary=vendor_summary.get(eid, "") if want_summary else "",
                    summary_max_chars=self.cfg.zeroshot.vendor_summary_max_chars,
                )
                zs_pending.append((i, eid, premise))
                continue  # progress will tick when this row finishes below

            # Nothing fired and zero-shot is off: abstain.
            out[i] = Prediction(eid, None, 0.0, "unknown")
            done += 1
            _tick()

        # ── Second pass: batched zero-shot ────────────────────────────
        if zs_pending:
            texts = [p for _, _, p in zs_pending]
            batch_size = max(1, int(self.cfg.zeroshot.batch_size))
            for start in range(0, len(texts), batch_size):
                chunk_texts = texts[start : start + batch_size]
                chunk_meta = zs_pending[start : start + batch_size]
                preds = self._zeroshot_predict_batch(chunk_texts)
                for (out_idx, eid, _premise), (cid, conf) in zip(
                    chunk_meta, preds, strict=True
                ):
                    if cid is not None:
                        out[out_idx] = Prediction(eid, cid, conf, "zeroshot")
                    else:
                        out[out_idx] = Prediction(eid, None, 0.0, "unknown")
                    done += 1
                    _tick()

        # Sanity: every slot must be filled -- a None here means a
        # stage path silently returned without ``out[i] = ...``.
        return [p if p is not None else Prediction(int(expense_ids[i]), None, 0.0, "unknown")
                for i, p in enumerate(out)]

    # ---- zeroshot -------------------------------------------------------

    def _zeroshot_predict(self, text: str) -> tuple[int | None, float]:
        """Single-row zero-shot. Kept as a thin convenience wrapper so
        any direct caller still works -- the cascade itself routes
        everything through :meth:`_zeroshot_predict_batch` so the
        transformers pipeline gets to amortise GPU launch overhead."""
        results = self._zeroshot_predict_batch([text])
        return results[0] if results else (None, 0.0)

    def _zeroshot_predict_batch(
        self, texts: list[str]
    ) -> list[tuple[int | None, float]]:
        """Run zero-shot NLI on a batch of premises.

        Returns one ``(category_id, confidence)`` tuple per input,
        same order. Empty/whitespace-only premises and any pipeline
        failure surface as ``(None, 0.0)`` -- the cascade then
        treats the row as an abstention.

        Why batching matters: the transformers zero-shot pipeline
        does roughly ``len(texts) * len(candidate_labels)`` NLI
        forward passes. With ~18 categories, ~200 fallback rows per
        Quality-tab fold, and a GPU, a per-row call wastes the
        majority of each forward pass to kernel-launch overhead.
        Passing the whole list lets the pipeline build proper
        mini-batches internally (size controlled by
        ``cfg.zeroshot.batch_size``).
        """
        if not texts:
            return []
        from expense_analyzer.storage.categories import list_categories

        cats = list_categories(self.conn)
        if not cats:
            return [(None, 0.0)] * len(texts)

        # Lazy-init the transformers pipeline. If it can't be built
        # (offline + no HF cache) we cache a no-op marker so the
        # second call doesn't keep retrying.
        if self._zs is None:
            try:
                from transformers import pipeline

                self._zs = pipeline(
                    "zero-shot-classification",
                    model=self.cfg.zeroshot_model,
                )
            except Exception:
                self._zs = lambda *_a, **_kw: None  # type: ignore
                return [(None, 0.0)] * len(texts)

        label_map = {f"{c.name}: {c.description}": c.id for c in cats if c.name}
        labels = list(label_map.keys())
        template = self.cfg.zeroshot.hypothesis_template or "In diesem Text geht es um {}."
        if "{}" not in template:
            template = "In diesem Text geht es um {}."

        # The pipeline raises on empty strings; substitute a single
        # space so the call doesn't blow up the whole batch, then map
        # those rows to (None, 0.0) on the way out.
        empty_mask = [not (t and t.strip()) for t in texts]
        safe_texts = [t if (t and t.strip()) else " " for t in texts]

        batch_size = max(1, int(self.cfg.zeroshot.batch_size))
        try:
            raw = self._zs(
                safe_texts,
                candidate_labels=labels,
                multi_label=False,
                hypothesis_template=template,
                batch_size=batch_size,
            )
        except Exception:
            return [(None, 0.0)] * len(texts)
        # Transformers returns a single dict if called with a single
        # string and a list of dicts otherwise; normalise.
        if isinstance(raw, dict):
            raw = [raw]
        results: list[tuple[int | None, float]] = []
        for is_empty, res in zip(empty_mask, raw, strict=True):
            if is_empty or not res:
                results.append((None, 0.0))
                continue
            top_label = res["labels"][0]
            top_score = float(res["scores"][0])
            results.append((label_map.get(top_label), top_score))
        return results


def _build_zeroshot_premise(
    text: str,
    industry: str,
    summary: str,
    summary_max_chars: int,
) -> str:
    """Compose the NLI premise from the expense text + optional vendor context.

    Format: ``"<expense_text>. Branche: <industry>. <truncated summary>"``
    so the NLI model gets a natural-language run rather than concatenated
    keywords. Each piece is appended only when non-empty so the empty-
    summary / empty-industry / both-empty paths all degrade gracefully
    to the original expense-only premise.
    """
    parts: list[str] = []
    base = (text or "").strip()
    if base:
        parts.append(base if base.endswith(".") else base + ".")
    ind = (industry or "").strip()
    if ind:
        parts.append(f"Branche: {ind}.")
    summ = (summary or "").strip()
    if summ:
        # Hard cap: long DDG snippets shouldn't dominate the premise
        # token budget. Cut on a word boundary when possible so we don't
        # truncate mid-word and confuse the tokenizer.
        if len(summ) > summary_max_chars:
            cut = summ[:summary_max_chars]
            sp = cut.rfind(" ")
            if sp > summary_max_chars * 0.6:
                cut = cut[:sp]
            summ = cut.rstrip(" ,.;:") + "…"
        parts.append(summ)
    return " ".join(parts).strip()
