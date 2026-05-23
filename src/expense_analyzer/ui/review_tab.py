"""Review tab — active-learning labelling queue.

Two kinds of work surface here:
  * *Confirm* — model predicted with medium confidence (CONF_LOW ≤ c < CONF_MED).
    User clicks ✓ or picks a different chip. These are the quick wins.
  * *Needs label* — no prediction or confidence below CONF_LOW.
    The cascade runs once on batch load; user assigns from scratch.

A batch of N expenses is loaded on demand; predictions and neighbor context
are cached in session_state for the lifetime of the batch. Saving commits
labels as source='user' and retrains the cascade inline so the next batch
benefits from improved predictions.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from expense_analyzer.ml.active_learning import get_neighbor_context, pick_candidates
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.categories import add_label, list_categories
from expense_analyzer.storage.database import transaction
from expense_analyzer.ui._shared import get_config, get_conn, get_embedder

_CONF_LOW: float = 0.40
_CONF_MED: float = 0.70

_BATCH_KEY = "review_batch"
_CURSOR_KEY = "review_cursor"
_PENDING_KEY = "review_pending"
_PREDS_KEY = "review_predictions"
_NEIGHBORS_KEY = "review_neighbors"
_ROWS_KEY = "review_expense_rows"
_STRATEGY_KEY = "review_strategy"
_BSIZE_KEY = "review_batch_size"
_SKIPPED_KEY = "review_skipped"

_STRATEGIES = ["uncertainty", "diverse", "mixed"]
_STRATEGY_HELP = (
    "**uncertainty** — label items the model is least sure about "
    "(best for improving accuracy).  \n"
    "**diverse** — spread across the embedding space "
    "(good for cold start with few labels).  \n"
    "**mixed** — round-robin between uncertainty and diverse."
)

_STAGE_LABELS = {
    "vendor_exact_match": "vendor match",
    "knn": "k-NN",
    "classifier": "classifier",
    "category_similarity": "similarity",
    "zeroshot": "zero-shot",
    "db_model": "prior prediction",
    "unknown": "?",
}


# ── Queue helpers ─────────────────────────────────────────────────────────────


def queue_size(conn: sqlite3.Connection) -> int:
    """Total items needing user attention. Used for the tab badge."""
    n, c = _queue_counts(conn)
    return n + c


def _queue_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (needs_label, confirm) counts."""
    needs = conn.execute(
        """
        SELECT COUNT(*) FROM expenses
        WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'user')
          AND (
            id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'model')
            OR id IN (
              SELECT expense_id FROM latest_label
              WHERE label_source = 'model'
                AND (confidence IS NULL OR confidence < ?)
            )
          )
        """,
        (_CONF_LOW,),
    ).fetchone()[0]
    confirm = conn.execute(
        """
        SELECT COUNT(*) FROM latest_label
        WHERE label_source = 'model'
          AND confidence >= ? AND confidence < ?
          AND expense_id NOT IN (
            SELECT DISTINCT expense_id FROM labels WHERE source = 'user'
          )
        """,
        (_CONF_LOW, _CONF_MED),
    ).fetchone()[0]
    return int(needs), int(confirm)


def _auto_coverage(conn: sqlite3.Connection) -> float:
    """% of expenses with a user label or a high-confidence model prediction."""
    total = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    if not total:
        return 0.0
    user = conn.execute(
        "SELECT COUNT(DISTINCT expense_id) FROM labels WHERE source = 'user'"
    ).fetchone()[0]
    high_model = conn.execute(
        "SELECT COUNT(*) FROM latest_label WHERE label_source='model' AND confidence >= ?",
        (_CONF_MED,),
    ).fetchone()[0]
    return min((user + high_model) / total * 100.0, 100.0)


# ── Batch loading ─────────────────────────────────────────────────────────────


def _build_batch(conn, cfg, embedder, n: int, strategy: str) -> list[int]:
    """Up to n expense IDs: confirm items first, then active-learning picks."""
    confirm_rows = conn.execute(
        """
        SELECT ll.expense_id FROM latest_label ll
        WHERE ll.label_source = 'model'
          AND ll.confidence >= ? AND ll.confidence < ?
          AND ll.expense_id NOT IN (
            SELECT DISTINCT expense_id FROM labels WHERE source = 'user'
          )
        ORDER BY ll.confidence ASC
        LIMIT ?
        """,
        (_CONF_LOW, _CONF_MED, n),
    ).fetchall()
    confirm_ids = [int(r["expense_id"]) for r in confirm_rows]

    remaining = n - len(confirm_ids)
    al_ids: list[int] = []
    if remaining > 0:
        try:
            cascade = CategorizationCascade(conn, cfg, embedder)
            cascade.fit()
            al_ids = pick_candidates(
                conn, cfg, embedder, cascade, n=remaining, strategy=strategy
            )
        except Exception:
            pass
        seen = set(confirm_ids)
        al_ids = [x for x in al_ids if x not in seen]

    return (confirm_ids + al_ids)[:n]


def _load_expense_rows(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT e.id, e.buchungsdatum, e.counterparty, e.verwendungszweck,
               e.betrag_cents,
               c.name  AS model_cat_name,
               ll.category_id AS model_cat_id,
               ll.confidence,
               ll.label_source
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c    ON c.id = ll.category_id
        WHERE e.id IN ({ph})
        """,
        ids,
    ).fetchall()
    return {int(r["id"]): dict(r) for r in rows}


def _run_predictions(
    conn, cfg, embedder, ids: list[int], rows: dict[int, dict]
) -> dict[int, dict]:
    """Return a prediction dict per expense_id.

    Reuses the stored model label for 'confirm' items (confidence in
    [CONF_LOW, CONF_MED)); runs the cascade for items below that threshold.
    Each value: {cat_name, cat_id, conf, stage, runner_up, runner_up_conf}.
    """
    preds: dict[int, dict] = {}
    needs_cascade: list[int] = []

    for eid in ids:
        row = rows.get(eid, {})
        conf = row.get("confidence")
        if (
            row.get("label_source") == "model"
            and conf is not None
            and float(conf) >= _CONF_LOW
        ):
            preds[eid] = {
                "cat_name": row.get("model_cat_name") or "",
                "cat_id": row.get("model_cat_id"),
                "conf": float(conf),
                "stage": "db_model",
                "runner_up": None,
                "runner_up_conf": 0.0,
            }
        else:
            needs_cascade.append(eid)

    if needs_cascade:
        try:
            cat_by_id = {c.id: c.name for c in list_categories(conn)}
            cascade = CategorizationCascade(conn, cfg, embedder)
            cascade.fit()
            predictions = cascade.predict_batch(needs_cascade)
            for p in predictions:
                if p.category_id is not None:
                    preds[p.expense_id] = {
                        "cat_name": cat_by_id.get(p.category_id, ""),
                        "cat_id": p.category_id,
                        "conf": p.confidence,
                        "stage": p.stage,
                        "runner_up": p.runner_up,
                        "runner_up_conf": p.runner_up_confidence,
                    }
        except Exception:
            pass

    return preds


def _load_neighbors(conn, embedder, ids: list[int]) -> dict[int, list[dict]]:
    nbrs: dict[int, list[dict]] = {}
    for eid in ids:
        try:
            nbrs[eid] = get_neighbor_context(conn, embedder, eid, n=2)
        except Exception:
            nbrs[eid] = []
    return nbrs


# ── Card rendering ────────────────────────────────────────────────────────────


def _render_card(
    expense_id: int,
    rows: dict,
    preds: dict,
    nbrs: dict,
    cats: list,
    pending: dict,
    skipped: set,
) -> None:
    row = rows.get(expense_id, {})
    pred = preds.get(expense_id)
    neighbors = nbrs.get(expense_id, [])
    pending_cat_id = pending.get(expense_id)
    cat_by_id = {c.id: c for c in cats}

    betrag_cents = int(row.get("betrag_cents") or 0)
    betrag = betrag_cents / 100.0
    amount_color = "#2d8a4e" if betrag >= 0 else "#c0392b"
    date_str = str(row.get("buchungsdatum") or "")[:10] or "—"
    counterparty = row.get("counterparty") or "—"
    vz = row.get("verwendungszweck") or ""

    # ── Expense header ──
    h1, h2, h3 = st.columns([2, 3, 1.5])
    h1.markdown(f"**{date_str}**")
    h2.markdown(f"**{counterparty[:50]}**")
    h3.markdown(
        f"<div style='text-align:right'>"
        f"<span style='color:{amount_color};font-weight:700'>"
        f"{betrag:+,.2f}&nbsp;€</span></div>",
        unsafe_allow_html=True,
    )
    if vz:
        st.caption(vz[:120])

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Model prediction ──
    if pred and pred.get("cat_name"):
        conf_pct = int(pred["conf"] * 100)
        stage_label = _STAGE_LABELS.get(pred.get("stage", ""), pred.get("stage", ""))
        rcat = cat_by_id.get(pred.get("runner_up")) if pred.get("runner_up") else None
        runner_txt = (
            f" &nbsp;·&nbsp; also: **{rcat.name}** {int(pred['runner_up_conf'] * 100)}%"
            if rcat
            else ""
        )
        st.markdown(
            f"🤖 **{pred['cat_name']}** — {conf_pct}%&nbsp;"
            f"<span style='color:#888;font-size:0.82em'>via {stage_label}"
            f"{runner_txt}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No model prediction — assign a category below.")

    # ── Neighbor context ──
    if neighbors:
        lines = []
        for nb in neighbors[:2]:
            sim_pct = int(float(nb.get("similarity", 0)) * 100)
            nb_date = str(nb.get("buchungsdatum", ""))[:10]
            nb_cp = (nb.get("counterparty") or "")[:30]
            nb_cat = nb.get("category_name", "")
            lines.append(
                f"• {nb_cp} ({nb_date}) → **{nb_cat}**&nbsp;·&nbsp;{sim_pct}% similar"
            )
        st.markdown(
            "<div style='color:#888;font-size:0.82em;margin-top:0.1rem'>"
            + "<br>".join(lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Assignment status ──
    if expense_id in skipped:
        st.info("Skipped — will not be saved in this batch.")
    elif pending_cat_id is not None:
        assigned = cat_by_id.get(pending_cat_id)
        if assigned:
            st.success(f"✓ Assigned: **{assigned.name}** — click another chip to change.")

    # ── Category chips (rows of 4) ──
    chips_per_row = 4
    for i in range(0, len(cats), chips_per_row):
        chunk = cats[i : i + chips_per_row]
        cols = st.columns(len(chunk))
        for col, cat in zip(cols, chunk, strict=True):
            is_pending = pending_cat_id == cat.id
            is_predicted = (
                pred is not None
                and pred.get("cat_name") == cat.name
                and pending_cat_id is None
                and expense_id not in skipped
            )
            btn_type = "primary" if (is_pending or is_predicted) else "secondary"
            if col.button(
                cat.name,
                key=f"review_chip_{expense_id}_{cat.id}",
                type=btn_type,
                use_container_width=True,
            ):
                pending[expense_id] = cat.id
                st.session_state[_PENDING_KEY] = pending
                skipped.discard(expense_id)
                st.session_state[_SKIPPED_KEY] = skipped
                batch: list = st.session_state.get(_BATCH_KEY, [])
                cursor = st.session_state.get(_CURSOR_KEY, 0)
                if cursor < len(batch) - 1:
                    st.session_state[_CURSOR_KEY] = cursor + 1
                st.rerun()


# ── Main render ────────────────────────────────────────────────────────────────


def _render_progress(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    if not total:
        return
    user_labeled = conn.execute(
        "SELECT COUNT(DISTINCT expense_id) FROM labels WHERE source='user'"
    ).fetchone()[0]
    pct = user_labeled / total * 100.0
    cov = _auto_coverage(conn)
    prog_col, metric_col = st.columns([4, 1])
    with prog_col:
        st.progress(
            min(pct / 100.0, 1.0),
            text=f"**{user_labeled} / {total}** user-labeled ({pct:.0f}%)",
        )
    with metric_col:
        st.metric(
            "Auto-coverage",
            f"{cov:.0f}%",
            help=(
                "Percentage of expenses with a user label OR a high-confidence "
                f"model prediction (≥{int(_CONF_MED * 100)}%)."
            ),
        )


def render() -> None:
    conn = get_conn()
    cfg = get_config()
    cats = list_categories(conn)

    _render_progress(conn)
    st.divider()

    if not cats:
        st.info("No categories defined yet. Add some in the **Categories** tab first.")
        return

    needs_n, confirm_n = _queue_counts(conn)
    total_q = needs_n + confirm_n

    # ── Batch controls ──
    c_strat, c_size, c_load, c_save, c_discard = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])

    strategy = c_strat.selectbox(
        "Strategy",
        _STRATEGIES,
        index=_STRATEGIES.index(
            st.session_state.get(_STRATEGY_KEY, cfg.active_learning.default_strategy)
        ),
        key=_STRATEGY_KEY,
        help=_STRATEGY_HELP,
    )
    batch_size = int(
        c_size.number_input(
            "Batch size",
            min_value=1,
            max_value=50,
            value=st.session_state.get(_BSIZE_KEY, cfg.active_learning.default_batch_size),
            step=5,
            key=_BSIZE_KEY,
        )
    )

    batch: list[int] = st.session_state.get(_BATCH_KEY, [])
    pending: dict[int, int] = st.session_state.get(_PENDING_KEY, {})
    skipped: set[int] = st.session_state.get(_SKIPPED_KEY, set())
    n_labeled = len(pending)

    load_clicked = c_load.button(
        "Load batch",
        type="primary",
        disabled=(total_q == 0),
        help=(
            f"{total_q} expense(s) in queue "
            f"({confirm_n} to confirm, {needs_n} needing a label)."
        ),
    )
    save_clicked = c_save.button(
        f"💾 Save ({n_labeled})",
        disabled=(n_labeled == 0),
        type="primary" if n_labeled > 0 else "secondary",
        help="Write user labels to DB and retrain the cascade.",
    )
    discard_clicked = c_discard.button(
        "Discard",
        disabled=not batch,
        help="Drop this batch without saving.",
    )

    # ── Handle load ──
    if load_clicked:
        with st.spinner("Loading batch…"):
            embedder = get_embedder()
            new_batch = _build_batch(conn, cfg, embedder, batch_size, strategy)
        if not new_batch:
            st.toast("Nothing new in queue.", icon="ℹ️")
        else:
            expense_rows = _load_expense_rows(conn, new_batch)
            with st.spinner("Running predictions…"):
                embedder = get_embedder()
                new_preds = _run_predictions(conn, cfg, embedder, new_batch, expense_rows)
            with st.spinner("Loading context…"):
                new_nbrs = _load_neighbors(conn, embedder, new_batch)
            st.session_state.update({
                _BATCH_KEY: new_batch,
                _CURSOR_KEY: 0,
                _PENDING_KEY: {},
                _SKIPPED_KEY: set(),
                _PREDS_KEY: new_preds,
                _ROWS_KEY: expense_rows,
                _NEIGHBORS_KEY: new_nbrs,
            })
        st.rerun()

    # ── Handle save ──
    if save_clicked and n_labeled > 0:
        with transaction(conn):
            for eid, cat_id in pending.items():
                add_label(conn, int(eid), int(cat_id), "user")
        cov_before = _auto_coverage(conn)
        retrain_msg = ""
        try:
            with st.spinner("Retraining model…"):
                embedder = get_embedder()
                cascade = CategorizationCascade(conn, cfg, embedder)
                cascade.fit()
                cov_after = _auto_coverage(conn)
                delta = cov_after - cov_before
                retrain_msg = f" Auto-coverage: {cov_after:.0f}%"
                if abs(delta) > 0.5:
                    retrain_msg += f" ({delta:+.0f}%)"
        except Exception as exc:
            retrain_msg = f" (retrain skipped: {exc})"
        st.toast(f"Saved {n_labeled} label(s).{retrain_msg}", icon="✅")
        for k in [_BATCH_KEY, _CURSOR_KEY, _PENDING_KEY, _PREDS_KEY,
                  _ROWS_KEY, _NEIGHBORS_KEY, _SKIPPED_KEY]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Handle discard ──
    if discard_clicked:
        for k in [_BATCH_KEY, _CURSOR_KEY, _PENDING_KEY, _PREDS_KEY,
                  _ROWS_KEY, _NEIGHBORS_KEY, _SKIPPED_KEY]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Empty state ──
    if not batch:
        if total_q == 0:
            st.success(
                "**All caught up!** 🎉  \n"
                "The model is handling expenses automatically. "
                "Import new data or lower the confidence threshold to surface more items."
            )
        else:
            st.info(
                f"**{total_q} expense(s) in queue** "
                f"({confirm_n} to confirm · {needs_n} needing a label).  \n"
                "Click **Load batch** above to start reviewing."
            )
        return

    # ── Active batch ──
    rows: dict = st.session_state.get(_ROWS_KEY, {})
    preds: dict = st.session_state.get(_PREDS_KEY, {})
    nbrs: dict = st.session_state.get(_NEIGHBORS_KEY, {})
    cursor: int = st.session_state.get(_CURSOR_KEY, 0)
    cursor = max(0, min(cursor, len(batch) - 1))
    current_id = batch[cursor]

    # ── Navigation bar ──
    nav_prev, nav_pos, nav_next = st.columns([1, 2, 1])
    if nav_prev.button("◀ Prev", disabled=(cursor == 0), key="review_nav_prev"):
        st.session_state[_CURSOR_KEY] = cursor - 1
        st.rerun()
    nav_pos.markdown(
        f"<p style='text-align:center;margin:0.4rem 0'>"
        f"Card&nbsp;<b>{cursor + 1}</b>&nbsp;/&nbsp;<b>{len(batch)}</b></p>",
        unsafe_allow_html=True,
    )
    if nav_next.button("Next ▶", disabled=(cursor >= len(batch) - 1), key="review_nav_next"):
        st.session_state[_CURSOR_KEY] = cursor + 1
        st.rerun()

    # ── Expense card ──
    with st.container(border=True):
        _render_card(current_id, rows, preds, nbrs, cats, pending, skipped)
        _, skip_col = st.columns([5, 1])
        if skip_col.button("⏭ Skip", key=f"review_skip_{current_id}"):
            skipped.add(current_id)
            st.session_state[_SKIPPED_KEY] = skipped
            pending.pop(current_id, None)
            st.session_state[_PENDING_KEY] = pending
            if cursor < len(batch) - 1:
                st.session_state[_CURSOR_KEY] = cursor + 1
            st.rerun()

    # ── Mini batch status ──
    cat_by_id = {c.id: c for c in cats}
    status_parts = []
    for eid in batch:
        cp = (rows.get(eid, {}).get("counterparty") or str(eid))[:16]
        if eid in pending:
            cat = cat_by_id.get(pending[eid])
            status_parts.append(f"✓ {cp} ({cat.name if cat else '?'})")
        elif eid in skipped:
            status_parts.append(f"⏭ {cp}")
        else:
            status_parts.append(f"○ {cp}")
    st.caption("  ·  ".join(status_parts))
