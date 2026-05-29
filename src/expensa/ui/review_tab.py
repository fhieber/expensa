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

from expensa.ml.active_learning import get_neighbor_context, pick_candidates
from expensa.ml.classifier import CategorizationCascade
from expensa.storage.categories import add_label, list_categories
from expensa.storage.database import transaction
from expensa.ui._shared import get_config, get_conn, get_embedder
from expensa.utils.colors import readable_text_color

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
# Flag set when a chip click / skip pushes the batch over the line so
# every item is now reviewed (pending OR skipped). Triggers the summary
# screen on the next render instead of leaving the user staring at the
# last card again. Cleared by the summary's "back to cards" button.
_FINISHED_KEY = "review_finished"
# Persisted Predict-all summary string. The status panel that ran the
# cascade lives only as long as its render; we stash the final
# breakdown here so the next render can surface it as a sticky banner
# until the user dismisses it OR starts a new batch via Load batch.
_PREDICT_SUMMARY_KEY = "review_predict_summary"

_STRATEGIES = ["uncertainty", "low-confidence-first", "diverse", "mixed"]
_STRATEGY_HELP = (
    "**uncertainty** — re-predicts every candidate and picks the "
    "ones the cascade is least sure about (best general default).  \n"
    "**low-confidence-first** — surfaces rows whose **stored** "
    "model label already has low confidence (<40%) first, in "
    "ascending order. Skips re-prediction; cheap and direct.  \n"
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
    return sum(_queue_counts(conn))


def review_queue_ids(conn: sqlite3.Connection) -> list[int]:
    """Every expense in the review queue: no user label yet, and either no
    model prediction at all or a model prediction below CONF_MED.

    This is exactly the union of the three queue buckets
    (``needs_label`` + ``low_conf`` + ``confirm``) surfaced by
    :func:`_queue_counts`. Used by the dashboard "To review" deep-link to
    pin these rows in the Data tab.
    """
    rows = conn.execute(
        """
        SELECT id FROM expenses
        WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'user')
          AND (
            id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'model')
            OR id IN (
              SELECT expense_id FROM latest_label
              WHERE label_source = 'model'
                AND (confidence IS NULL OR confidence < ?)
            )
          )
        ORDER BY buchungsdatum DESC, id DESC
        """,
        (_CONF_MED,),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def _queue_counts(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Return ``(needs_label, low_conf, confirm)`` counts.

    Three disjoint buckets covering every expense without a user label:
      * ``needs_label`` -- no model prediction at all yet.
      * ``low_conf``    -- model labelled it with conf < CONF_LOW.
                           The model has a guess but is very unsure;
                           worth a careful review.
      * ``confirm``     -- model labelled it with CONF_LOW ≤ conf
                           < CONF_MED. Often a one-click confirmation.
    """
    needs = conn.execute(
        """
        SELECT COUNT(*) FROM expenses
        WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'user')
          AND id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'model')
        """
    ).fetchone()[0]
    low_conf = conn.execute(
        """
        SELECT COUNT(*) FROM latest_label
        WHERE label_source = 'model'
          AND (confidence IS NULL OR confidence < ?)
          AND expense_id NOT IN (
            SELECT DISTINCT expense_id FROM labels WHERE source = 'user'
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
    return int(needs), int(low_conf), int(confirm)


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
        SELECT e.id, e.buchungsdatum,
               e.counterparty,
               e.zahlungspflichtiger, e.verwendungszweck,
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
    out: dict[int, dict] = {}
    for r in rows:
        d = dict(r)
        out[int(d["id"])] = d
    return out


def _run_predictions(
    conn,
    cfg,
    embedder,
    ids: list[int],
    rows: dict[int, dict],
    *,
    progress_callback=None,
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
            predictions = cascade.predict_batch(
                needs_cascade, progress_callback=progress_callback,
            )
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


def _ids_needing_prediction(conn: sqlite3.Connection) -> list[int]:
    """Every expense the "Predict all unlabeled" button should hit.

    Skips rows the user has already confirmed (those carry a
    ``source='user'`` label) and rows we're already confident about
    (latest label is ``source='model'`` with confidence ≥
    :data:`_CONF_MED`). Re-predicts everything else, including rows
    whose only existing label is a low-confidence model prediction --
    a fresh cascade fit is usually better than a stale one, and
    ``add_label`` appends rather than overwriting so the audit trail
    is preserved.

    The returned set is exactly the union of the "to confirm" and
    "needs label" buckets shown in the Review queue counter.
    """
    rows = conn.execute(
        """
        SELECT id FROM expenses
        WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'user')
          AND (
            id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source = 'model')
            OR id IN (
              SELECT expense_id FROM latest_label
              WHERE label_source = 'model'
                AND (confidence IS NULL OR confidence < ?)
            )
          )
        ORDER BY id
        """,
        (_CONF_MED,),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def _predict_all_unlabeled(conn, cfg) -> None:
    """Run the cascade across every expense in
    :func:`_ids_needing_prediction` and persist each result as a
    ``source='model'`` label. Mirrors the CLI's ``expensa predict`` but
    with a Streamlit status panel + progress bar so the user can watch
    it land. After commit, predictions in the confirm-confidence
    window automatically populate the Review queue's "to confirm"
    bucket -- the whole point of the action.
    """
    from collections import Counter

    ids = _ids_needing_prediction(conn)
    if not ids:
        st.toast(
            "Nothing to predict -- every expense is user-labeled or "
            "already high-confidence.",
            icon="ℹ️",
        )
        return

    with st.status(
        f"Auto-labeling {len(ids)} unlabeled expense(s)…", expanded=True,
    ) as status:
        status.write(f"loading embedding model `{cfg.embedding_model}`…")
        embedder = get_embedder()
        status.write("fitting cascade on the latest user labels…")
        cascade = CategorizationCascade(conn, cfg, embedder)
        try:
            cascade.fit()
        except Exception as e:
            status.write(f"  fit skipped: {e}")
        status.write(f"running cascade on {len(ids)} record(s)…")
        progress = st.progress(0, text=f"0 / {len(ids)}")

        def _cb(done: int, total: int) -> None:
            progress.progress(done / total, text=f"{done} / {total}")

        preds = cascade.predict_batch(ids, progress_callback=_cb)
        progress.empty()

        status.write("persisting model labels…")
        n_persisted = n_high = n_confirm = n_low = 0
        with transaction(conn):
            for p in preds:
                if p.category_id is None:
                    continue
                add_label(
                    conn, p.expense_id, p.category_id, "model",
                    confidence=p.confidence,
                )
                n_persisted += 1
                conf = float(p.confidence or 0.0)
                if conf >= _CONF_MED:
                    n_high += 1
                elif conf >= _CONF_LOW:
                    n_confirm += 1
                else:
                    n_low += 1

        stages = Counter(p.stage for p in preds)
        stage_str = " · ".join(f"{k}={v}" for k, v in stages.most_common())
        n_unknown = len(preds) - n_persisted
        unknown_str = f" · unpredicted={n_unknown}" if n_unknown else ""
        summary_label = (
            f"Persisted {n_persisted} model label(s)."
            f"  high (auto-covered): {n_high}"
            f" · to confirm: {n_confirm}"
            f" · low confidence: {n_low}"
            f"{unknown_str}.  Cascade stages — {stage_str}."
        )
        status.update(label=summary_label, state="complete")
    # Stash the summary in session_state so the next render can show
    # it as a persistent banner. The st.status panel lives only as
    # long as its render -- after st.rerun() it'd otherwise vanish
    # before the user can read the breakdown.
    st.session_state[_PREDICT_SUMMARY_KEY] = summary_label
    st.rerun()


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
    payer = row.get("zahlungspflichtiger") or ""
    vz = row.get("verwendungszweck") or ""

    # ── Expense header: date + amount ──
    h_date, _spacer, h_amt = st.columns([2, 4, 2])
    h_date.markdown(f"**{date_str}**")
    h_amt.markdown(
        f"<div style='text-align:right'>"
        f"<span style='color:{amount_color};font-weight:700;font-size:1.6rem'>"
        f"{betrag:+,.2f}&nbsp;€</span></div>",
        unsafe_allow_html=True,
    )

    # ── Labelled field rows ──
    # Explicit field labels because just showing the values inline
    # (the previous layout) leaves the user guessing which string is
    # which. Payer is only shown when distinct from counterparty (in
    # most expense rows they're the same; in income rows the payer is
    # the interesting field).
    _LABEL_STYLE = "color:#888;font-size:0.8em;font-weight:500"
    def _field(label: str, value: str) -> None:
        st.markdown(
            f"<div style='margin:0.15rem 0'>"
            f"<span style='{_LABEL_STYLE}'>{label}</span> "
            f"<span style='font-weight:600'>{value}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _field("Counterparty", counterparty[:80])
    if payer and payer.strip() and payer != counterparty:
        _field("Payer", payer[:80])
    if vz:
        _field("Verwendungszweck", vz[:200])

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
    # The per-card "✓ Assigned" tooltip was noisy and only ever showed
    # on the last visited card (we auto-advance immediately after the
    # click). The mini batch status strip at the bottom of the tab now
    # turns each item's slug green when it's pending, which gives the
    # same feedback without the in-card noise. Skipped status stays
    # since skipped cards aren't visited automatically.
    if expense_id in skipped:
        st.info("Skipped — will not be saved in this batch.")

    # ── Category chips (rows of 4) ──
    # Visual semantics (after the colour pass):
    #   * Every chip is painted in the user-chosen category colour with
    #     auto-picked legible foreground text.
    #   * ✓ prefix on the user-confirmed chip (the pending label that
    #     Save will commit).
    #   * 🤖 prefix on the model's prediction chip -- same colour /
    #     border as the rest, just marked so the user can spot the
    #     suggestion without it looking pre-selected.
    #
    # How the per-chip background colour gets applied: Streamlit's
    # `st.button` doesn't expose a `style=` arg, so we inject a CSS
    # rule per category and a tiny invisible marker `<span>` right
    # before each chip's button. The CSS uses `:has()` + the adjacent-
    # sibling combinator to paint the matching button. `:has()` is
    # standard CSS since ~2023, supported in every browser we'd
    # realistically see on a localhost Streamlit instance.
    _emit_chip_colour_css(cats)

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
            if is_pending:
                label = f"✓ {cat.name}"
            elif is_predicted:
                label = f"🤖 {cat.name}"
            else:
                label = cat.name
            # Key encodes the cat id with a trailing `end` token so the
            # CSS rule emitted by `_emit_chip_colour_css` can substring-
            # match it without confusing cat 5 with cat 50 / 500 (which
            # would otherwise share the prefix). Streamlit attaches a
            # `st-key-<key>` class to the widget's wrapper div, which
            # is what the selector targets.
            if col.button(
                label,
                key=f"review_chip_{expense_id}_cat{cat.id}end",
                type="secondary",
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
                elif all(e in pending or e in skipped for e in batch):
                    # We just reviewed the last unreviewed card --
                    # surface the summary on the next render.
                    st.session_state[_FINISHED_KEY] = True
                st.rerun()


def _emit_chip_colour_css(cats: list) -> None:
    """Inject the CSS that paints every category chip in its chosen
    colour, plus a focus ring for keyboard navigation.

    Streamlit 1.36+ attaches a ``st-key-<key>`` class to each widget's
    wrapper ``<div>``. Our chip keys encode the cat id with a trailing
    ``end`` token (e.g. ``review_chip_42_cat5end``) so a substring
    selector ``[class*="_cat5end"]`` matches cat 5 but NOT cat 50 /
    500 (the trailing ``end`` is the delimiter that prevents prefix
    collisions).
    """
    rules: list[str] = []
    for cat in cats:
        bg = cat.color or "#888888"
        fg = readable_text_color(bg)
        sel = f'[class*="_cat{cat.id}end"] button'
        # `!important` because Streamlit's own primary/secondary rules
        # use specificity we can't reliably out-weight without it.
        rules.append(
            f"{sel} {{"
            f"  background-color: {bg} !important;"
            f"  color: {fg} !important;"
            f"  border-color: {bg} !important;"
            f"}}"
        )
        # Hover state: keep the colour, brighten slightly so the user
        # gets a hover affordance.
        rules.append(
            f"{sel}:hover {{"
            f"  background-color: {bg} !important;"
            f"  color: {fg} !important;"
            f"  border-color: {bg} !important;"
            f"  filter: brightness(1.15);"
            f"}}"
        )
    # Focus ring: bright, contrasty outline so the user can see which
    # chip the keyboard cursor is on, regardless of the chip's
    # category colour underneath. `box-shadow` instead of `outline`
    # because Streamlit's own focus styles clobber `outline`.
    rules.append(
        '[class*="st-key-review_chip_"] button:focus,'
        '[class*="st-key-review_chip_"] button:focus-visible {'
        '  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.85),'
        '              0 0 0 5px rgba(0, 0, 0, 0.6) !important;'
        '  outline: none !important;'
        '}'
    )
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


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

    # Sticky Predict-all summary banner. Lives until the user clicks
    # the ✕ to dismiss it OR until Load batch supersedes it. Keeps
    # the stage / confidence-bucket breakdown visible long enough to
    # actually read.
    if (predict_summary := st.session_state.get(_PREDICT_SUMMARY_KEY)):
        sb_cols = st.columns([10, 1], vertical_alignment="center")
        with sb_cols[0]:
            st.success(predict_summary, icon="✅")
        with sb_cols[1]:
            if st.button("✕", key="review_dismiss_predict_summary",
                          help="Dismiss this summary."):
                st.session_state.pop(_PREDICT_SUMMARY_KEY, None)
                st.rerun()

    needs_n, low_conf_n, confirm_n = _queue_counts(conn)
    total_q = needs_n + low_conf_n + confirm_n
    queue_breakdown = (
        f"{confirm_n} to confirm · "
        f"{low_conf_n} low-confidence · "
        f"{needs_n} needs label"
    )

    # ── Batch controls ──
    # vertical_alignment="bottom" lines up every column's *bottom* edge,
    # so the labelled widgets (selectbox + number_input) and the
    # label-less buttons share the same baseline without needing a
    # manual spacer above each button.
    c_strat, c_size, c_load, c_save, c_discard = st.columns(
        [2.5, 1.5, 1.5, 1.5, 1.5], vertical_alignment="bottom",
    )

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
        help=f"{total_q} expense(s) in queue  ({queue_breakdown}).",
        use_container_width=True,
    )
    save_clicked = c_save.button(
        f"💾 Save ({n_labeled})",
        disabled=(n_labeled == 0),
        type="primary" if n_labeled > 0 else "secondary",
        help="Write user labels to DB and retrain the cascade.",
        use_container_width=True,
    )
    discard_clicked = c_discard.button(
        "Discard",
        disabled=not batch,
        help="Drop this batch without saving.",
        use_container_width=True,
    )

    # ── Handle load ──
    if load_clicked:
        # Starting a fresh review batch supersedes any prior Predict-
        # all summary -- drop the sticky banner so the screen isn't
        # cluttered with stale stats during card review.
        st.session_state.pop(_PREDICT_SUMMARY_KEY, None)
        # Mirrors the Data tab's auto-label flow: one st.status panel
        # so the user sees each phase fill, with a progress bar on the
        # cascade pass (which is the slowest step). Same shape as
        # data_tab.py:_autolabel_predictions for consistency.
        with st.status("Loading batch…", expanded=True) as status:
            status.write(f"loading embedding model `{cfg.embedding_model}`…")
            embedder = get_embedder()
            status.write(
                f"picking up to {batch_size} candidate(s) "
                f"(strategy: {strategy})…"
            )
            new_batch = _build_batch(conn, cfg, embedder, batch_size, strategy)
            if not new_batch:
                status.update(
                    label="Nothing new in queue.",
                    state="complete",
                )
            else:
                expense_rows = _load_expense_rows(conn, new_batch)
                status.write(f"running cascade on {len(new_batch)} record(s)…")
                progress = st.progress(0, text=f"0 / {len(new_batch)}")

                def _cb(done: int, total: int) -> None:
                    progress.progress(done / total, text=f"{done} / {total}")

                new_preds = _run_predictions(
                    conn, cfg, embedder, new_batch, expense_rows,
                    progress_callback=_cb,
                )
                progress.empty()
                status.write("loading neighbor context…")
                new_nbrs = _load_neighbors(conn, embedder, new_batch)
                st.session_state.update({
                    _BATCH_KEY: new_batch,
                    _CURSOR_KEY: 0,
                    _PENDING_KEY: {},
                    _SKIPPED_KEY: set(),
                    _PREDS_KEY: new_preds,
                    _ROWS_KEY: expense_rows,
                    _NEIGHBORS_KEY: new_nbrs,
                    _FINISHED_KEY: False,
                })
                status.update(
                    label=f"Loaded batch of {len(new_batch)} record(s).",
                    state="complete",
                )
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
                  _ROWS_KEY, _NEIGHBORS_KEY, _SKIPPED_KEY, _FINISHED_KEY]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Handle discard ──
    if discard_clicked:
        for k in [_BATCH_KEY, _CURSOR_KEY, _PENDING_KEY, _PREDS_KEY,
                  _ROWS_KEY, _NEIGHBORS_KEY, _SKIPPED_KEY, _FINISHED_KEY]:
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
                f"({queue_breakdown}).  \n"
                "Click **Load batch** above to start reviewing one card at a time."
            )
            # Bulk autolabel CTA. Only meaningful when there's actually
            # work to do, so we gate it on total_q > 0 (this branch).
            # Lives in the empty-state block because running it mid-
            # batch would invalidate the displayed predictions; running
            # it BEFORE a batch is the sane workflow.
            ctab_btn, ctab_help = st.columns([1.5, 6],
                                              vertical_alignment="center")
            if ctab_btn.button(
                f"🤖 Predict all {total_q} unlabeled",
                type="primary",
                key="review_predict_all_btn",
                use_container_width=True,
            ):
                _predict_all_unlabeled(conn, cfg)
            ctab_help.caption(
                "One-shot: runs the cascade across every unlabeled "
                "expense + every existing low-confidence prediction, "
                "persists the result as model labels. Mid-confidence "
                "predictions land in the **to confirm** queue above so "
                "you can batch-review them next."
            )
        return

    # ── Active batch ──
    rows: dict = st.session_state.get(_ROWS_KEY, {})
    preds: dict = st.session_state.get(_PREDS_KEY, {})
    nbrs: dict = st.session_state.get(_NEIGHBORS_KEY, {})
    cursor: int = st.session_state.get(_CURSOR_KEY, 0)
    cursor = max(0, min(cursor, len(batch) - 1))

    # ── Summary screen takes over once every batch item is reviewed ──
    # Flag is set by the last chip click / skip in the batch (see the
    # chip + skip handlers). The user can dismiss it to revisit any
    # card; reviewing items again won't re-trigger until they leave the
    # summary and come back via another last-item click.
    if st.session_state.get(_FINISHED_KEY):
        _render_batch_summary(batch, rows, pending, skipped, cats, n_labeled)
        _render_status_strip(batch, rows, pending, skipped, cats)
        _inject_keyboard_shortcuts()
        return

    current_id = batch[cursor]

    # ── Navigation bar ──
    # Prev | Card N/M | Next | Skip -- Skip lives here (not inside the
    # card) so it sits next to its semantic sibling Next, and so the
    # card body is just the data, not a mix of data + navigation.
    nav_prev, nav_pos, nav_next, nav_skip = st.columns([1, 2, 1, 1])
    if nav_prev.button("◀ Prev", disabled=(cursor == 0),
                       key="review_nav_prev", use_container_width=True):
        st.session_state[_CURSOR_KEY] = cursor - 1
        st.rerun()
    nav_pos.markdown(
        f"<p style='text-align:center;margin:0.4rem 0'>"
        f"Card&nbsp;<b>{cursor + 1}</b>&nbsp;/&nbsp;<b>{len(batch)}</b></p>",
        unsafe_allow_html=True,
    )
    if nav_next.button("Next ▶", disabled=(cursor >= len(batch) - 1),
                       key="review_nav_next", use_container_width=True):
        st.session_state[_CURSOR_KEY] = cursor + 1
        st.rerun()
    if nav_skip.button("⏭ Skip", key=f"review_skip_{current_id}",
                       use_container_width=True):
        skipped.add(current_id)
        st.session_state[_SKIPPED_KEY] = skipped
        pending.pop(current_id, None)
        st.session_state[_PENDING_KEY] = pending
        if cursor < len(batch) - 1:
            st.session_state[_CURSOR_KEY] = cursor + 1
        elif all(e in pending or e in skipped for e in batch):
            st.session_state[_FINISHED_KEY] = True
        st.rerun()

    # ── Expense card ──
    with st.container(border=True):
        _render_card(current_id, rows, preds, nbrs, cats, pending, skipped)

    _render_status_strip(batch, rows, pending, skipped, cats)
    _inject_keyboard_shortcuts()


# ── Batch summary + status strip + keyboard shortcuts ─────────────────────


def _render_batch_summary(
    batch: list[int],
    rows: dict[int, dict],
    pending: dict[int, int],
    skipped: set[int],
    cats: list,
    n_labeled: int,
) -> None:
    """Final screen shown once every card in the batch is reviewed.

    Replaces the per-card view so the user isn't stranded staring at
    the last reviewed card. Three primary actions:
      * Save (top action bar above is the primary commit path; we
        nudge the user towards it here).
      * Discard (top action bar above).
      * "← Back to cards" pops the finished flag so the user can
        return to any card and re-pick.
    """
    n_skipped = sum(1 for eid in batch if eid in skipped)
    with st.container(border=True):
        st.markdown("### ✅ Batch reviewed")
        st.markdown(
            f"You assigned a category to **{n_labeled}** of "
            f"**{len(batch)}** expense(s) "
            f"({n_skipped} skipped).  \n"
            "Click **💾 Save** at the top to commit these labels and "
            "retrain the model. Or **← Back to cards** to change any "
            "of them first."
        )
        back_col, _spacer = st.columns([1, 5])
        if back_col.button("← Back to cards", key="review_back_to_cards"):
            st.session_state.pop(_FINISHED_KEY, None)
            # Park the cursor on the last card so the user resumes where
            # the auto-advance left them.
            st.session_state[_CURSOR_KEY] = len(batch) - 1
            st.rerun()


def _render_status_strip(
    batch: list[int],
    rows: dict[int, dict],
    pending: dict[int, int],
    skipped: set[int],
    cats: list,
) -> None:
    """Per-batch status caption coloured by review state."""
    cat_by_id = {c.id: c for c in cats}
    spans = []
    for eid in batch:
        cp = (rows.get(eid, {}).get("counterparty") or str(eid))[:16]
        if eid in pending:
            cat = cat_by_id.get(pending[eid])
            spans.append(
                "<span style='color:#2d8a4e;font-weight:600'>"
                f"✓ {cp} ({cat.name if cat else '?'})</span>"
            )
        elif eid in skipped:
            spans.append(
                f"<span style='color:#a07b00'>⏭ {cp}</span>"
            )
        else:
            spans.append(
                f"<span style='color:#888'>○ {cp}</span>"
            )
    st.markdown(
        "<div style='font-size:0.78rem;margin-top:0.4rem;line-height:1.4'>"
        + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(spans)
        + "</div>",
        unsafe_allow_html=True,
    )


def _inject_keyboard_shortcuts() -> None:
    """Keyboard navigation for the active card:
      * **← / →** focus the previous / next category chip in the grid.
      * **↑ / ↓** jump up / down one row (chips render 4 per row).
      * **Enter** activates the focused chip (native button behaviour).
      * **S** clicks the Skip button (no chip activation needed).
      * Card-level Prev / Next nav is intentionally *not* bound -- the
        user only ever wants to advance after a chip pick or skip.

    On every render: if no chip is focused yet, auto-focus the model's
    prediction (the one with the 🤖 prefix). Falls back to the first
    chip when there's no prediction.

    Same-origin caveat applies: the iframe rendered by
    ``st.components.v1.html`` accesses ``window.parent.document``,
    which works under ``streamlit run`` on localhost and any
    deployment where the iframe shares an origin with the app shell.
    """
    import streamlit.components.v1 as _components

    _components.html(
        """
        <script>
        (function() {
          const parent = window.parent;
          if (!parent || !parent.document) return;
          const doc = parent.document;
          const PER_ROW = 4;

          // Chip buttons in DOM order (= rendering order).
          function chips() {
            return Array.from(
              doc.querySelectorAll('[class*="st-key-review_chip_"] button')
            );
          }
          function currentIndex(cs) {
            const i = cs.indexOf(doc.activeElement);
            return i < 0 ? 0 : i;
          }
          function focusAt(i) {
            const cs = chips();
            if (!cs.length) return;
            if (i < 0) i = 0;
            if (i >= cs.length) i = cs.length - 1;
            cs[i].focus();
          }
          function clickByLabel(needle) {
            const buttons = doc.querySelectorAll('button');
            for (const b of buttons) {
              const txt = (b.innerText || '').trim();
              if (txt.indexOf(needle) !== -1 && !b.disabled) {
                b.click();
                return true;
              }
            }
            return false;
          }

          // Bind the listener exactly once per browser session; the
          // `parent` object survives Streamlit reruns so we can stash
          // a sentinel flag on it.
          //
          // CAPTURE PHASE (third arg `true`) is critical. Streamlit's
          // tab strip and other inner widgets bind their own keydown
          // listeners and frequently call stopPropagation(), which
          // means a normal bubbling listener on `document` never sees
          // the arrow keys. Capturing fires our handler before any
          // inner listener can swallow the event.
          if (!parent._reviewKbCardBound) {
            parent._reviewKbCardBound = true;
            doc.addEventListener('keydown', function(e) {
              const t = e.target;
              if (t && t.matches &&
                  t.matches('input, textarea, select, [contenteditable="true"]')) {
                return;
              }
              const cs = chips();
              if (!cs.length) {
                // No chips on this render -- empty state / summary.
                return;
              }
              let handled = true;
              if (e.key === 'ArrowRight') {
                focusAt(currentIndex(cs) + 1);
              } else if (e.key === 'ArrowLeft') {
                focusAt(currentIndex(cs) - 1);
              } else if (e.key === 'ArrowDown') {
                focusAt(currentIndex(cs) + PER_ROW);
              } else if (e.key === 'ArrowUp') {
                focusAt(currentIndex(cs) - PER_ROW);
              } else if (e.key === 's' || e.key === 'S') {
                clickByLabel('⏭ Skip');
              } else {
                handled = false;
              }
              if (handled) {
                // Stop both the default action (page scroll on
                // arrows) AND propagation to any inner widget that
                // would otherwise reinterpret the arrow keys (e.g.
                // tab navigation, AgGrid).
                e.preventDefault();
                e.stopPropagation();
              }
              // Enter: leave it to the browser. A focused button
              // activates on Enter natively, which triggers the chip's
              // Python click handler.
            }, true);
          }

          // On every render: park focus on the predicted chip (or
          // first chip) IF the user doesn't already have something
          // useful focused. 400ms because chips don't exist in the
          // DOM until Streamlit finishes mounting the new card +
          // injecting AgGrid; shorter delays raced empty.
          setTimeout(function() {
            const cs = chips();
            if (!cs.length) return;
            const active = doc.activeElement;
            // Already focused on a chip? Leave it alone.
            if (active && cs.indexOf(active) !== -1) return;
            // User has clicked into a text input etc? Don't steal.
            if (active && active.matches &&
                active.matches('input, textarea, select, [contenteditable="true"]')) {
              return;
            }
            // Find the predicted chip (🤖 prefix); fall back to 0.
            let initial = 0;
            for (let i = 0; i < cs.length; i++) {
              if ((cs[i].innerText || '').includes('🤖')) { initial = i; break; }
            }
            cs[initial].focus();
          }, 400);
        })();
        </script>
        """,
        height=0,
    )

