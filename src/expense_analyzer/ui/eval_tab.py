"""Quality tab: evaluate the categorization cascade on user labels.

Runs leak-free stratified k-fold cross-validation over the hand-labeled
expenses and a stage ablation (cumulative + leave-one-out), then renders
headline metrics, a per-stage contribution chart, a confusion matrix, a
per-category precision/recall/F1 table, the two ablation charts, and a
misclassification table.

The heavy run is gated behind a button and wrapped in an ``st.status``
panel (same pattern as the Review tab's Predict-all flow). Results are
stashed under the ``eval_`` session_state prefix so they survive reruns
and are cleared on account switch by ``clear_tab_state()``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from expense_analyzer.ml.evaluation import ablation, cross_validate
from expense_analyzer.storage.categories import list_categories
from expense_analyzer.ui._components import chart_expander
from expense_analyzer.ui._shared import get_config, get_conn, get_embedder
from expense_analyzer.viz import (
    ablation_cumulative_curve,
    ablation_leave_one_out_bar,
    confusion_matrix_heatmap,
    stage_breakdown_bar,
)

_RESULT_KEY = "eval_result"
_ABLATION_KEY = "eval_ablation"


def render() -> None:
    st.header("Cascade quality")
    st.caption(
        "Cross-validate the categorization cascade on your labeled "
        "expenses and see how much each prediction stage contributes. "
        "All computation is local."
    )

    conn = get_conn()
    cfg = get_config()

    cats = list_categories(conn)
    id_to_name = {c.id: c.name for c in cats}

    labeled = conn.execute(
        """
        SELECT category_id, COUNT(*) AS n FROM (
            SELECT l.expense_id, l.category_id
            FROM labels l
            JOIN (
                SELECT expense_id, MAX(id) AS max_id
                FROM labels WHERE source = 'user'
                GROUP BY expense_id
            ) latest ON l.id = latest.max_id
        ) GROUP BY category_id
        """
    ).fetchall()
    n_labeled = sum(int(r["n"]) for r in labeled)
    n_classes = len([r for r in labeled if int(r["n"]) >= 2])

    if len(cats) < 2 or n_classes < 2 or n_labeled < 4:
        st.info(
            "Not enough labeled data yet. You need at least two categories "
            "with two or more user labels each. Head to the **Review** tab "
            "to label some expenses first."
        )
        return

    max_per_class = max(int(r["n"]) for r in labeled if int(r["n"]) >= 2)
    fold_cap = max(2, min(10, max_per_class))

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        n_folds = st.slider(
            "Cross-validation folds", min_value=2, max_value=fold_cap,
            value=min(5, fold_cap), key="eval_n_folds",
        )
    with c2:
        seed = st.number_input("Seed", min_value=0, value=0, step=1, key="eval_seed")
    with c3:
        include_zeroshot = st.checkbox(
            "Include zero-shot NLI stage (slow — needs the model downloaded)",
            value=cfg.zeroshot.enabled,
            key="eval_zeroshot",
        )

    if st.button("Run evaluation", type="primary", key="eval_run"):
        _run(conn, cfg, int(n_folds), int(seed), include_zeroshot)

    result = st.session_state.get(_RESULT_KEY)
    abl = st.session_state.get(_ABLATION_KEY)
    if result is None:
        return

    _render_results(result, abl, id_to_name)


def _run(conn, cfg, n_folds: int, seed: int, include_zeroshot: bool) -> None:
    eval_cfg = cfg.model_copy(deep=True)
    if not include_zeroshot:
        eval_cfg.zeroshot.enabled = False

    with st.status("Evaluating cascade…", expanded=True) as status:
        status.write(f"loading embedding model `{cfg.embedding_model}`…")
        embedder = get_embedder()

        status.write(f"cross-validating ({n_folds} folds)…")
        cv_progress = st.progress(0, text="fold 0")

        def _cv_cb(done: int, total: int) -> None:
            cv_progress.progress(done / total, text=f"fold {done} / {total}")

        result = cross_validate(
            conn, eval_cfg, embedder, n_folds=n_folds, seed=seed,
            progress_callback=_cv_cb,
        )
        cv_progress.empty()

        if result.n_folds == 0:
            status.update(label=result.notes or "Not enough data.", state="error")
            st.session_state[_RESULT_KEY] = result
            st.session_state[_ABLATION_KEY] = None
            st.rerun()
            return

        status.write("running stage ablation…")
        abl_progress = st.progress(0, text="ablation 0")

        def _abl_cb(done: int, total: int) -> None:
            abl_progress.progress(done / total, text=f"ablation run {done} / {total}")

        abl = ablation(
            conn, eval_cfg, embedder, n_folds=n_folds, seed=seed,
            progress_callback=_abl_cb,
        )
        abl_progress.empty()

        status.update(
            label=(
                f"Done. Accuracy {result.accuracy:.1%} · "
                f"macro-F1 {result.macro_f1:.2f} · "
                f"coverage {result.coverage:.1%} over {result.n_labeled} labels."
            ),
            state="complete",
        )

    st.session_state[_RESULT_KEY] = result
    st.session_state[_ABLATION_KEY] = abl
    st.rerun()


def _render_results(result, abl, id_to_name: dict[int, str]) -> None:
    if result.n_folds == 0:
        st.warning(result.notes or "Not enough labeled data per category.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.accuracy:.1%}")
    m2.metric("Macro-F1", f"{result.macro_f1:.2f}")
    m3.metric("Coverage", f"{result.coverage:.1%}")
    m4.metric("Labels evaluated", f"{result.n_labeled}")

    if result.dropped_singletons:
        st.caption(
            f"{result.dropped_singletons} label(s) excluded: their category "
            "had fewer than two examples, so they can't be cross-validated."
        )

    chart_expander(
        "Per-stage contribution",
        stage_breakdown_bar(result.stage_breakdown),
        expanded=True,
        key="eval_stage_chart",
    )

    names = [id_to_name.get(cid, str(cid)) for cid in result.confusion_labels]
    chart_expander(
        "Confusion matrix",
        confusion_matrix_heatmap(result.confusion, names),
        expanded=False,
        key="eval_confusion_chart",
    )

    per_cat_df = pd.DataFrame(
        [
            {
                "Category": id_to_name.get(pc.category_id, str(pc.category_id)),
                "Precision": round(pc.precision, 3),
                "Recall": round(pc.recall, 3),
                "F1": round(pc.f1, 3),
                "Support": pc.support,
            }
            for pc in result.per_category
        ]
    )
    with st.expander("Per-category metrics", expanded=False):
        st.dataframe(per_cat_df, hide_index=True, width="stretch")

    if abl is not None:
        chart_expander(
            "Ablation — cumulative stages",
            ablation_cumulative_curve(abl.cumulative),
            expanded=False,
            key="eval_abl_cumulative",
        )
        chart_expander(
            "Ablation — leave one stage out",
            ablation_leave_one_out_bar(abl.leave_one_out),
            expanded=False,
            key="eval_abl_loo",
        )

    _render_misclassifications(result, id_to_name)


def _render_misclassifications(result, id_to_name: dict[int, str]) -> None:
    wrong = [r for r in result.records if not r[4]]
    if not wrong:
        st.success("No misclassifications across the held-out folds.")
        return
    eids = [r[0] for r in wrong]
    placeholders = ",".join("?" * len(eids))
    conn = get_conn()
    rows = {
        int(r["id"]): r
        for r in conn.execute(
            f"SELECT id, buchungsdatum, counterparty_normalized "
            f"FROM expenses WHERE id IN ({placeholders})",
            eids,
        ).fetchall()
    }
    df = pd.DataFrame(
        [
            {
                "Date": (rows.get(eid) or {})["buchungsdatum"] if eid in rows else "",
                "Counterparty": (rows.get(eid) or {})["counterparty_normalized"]
                if eid in rows
                else "",
                "True": id_to_name.get(true_cid, str(true_cid)),
                "Predicted": id_to_name.get(pred_cid, "—") if pred_cid is not None else "—",
                "Stage": stage,
            }
            for (eid, true_cid, pred_cid, stage, _correct) in wrong
        ]
    )
    with st.expander(f"Misclassifications ({len(wrong)})", expanded=False):
        st.dataframe(df, hide_index=True, width="stretch")
