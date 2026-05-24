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

from expense_analyzer.ml import eval_cache
from expense_analyzer.ml.eval_report import ReportContext, build_pdf, default_filename
from expense_analyzer.ml.evaluation import (
    ablation,
    cross_validate,
    fold_sizes,
    planned_ablation_runs,
)
from expense_analyzer.storage.categories import list_categories
from expense_analyzer.ui._components import chart_expander
from expense_analyzer.ui._shared import (
    get_active_account,
    get_config,
    get_conn,
    get_embedder,
)
from expense_analyzer.viz import (
    ablation_cumulative_curve,
    ablation_leave_one_out_bar,
    confusion_matrix_heatmap,
    stage_breakdown_bar,
)

_RESULT_KEY = "eval_result"
_ABLATION_KEY = "eval_ablation"
_RUN_META_KEY = "eval_run_meta"  # snapshot of (seed, include_zeroshot) used for the run
_SAVED_AT_KEY = "eval_saved_at"  # ISO timestamp of the run that produced the cached result


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

    # Stratified k-fold is bounded by the SMALLEST eligible class: each
    # class needs at least one example per fold to appear in every split.
    min_eligible_class = min(int(r["n"]) for r in labeled if int(r["n"]) >= 2)
    fold_cap = max(2, min(10, min_eligible_class))

    # vertical_alignment="bottom" so the checkbox baseline lines up with
    # the slider track and number_input field on the same row.
    c1, c2, c3 = st.columns([2, 1, 2], vertical_alignment="bottom")
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

    # Hydrate session state from the on-disk cache on first render of
    # this session (or after an account switch, which clears the
    # session_state but leaves the per-account cache file alone).
    if st.session_state.get(_RESULT_KEY) is None:
        cached = eval_cache.load(get_active_account().data_dir)
        if cached is not None:
            st.session_state[_RESULT_KEY] = cached.result
            st.session_state[_ABLATION_KEY] = cached.ablation
            st.session_state[_RUN_META_KEY] = cached.meta
            st.session_state[_SAVED_AT_KEY] = cached.saved_at.isoformat(timespec="minutes")

    result = st.session_state.get(_RESULT_KEY)
    abl = st.session_state.get(_ABLATION_KEY)
    if result is None:
        return

    saved_at = st.session_state.get(_SAVED_AT_KEY)
    if saved_at:
        # Format ISO timestamp -> "2026-05-24 14:32" without parsing.
        pretty = saved_at.replace("T", " ")
        st.caption(f"📌 Showing results from **{pretty}**. Re-run above to refresh.")
    _render_results(result, abl, id_to_name)


def _run(conn, cfg, n_folds: int, seed: int, include_zeroshot: bool) -> None:
    eval_cfg = cfg.model_copy(deep=True)
    if not include_zeroshot:
        eval_cfg.zeroshot.enabled = False

    with st.status("Evaluating cascade…", expanded=True) as status:
        status.write(f"loading embedding model `{cfg.embedding_model}`…")
        embedder = get_embedder()

        # Pre-compute the fold partition so the user sees label counts
        # before any model fires. Cheap (no model calls).
        n_kept, n_dropped, fsizes = fold_sizes(conn, n_folds, seed)
        if fsizes:
            train_min, train_max = min(t for t, _ in fsizes), max(t for t, _ in fsizes)
            test_min, test_max = min(t for _, t in fsizes), max(t for _, t in fsizes)
            train_txt = f"{train_min}" if train_min == train_max else f"{train_min}–{train_max}"
            test_txt = f"{test_min}" if test_min == test_max else f"{test_min}–{test_max}"
            status.write(
                f"data: **{n_kept}** labels usable for CV "
                f"({n_dropped} singletons excluded) — per fold "
                f"~{train_txt} train / ~{test_txt} test rows."
            )
        else:
            status.write(f"data: {n_kept} labels, but no class has ≥2 examples.")

        status.write(f"cross-validating ({n_folds} folds)…")
        cv_progress = st.progress(0, text="starting…")

        def _cv_cb(done: int, total: int) -> None:
            train_n, test_n = fsizes[done - 1] if done - 1 < len(fsizes) else (0, 0)
            cv_progress.progress(
                done / total,
                text=f"fold {done}/{total} — train={train_n}, test={test_n}",
            )

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

        # Plan the ablation runs so the progress text can name each one.
        # Each run does n_folds CV fits internally.
        planned = planned_ablation_runs(eval_cfg)
        status.write(
            f"running stage ablation — **{len(planned)}** runs × "
            f"{result.n_folds} folds = {len(planned) * result.n_folds} cascade fits."
        )
        abl_progress = st.progress(0, text="starting…")

        def _abl_cb(done: int, total: int) -> None:
            label = planned[done - 1][0] if done - 1 < len(planned) else ""
            abl_progress.progress(
                done / total,
                text=f"ablation {done}/{total} — {label}",
            )

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

    from datetime import datetime

    meta = {"seed": seed, "include_zeroshot": include_zeroshot}
    st.session_state[_RESULT_KEY] = result
    st.session_state[_ABLATION_KEY] = abl
    st.session_state[_RUN_META_KEY] = meta
    st.session_state[_SAVED_AT_KEY] = datetime.now().isoformat(timespec="minutes")
    # Persist to <data_dir>/cache/eval_latest.pkl so the result survives
    # UI restarts. Failures are non-fatal -- the user keeps the in-
    # memory result either way.
    try:
        eval_cache.save(get_active_account().data_dir, result, abl, meta)
    except Exception as e:  # noqa: BLE001
        st.warning(f"could not persist eval cache: {e}")
    st.rerun()


def _render_results(result, abl, id_to_name: dict[int, str]) -> None:
    if result.n_folds == 0:
        st.warning(result.notes or "Not enough labeled data per category.")
        return

    covered = (
        "—" if result.accuracy_covered != result.accuracy_covered  # NaN guard
        else f"{result.accuracy_covered:.1%}"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{result.accuracy:.1%}", help="Over all held-out rows; abstentions count as errors.")
    m2.metric("Accuracy (covered)", covered, help="Over only the rows the cascade actually predicted (excludes abstentions).")
    m3.metric("Coverage", f"{result.coverage:.1%}", help="Fraction of rows that got a concrete prediction (vs. abstaining).")
    m4, m5, m6 = st.columns(3)
    m4.metric("Macro-F1", f"{result.macro_f1:.2f}", help="Unweighted mean of per-category F1 — surfaces weak rare categories.")
    m5.metric("Weighted-F1", f"{result.weighted_f1:.2f}", help="Per-category F1 weighted by support — closer to the overall hit rate.")
    m6.metric("Labels evaluated", f"{result.n_labeled}")

    st.caption(f"{result.n_folds}-fold stratified cross-validation.")
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

    _render_pdf_export(result, abl, id_to_name)


def _render_pdf_export(result, abl, id_to_name: dict[int, str]) -> None:
    """Bottom-of-tab "Export as PDF" affordance.

    Builds the report lazily on click so the kaleido / reportlab
    imports (and the per-figure PNG rendering) only fire when the
    user actually wants the file. Missing-deps surface as an inline
    error rather than crashing the tab.
    """
    st.divider()
    # Small breathing room so the button isn't hugging the divider line
    # above -- with the column row going edge-to-edge, the primary
    # button's top border was reading as part of the divider.
    st.write("")
    cfg = get_config()
    account = get_active_account()
    meta = st.session_state.get(_RUN_META_KEY) or {}
    col_btn, col_hint = st.columns([1, 3], vertical_alignment="center")
    with col_btn:
        build_clicked = st.button(
            "📄 Build PDF report",
            key="eval_build_pdf",
            help="Generate a single-file PDF with the headline metrics, "
                 "per-stage chart, confusion matrix, per-category table, "
                 "ablation charts and the top misclassifications.",
        )
    with col_hint:
        st.caption(
            "All charts are rendered as embedded PNGs (via kaleido). "
            "PDF generation needs the `report-export` extras "
            "(`pip install reportlab kaleido`)."
        )

    if not build_clicked:
        return

    # Build the cascade-settings snapshot from the *effective* config
    # at run time (mirrors _run()'s eval_cfg). If include_zeroshot was
    # off we set zeroshot.enabled=False here too so the appendix shows
    # the same effective config the cascade actually saw.
    include_zs = bool(meta.get("include_zeroshot", cfg.zeroshot.enabled))
    eff_cfg = cfg.model_copy(deep=True)
    if not include_zs:
        eff_cfg.zeroshot.enabled = False
    cascade_settings: dict[str, object] = {
        "vendor_exact_match": eff_cfg.vendor_exact_match.model_dump(),
        "knn": eff_cfg.knn.model_dump(),
        "classifier": eff_cfg.classifier.model_dump(),
        "category_similarity": eff_cfg.category_similarity.model_dump(),
        "zeroshot": eff_cfg.zeroshot.model_dump(),
    }
    ctx = ReportContext(
        account_name=account.name,
        embedding_model=cfg.embedding_model,
        n_folds=result.n_folds,
        seed=int(meta.get("seed", 0)),
        include_zeroshot=include_zs,
        category_id_to_name=id_to_name,
        cascade_settings=cascade_settings,
        zeroshot_model=cfg.zeroshot_model,
        device=cfg.device,
    )
    try:
        with st.spinner("Rendering report (charts → PNG → PDF)…"):
            pdf_bytes = build_pdf(result, abl, ctx)
    except RuntimeError as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF build failed: {e}")
        return

    st.download_button(
        "⬇ Download PDF",
        data=pdf_bytes,
        file_name=default_filename(account.name),
        mime="application/pdf",
        key="eval_download_pdf",
    )


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
