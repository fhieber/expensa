"""Build a PDF report of a cross-validation + ablation run.

Wraps reportlab + plotly's kaleido-backed ``to_image`` so the heavy
imports (reportlab, kaleido) only fire when the user clicks "Export
PDF". Both are optional extras (``pip install
expensa[report-export]``); :func:`build_pdf` raises a
clear ``RuntimeError`` listing the missing packages if either is
absent so the UI can surface a usable hint.

Pure / Streamlit-free so it's straightforward to unit-test.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from expensa.ml.evaluation import AblationResult, EvalResult
from expensa.viz import (
    ablation_cumulative_curve,
    ablation_leave_one_out_bar,
    confusion_matrix_heatmap,
    stage_breakdown_bar,
)


@dataclass
class ReportContext:
    """Carry the bits the UI knows but the eval results don't.

    Keeps the report module loose-coupled from Streamlit + config.
    """

    account_name: str
    embedding_model: str
    n_folds: int
    seed: int
    include_zeroshot: bool
    category_id_to_name: dict[int, str]
    # Full cascade configuration as a plain dict (typically
    # ``Config.model_dump()`` filtered to the cascade-relevant keys).
    # Rendered as an appendix at the end of the report so the user can
    # reproduce exactly *which* settings produced the numbers in the
    # rest of the document. Default ``{}`` keeps backwards-compat with
    # any caller that hasn't been updated yet -- they'll just get a
    # PDF without the appendix.
    cascade_settings: dict[str, object] = field(default_factory=dict)
    # Zero-shot model id, mirrored into the appendix alongside the
    # embedding model so the "Models" section is self-contained.
    zeroshot_model: str = ""
    # Compute device the eval actually ran on. Same rationale.
    device: str = ""
    # Wall-clock seconds the cross-validation + ablation took. Surfaced
    # in the header line ("Generated: … · Runtime: 2m 22s") and in the
    # settings appendix so the PDF reader can compare runs on identical
    # config vs. different config. ``None`` skips the line.
    duration_seconds: float | None = None


def _require_deps() -> None:
    missing: list[str] = []
    try:
        import reportlab  # noqa: F401
    except ImportError:
        missing.append("reportlab")
    try:
        import kaleido  # noqa: F401
    except ImportError:
        missing.append("kaleido")
    if missing:
        raise RuntimeError(
            "PDF export needs extra packages: "
            + ", ".join(missing)
            + ". Install with `pip install "
            + " ".join(missing)
            + "` (or `pip install expensa[report-export]`)."
        )


def _fig_to_png(fig, width: int = 900, height: int = 480) -> bytes:
    # scale=2 makes the embedded PNG sharp on print without bloating
    # the PDF too much (plotly's default 1x looks fuzzy on paper).
    return fig.to_image(format="png", width=width, height=height, scale=2)


def format_duration(seconds: float | None) -> str:
    """Human-readable wall-clock duration.

    Picks the smallest unit that keeps the number short and scannable:
      * < 60s  -> "12.4s"
      * < 1h   -> "2m 22s"
      * else   -> "1h 05m"

    Lives here (not in the UI) so the PDF can render runtime without
    importing Streamlit. The eval tab re-exports / mirrors this for
    its caption.
    """
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    if s < 3600.0:
        m, sec = divmod(int(round(s)), 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(int(round(s)), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


def build_pdf(
    result: EvalResult,
    ablation: AblationResult | None,
    ctx: ReportContext,
) -> bytes:
    """Render the eval result + ablation into a single PDF; return bytes.

    Layout mirrors the on-screen eval tab so the user gets a faithful
    "this is what I saw" artifact rather than a stripped-down summary.
    """
    _require_deps()
    # Local imports (defer the heavy modules until the user actually clicks).
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title=f"Cascade quality — {ctx.account_name}",
        author="expensa",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    story: list = []

    # ─── Header ───────────────────────────────────────────────────────
    story.append(Paragraph("Cascade quality report", h1))
    header_bits = [
        f"Account: <b>{ctx.account_name}</b>",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if ctx.duration_seconds is not None:
        header_bits.append(f"Runtime: <b>{format_duration(ctx.duration_seconds)}</b>")
    story.append(Paragraph(" &nbsp;·&nbsp; ".join(header_bits), body))
    story.append(Paragraph(
        f"Embedding model: <font face='Courier'>{ctx.embedding_model}</font>",
        small,
    ))
    story.append(Paragraph(
        f"{result.n_folds}-fold stratified cross-validation · seed {ctx.seed} · "
        f"zero-shot stage {'INCLUDED' if ctx.include_zeroshot else 'excluded'}.",
        small,
    ))
    if result.dropped_singletons:
        story.append(Paragraph(
            f"{result.dropped_singletons} label(s) excluded "
            "(their category had fewer than two examples).",
            small,
        ))
    story.append(Spacer(1, 0.4 * cm))

    # ─── Headline metrics table ───────────────────────────────────────
    cov_str = "—" if result.accuracy_covered != result.accuracy_covered else f"{result.accuracy_covered:.1%}"
    metric_rows = [
        ["Accuracy", f"{result.accuracy:.1%}"],
        ["Accuracy (covered)", cov_str],
        ["Coverage", f"{result.coverage:.1%}"],
        ["Macro-F1", f"{result.macro_f1:.3f}"],
        ["Weighted-F1", f"{result.weighted_f1:.3f}"],
        ["Labels evaluated", str(result.n_labeled)],
    ]
    tbl = Table(metric_rows, colWidths=[6 * cm, 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    # ─── Per-stage contribution chart ─────────────────────────────────
    if result.stage_breakdown:
        story.append(Paragraph("Per-stage contribution", h2))
        png = _fig_to_png(stage_breakdown_bar(result.stage_breakdown))
        story.append(Image(io.BytesIO(png), width=16 * cm, height=8 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # ─── Per-category metrics table ───────────────────────────────────
    if result.per_category:
        story.append(PageBreak())
        story.append(Paragraph("Per-category metrics", h2))
        cat_rows = [["Category", "Precision", "Recall", "F1", "Support"]]
        # Sort by F1 desc to mirror the on-screen table.
        for pc in sorted(result.per_category, key=lambda p: -p.f1):
            cat_rows.append([
                ctx.category_id_to_name.get(pc.category_id, str(pc.category_id)),
                f"{pc.precision:.3f}",
                f"{pc.recall:.3f}",
                f"{pc.f1:.3f}",
                str(pc.support),
            ])
        tbl = Table(cat_rows, colWidths=[6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # ─── Confusion matrix ─────────────────────────────────────────────
    if result.confusion_labels:
        story.append(PageBreak())
        story.append(Paragraph("Confusion matrix", h2))
        story.append(Paragraph(
            "Rows = true category; columns = predicted category. "
            "Diagonal cells are correct predictions.", small,
        ))
        names = [ctx.category_id_to_name.get(cid, str(cid)) for cid in result.confusion_labels]
        png = _fig_to_png(
            confusion_matrix_heatmap(result.confusion, names),
            width=900, height=720,
        )
        story.append(Image(io.BytesIO(png), width=16 * cm, height=12.5 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # ─── Ablation charts ──────────────────────────────────────────────
    if ablation is not None and (ablation.cumulative or ablation.leave_one_out):
        story.append(PageBreak())
        story.append(Paragraph("Stage ablation", h2))
        story.append(Paragraph(
            f"Full-cascade baseline: accuracy {ablation.full_accuracy:.1%}, "
            f"macro-F1 {ablation.full_macro_f1:.3f}.", body,
        ))
        story.append(Spacer(1, 0.2 * cm))
        if ablation.cumulative:
            story.append(Paragraph("Cumulative — stages enabled in pipeline order:", body))
            png = _fig_to_png(ablation_cumulative_curve(ablation.cumulative))
            story.append(Image(io.BytesIO(png), width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.3 * cm))
        if ablation.leave_one_out:
            story.append(Paragraph("Leave-one-out — Δ vs. full cascade:", body))
            png = _fig_to_png(ablation_leave_one_out_bar(ablation.leave_one_out))
            story.append(Image(io.BytesIO(png), width=16 * cm, height=8 * cm))

    # ─── Misclassifications summary ───────────────────────────────────
    wrong = [r for r in result.records if not r[4]]
    if wrong:
        story.append(PageBreak())
        story.append(Paragraph(f"Misclassifications ({len(wrong)})", h2))
        story.append(Paragraph(
            "Up to 50 most-confused rows, grouped by (true → predicted). "
            "Stage column shows which cascade stage produced the wrong answer.",
            small,
        ))
        # Aggregate counts by (true, pred, stage) so the table stays compact.
        bucket: dict[tuple[int, int | None, str], int] = {}
        for _eid, true_cid, pred_cid, stage, _ok in wrong:
            bucket[(true_cid, pred_cid, stage)] = bucket.get((true_cid, pred_cid, stage), 0) + 1
        rows = [["True", "Predicted", "Stage", "Count"]]
        for (true_cid, pred_cid, stage), n in sorted(bucket.items(), key=lambda kv: -kv[1])[:50]:
            rows.append([
                ctx.category_id_to_name.get(true_cid, str(true_cid)),
                "—" if pred_cid is None else ctx.category_id_to_name.get(pred_cid, str(pred_cid)),
                stage,
                str(n),
            ])
        tbl = Table(rows, colWidths=[5 * cm, 5 * cm, 4 * cm, 1.8 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)

    # ─── Appendix: cascade settings used for this run ─────────────────
    if ctx.cascade_settings or ctx.zeroshot_model or ctx.device:
        _append_settings_appendix(
            story=story,
            ctx=ctx,
            styles=styles,
            colors=colors,
            cm=cm,
            include_zeroshot=ctx.include_zeroshot,
        )

    doc.build(story)
    return buf.getvalue()


# Pipeline-order list driving the appendix layout. Stages a user
# never sees in the cascade have no business appearing here, so we
# explicitly enumerate them rather than dumping ``cfg.model_dump()``
# wholesale. The order matches ``evaluation.STAGE_ORDER``.
_APPENDIX_STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("vendor_exact_match", "1. Vendor exact match"),
    ("knn",                "2. k-NN (embedding neighbours)"),
    ("classifier",         "3. Classifier"),
    ("category_similarity","4. Category similarity"),
    ("zeroshot",           "5. Zero-shot NLI (fallback)"),
)


def _format_setting_value(v: object) -> str:
    """Render a config value as it should appear in the appendix.

    Hides the booleans/numbers/strings-vs-templates distinction
    behind one helper so every stage table looks the same.
    """
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        # Three significant digits is enough for thresholds; trailing
        # zeros stripped so 0.700 -> 0.7.
        return f"{v:.3f}".rstrip("0").rstrip(".") or "0"
    if isinstance(v, str):
        return v if v else "(empty)"
    return str(v)


def _append_settings_appendix(
    story: list,
    ctx: ReportContext,
    styles,
    colors,
    cm: float,
    include_zeroshot: bool,
) -> None:
    """Render the per-stage settings appendix.

    Layout: one Models table at the top (embedding + zero-shot + device),
    then one mini-table per cascade stage in pipeline order. Stages
    that weren't included in this run (zeroshot when the user
    unchecked it) are still shown with their on-disk values but
    annotated as "(disabled for this run)" so the reader can tell
    documented config from effective config.
    """
    from reportlab.platypus import PageBreak, Paragraph, Spacer

    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    story.append(PageBreak())
    story.append(Paragraph("Appendix — cascade settings used for this run", h2))
    story.append(Paragraph(
        "These are the resolved configuration values that produced "
        "the metrics above. Tweak any of them under "
        "<b>Settings → Categorization Cascade</b> in the Streamlit UI "
        "(or in <font face='Courier'>~/.expensa/config.yaml</font>) "
        "and re-run the Quality tab to A/B the impact.",
        body,
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Models / device block --------------------------------------
    if ctx.embedding_model or ctx.zeroshot_model or ctx.device:
        story.append(Paragraph("Models &amp; device", h3))
        rows: list[list[str]] = [["Setting", "Value"]]
        if ctx.embedding_model:
            rows.append(["embedding_model", ctx.embedding_model])
        if ctx.zeroshot_model:
            rows.append(["zeroshot_model", ctx.zeroshot_model])
        if ctx.device:
            rows.append(["device", ctx.device])
        rows.append(["seed", str(ctx.seed)])
        rows.append(["n_folds", str(ctx.n_folds)])
        rows.append(["zeroshot stage included", "yes" if include_zeroshot else "no"])
        if ctx.duration_seconds is not None:
            rows.append(["runtime", format_duration(ctx.duration_seconds)])
        _append_kv_table(story, rows, colors, cm)
        story.append(Spacer(1, 0.4 * cm))

    # --- Per-stage tables in pipeline order -------------------------
    cs = ctx.cascade_settings or {}
    for stage_key, stage_title in _APPENDIX_STAGE_ORDER:
        stage_cfg = cs.get(stage_key)
        if not isinstance(stage_cfg, dict):
            continue  # caller didn't populate this stage; skip rather than blank-row
        # Annotate if the stage was disabled for the run we just reported
        # on. The classifier doesn't have an `enabled` field (always on);
        # for zeroshot we additionally honour the include_zeroshot UI flag.
        was_active = not (
            (stage_key == "zeroshot" and not include_zeroshot)
            or ("enabled" in stage_cfg and not stage_cfg["enabled"])
        )
        suffix = "" if was_active else "  <i>(disabled for this run)</i>"
        story.append(Paragraph(stage_title + suffix, h3))
        rows = [["Setting", "Value"]]
        for k, v in stage_cfg.items():
            rows.append([k, _format_setting_value(v)])
        _append_kv_table(story, rows, colors, cm)
        story.append(Spacer(1, 0.3 * cm))


def _append_kv_table(story: list, rows: list[list[str]], colors, cm: float) -> None:
    """Render a two-column "Setting / Value" table with consistent styling."""
    from reportlab.platypus import Table, TableStyle

    tbl = Table(rows, colWidths=[6 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("FONTNAME", (1, 1), (1, -1), "Courier"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(tbl)


def default_filename(account_name: str) -> str:
    """Produce a filesystem-safe filename for the PDF download_button."""
    return _default_filename(account_name, "pdf")


def default_html_filename(account_name: str) -> str:
    """Produce a filesystem-safe filename for the HTML download_button."""
    return _default_filename(account_name, "html")


def _default_filename(account_name: str, ext: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in account_name).strip("_")
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return f"cascade-quality_{safe or 'account'}_{ts}.{ext}"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
#
# Unlike the PDF path, this needs no optional extras: plotly is a core
# dependency and charts render to inline <div>s (interactive, not flat
# PNGs). plotly.js is embedded once so the file is fully self-contained
# and works offline / when emailed.


def _esc(v: object) -> str:
    """HTML-escape any value for safe interpolation into the template."""
    import html as _html

    return _html.escape(str(v), quote=True)


def _html_kv_table(rows: list[tuple[str, str]], *, mono: bool = False) -> str:
    """Two-column Setting/Value table. ``mono`` renders cells in a
    monospace font (used for the config appendix)."""
    cls = " class='mono'" if mono else ""
    body = "".join(
        f"<tr><th scope='row'>{_esc(k)}</th><td{cls}>{_esc(val)}</td></tr>"
        for k, val in rows
    )
    return f"<table class='kv'><tbody>{body}</tbody></table>"


def build_html(
    result: EvalResult,
    ablation: AblationResult | None,
    ctx: ReportContext,
) -> str:
    """Render the eval result + ablation into a single self-contained
    HTML document and return it as a string.

    Mirrors :func:`build_pdf` section-for-section so the two exports
    are interchangeable, but keeps the charts interactive and needs no
    reportlab / kaleido. plotly.js is inlined once at the top.
    """
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    def fig_div(fig) -> str:
        # plotly.js is embedded once in <head>; per-figure divs reference it.
        return pio.to_html(
            fig, include_plotlyjs=False, full_html=False,
            default_width="100%", default_height="460px",
            config={"displaylogo": False},
        )

    parts: list[str] = []

    # ─── Header ───────────────────────────────────────────────────────
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_bits = [f"Account: <b>{_esc(ctx.account_name)}</b>", f"Generated: {gen}"]
    if ctx.duration_seconds is not None:
        header_bits.append(f"Runtime: <b>{_esc(format_duration(ctx.duration_seconds))}</b>")
    parts.append("<header>")
    parts.append("<h1>Cascade quality report</h1>")
    parts.append(f"<p class='subtle'>{' &middot; '.join(header_bits)}</p>")
    parts.append(
        f"<p class='small'>Embedding model: <code>{_esc(ctx.embedding_model)}</code></p>"
    )
    parts.append(
        f"<p class='small'>{result.n_folds}-fold stratified cross-validation "
        f"&middot; seed {ctx.seed} &middot; zero-shot stage "
        f"{'INCLUDED' if ctx.include_zeroshot else 'excluded'}.</p>"
    )
    if result.dropped_singletons:
        parts.append(
            f"<p class='small'>{result.dropped_singletons} label(s) excluded "
            "(their category had fewer than two examples).</p>"
        )
    parts.append("</header>")

    # ─── Headline metrics ─────────────────────────────────────────────
    cov = "—" if result.accuracy_covered != result.accuracy_covered else f"{result.accuracy_covered:.1%}"
    metrics = [
        ("Accuracy", f"{result.accuracy:.1%}"),
        ("Accuracy (covered)", cov),
        ("Coverage", f"{result.coverage:.1%}"),
        ("Macro-F1", f"{result.macro_f1:.3f}"),
        ("Weighted-F1", f"{result.weighted_f1:.3f}"),
        ("Labels evaluated", str(result.n_labeled)),
    ]
    cards = "".join(
        f"<div class='card'><div class='card-val'>{_esc(v)}</div>"
        f"<div class='card-lbl'>{_esc(k)}</div></div>"
        for k, v in metrics
    )
    parts.append(f"<section class='metrics'>{cards}</section>")

    # ─── Per-stage contribution ───────────────────────────────────────
    if result.stage_breakdown:
        parts.append("<section><h2>Per-stage contribution</h2>")
        parts.append(fig_div(stage_breakdown_bar(result.stage_breakdown)))
        parts.append("</section>")

    # ─── Per-category metrics ─────────────────────────────────────────
    if result.per_category:
        rows = "".join(
            "<tr>"
            f"<td>{_esc(ctx.category_id_to_name.get(pc.category_id, pc.category_id))}</td>"
            f"<td class='num'>{pc.precision:.3f}</td>"
            f"<td class='num'>{pc.recall:.3f}</td>"
            f"<td class='num'>{pc.f1:.3f}</td>"
            f"<td class='num'>{pc.support}</td>"
            "</tr>"
            for pc in sorted(result.per_category, key=lambda p: -p.f1)
        )
        parts.append(
            "<section><h2>Per-category metrics</h2>"
            "<table class='grid'><thead><tr>"
            "<th>Category</th><th>Precision</th><th>Recall</th>"
            "<th>F1</th><th>Support</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>"
        )

    # ─── Confusion matrix ─────────────────────────────────────────────
    if result.confusion_labels:
        names = [ctx.category_id_to_name.get(cid, str(cid)) for cid in result.confusion_labels]
        parts.append("<section><h2>Confusion matrix</h2>")
        parts.append(
            "<p class='small'>Rows = true category; columns = predicted "
            "category. Diagonal cells are correct predictions.</p>"
        )
        parts.append(fig_div(confusion_matrix_heatmap(result.confusion, names)))
        parts.append("</section>")

    # ─── Ablation ─────────────────────────────────────────────────────
    if ablation is not None and (ablation.cumulative or ablation.leave_one_out):
        parts.append("<section><h2>Stage ablation</h2>")
        parts.append(
            f"<p class='small'>Full-cascade baseline: accuracy "
            f"{ablation.full_accuracy:.1%}, macro-F1 {ablation.full_macro_f1:.3f}.</p>"
        )
        if ablation.cumulative:
            parts.append("<h3>Cumulative — stages enabled in pipeline order</h3>")
            parts.append(fig_div(ablation_cumulative_curve(ablation.cumulative)))
        if ablation.leave_one_out:
            parts.append("<h3>Leave-one-out — Δ vs. full cascade</h3>")
            parts.append(fig_div(ablation_leave_one_out_bar(ablation.leave_one_out)))
        parts.append("</section>")

    # ─── Misclassifications ───────────────────────────────────────────
    wrong = [r for r in result.records if not r[4]]
    if wrong:
        bucket: dict[tuple[int, int | None, str], int] = {}
        for _eid, true_cid, pred_cid, stage, _ok in wrong:
            bucket[(true_cid, pred_cid, stage)] = bucket.get((true_cid, pred_cid, stage), 0) + 1
        rows = "".join(
            "<tr>"
            f"<td>{_esc(ctx.category_id_to_name.get(true_cid, true_cid))}</td>"
            f"<td>{'—' if pred_cid is None else _esc(ctx.category_id_to_name.get(pred_cid, pred_cid))}</td>"
            f"<td>{_esc(stage)}</td>"
            f"<td class='num'>{n}</td>"
            "</tr>"
            for (true_cid, pred_cid, stage), n in sorted(
                bucket.items(), key=lambda kv: -kv[1]
            )[:50]
        )
        parts.append(
            f"<section><h2>Misclassifications ({len(wrong)})</h2>"
            "<p class='small'>Up to 50 most-confused rows, grouped by "
            "(true → predicted). Stage shows which cascade stage produced "
            "the wrong answer.</p>"
            "<table class='grid'><thead><tr><th>True</th><th>Predicted</th>"
            "<th>Stage</th><th>Count</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>"
        )

    # ─── Appendix: cascade settings ───────────────────────────────────
    if ctx.cascade_settings or ctx.zeroshot_model or ctx.device:
        parts.append(_html_settings_appendix(ctx, ctx.include_zeroshot))

    body = "\n".join(parts)
    title = f"Cascade quality — {_esc(ctx.account_name)}"
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        f"<style>{_HTML_CSS}</style>"
        f"<script>{get_plotlyjs()}</script>"
        f"</head><body><main>{body}</main></body></html>"
    )


def _html_settings_appendix(ctx: ReportContext, include_zeroshot: bool) -> str:
    """Mirror the PDF cascade-settings appendix as HTML tables."""
    parts: list[str] = ["<section class='appendix'>"]
    parts.append("<h2>Appendix — cascade settings used for this run</h2>")
    parts.append(
        "<p class='small'>The resolved configuration values that produced "
        "the metrics above. Tweak them under <b>Settings → Categorization "
        "Cascade</b> (or <code>~/.expensa/config.yaml</code>) and re-run "
        "the Quality tab to A/B the impact.</p>"
    )

    # Models / device block.
    if ctx.embedding_model or ctx.zeroshot_model or ctx.device:
        mrows: list[tuple[str, str]] = []
        if ctx.embedding_model:
            mrows.append(("embedding_model", ctx.embedding_model))
        if ctx.zeroshot_model:
            mrows.append(("zeroshot_model", ctx.zeroshot_model))
        if ctx.device:
            mrows.append(("device", ctx.device))
        mrows.append(("seed", str(ctx.seed)))
        mrows.append(("n_folds", str(ctx.n_folds)))
        mrows.append(("zeroshot stage included", "yes" if include_zeroshot else "no"))
        if ctx.duration_seconds is not None:
            mrows.append(("runtime", format_duration(ctx.duration_seconds)))
        parts.append("<h3>Models &amp; device</h3>")
        parts.append(_html_kv_table(mrows, mono=True))

    # Per-stage tables in pipeline order.
    cs = ctx.cascade_settings or {}
    for stage_key, stage_title in _APPENDIX_STAGE_ORDER:
        stage_cfg = cs.get(stage_key)
        if not isinstance(stage_cfg, dict):
            continue
        was_active = not (
            (stage_key == "zeroshot" and not include_zeroshot)
            or ("enabled" in stage_cfg and not stage_cfg["enabled"])
        )
        suffix = "" if was_active else " <span class='muted'>(disabled for this run)</span>"
        parts.append(f"<h3>{_esc(stage_title)}{suffix}</h3>")
        parts.append(
            _html_kv_table(
                [(k, _format_setting_value(v)) for k, v in stage_cfg.items()],
                mono=True,
            )
        )

    parts.append("</section>")
    return "".join(parts)


_HTML_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
  Roboto, Helvetica, Arial, sans-serif; line-height: 1.45; color: #1a1a1a;
  background: #fff; }
main { max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.7rem; margin: 0 0 .5rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 .75rem; border-bottom: 2px solid #eee;
  padding-bottom: .3rem; }
h3 { font-size: 1.02rem; margin: 1.2rem 0 .4rem; }
p.subtle { color: #444; margin: .2rem 0; }
p.small { color: #666; font-size: .85rem; margin: .2rem 0 .6rem; }
code { background: #f3f3f3; padding: .05rem .3rem; border-radius: 3px;
  font-size: .85em; }
section.metrics { display: grid; grid-template-columns: repeat(auto-fit,
  minmax(150px, 1fr)); gap: .6rem; margin: 1rem 0 1.5rem; }
.card { border: 1px solid #e4e4e4; border-radius: 8px; padding: .8rem 1rem;
  background: #fafafa; }
.card-val { font-size: 1.5rem; font-weight: 650; }
.card-lbl { color: #666; font-size: .8rem; margin-top: .15rem; }
table { border-collapse: collapse; width: 100%; margin: .4rem 0 1rem;
  font-size: .88rem; }
table.grid th, table.grid td { border: 1px solid #e0e0e0; padding: .35rem .6rem;
  text-align: left; }
table.grid thead th { background: #f3f3f3; }
table.grid tbody tr:nth-child(even) { background: #fafafa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.kv { max-width: 640px; }
table.kv th { text-align: left; font-weight: 600; padding: .25rem .6rem;
  width: 14rem; vertical-align: top; }
table.kv td { padding: .25rem .6rem; }
table.kv td.mono, table.kv th { font-family: ui-monospace, SFMono-Regular,
  Menlo, Consolas, monospace; }
.muted { color: #999; font-weight: 400; font-size: .85em; }
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e6e6e6; }
  h2 { border-bottom-color: #2a2d33; }
  .card { background: #1e2127; border-color: #2a2d33; }
  .card-lbl, p.small, p.subtle { color: #a0a4ad; }
  code { background: #23262d; }
  table.grid th, table.grid td { border-color: #2a2d33; }
  table.grid thead th { background: #23262d; }
  table.grid tbody tr:nth-child(even) { background: #1b1e24; }
}
"""


__all__: Iterable[str] = (
    "ReportContext",
    "build_pdf",
    "build_html",
    "default_filename",
    "default_html_filename",
    "format_duration",
)
