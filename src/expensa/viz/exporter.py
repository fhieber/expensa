"""Save Plotly figures to disk in HTML or PNG format."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save_figure(fig: go.Figure, out: Path) -> Path:
    """Write `fig` to `out`. The file extension picks the format:

    * ``.html`` -> standalone interactive HTML
    * ``.png``  -> static raster (requires kaleido)
    * ``.svg`` / ``.pdf`` -> static vector (kaleido)
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".html" or suffix == "":
        if suffix == "":
            out = out.with_suffix(".html")
        fig.write_html(str(out), include_plotlyjs="cdn")
    elif suffix in {".png", ".svg", ".pdf", ".jpeg", ".jpg", ".webp"}:
        fig.write_image(str(out))
    else:
        raise ValueError(f"unsupported output format: {suffix}")
    return out
