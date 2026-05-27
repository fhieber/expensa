"""Tiny color utilities. Kept separate so we don't import streamlit/pandas
just to suggest a hex string."""

from __future__ import annotations

import colorsys
import random


def random_hex_color(rng: random.Random | None = None) -> str:
    """Suggest a pleasant random hex color for a new category.

    Samples from HSL with a moderate-to-vivid saturation and mid lightness,
    so the result is readable on both light and dark Streamlit themes.
    """
    r = rng or random
    h = r.random()
    s = 0.55 + r.random() * 0.25  # 0.55–0.80
    lightness = 0.50 + r.random() * 0.10  # 0.50–0.60
    r_f, g_f, b_f = colorsys.hls_to_rgb(h, lightness, s)
    return f"#{int(r_f * 255):02x}{int(g_f * 255):02x}{int(b_f * 255):02x}"


def readable_text_color(bg_hex: str) -> str:
    """Return ``'#000000'`` or ``'#ffffff'`` — whichever is more readable on
    the given background hex (``#rrggbb`` or ``#rgb``). Uses the WCAG
    relative-luminance formula and the standard 0.179 threshold.

    Defensive: malformed input falls back to black, since most category
    colours sampled by ``random_hex_color`` sit in the mid-luminance
    band where black text is legible."""
    s = (bg_hex or "").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return "#000000"
    try:
        r, g, b = (int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return "#000000"

    def _channel(c: float) -> float:
        # sRGB → linear (WCAG 2.x)
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)
    return "#000000" if lum > 0.179 else "#ffffff"
