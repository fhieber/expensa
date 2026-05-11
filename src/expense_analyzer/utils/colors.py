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
