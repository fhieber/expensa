"""Tests for the colour helpers."""

from __future__ import annotations

import random

from expensa.utils.colors import random_hex_color, readable_text_color


def test_random_hex_color_format() -> None:
    for _ in range(20):
        c = random_hex_color()
        assert c.startswith("#")
        assert len(c) == 7
        int(c[1:], 16)  # raises if not hex


def test_random_hex_color_seeded_is_deterministic() -> None:
    a = random_hex_color(random.Random(42))
    b = random_hex_color(random.Random(42))
    assert a == b


def test_random_hex_colors_are_varied() -> None:
    samples = {random_hex_color() for _ in range(50)}
    assert len(samples) > 10


def test_readable_text_color_picks_white_on_dark_bg() -> None:
    # Pure black, very dark blue, deep purple -> white text.
    assert readable_text_color("#000000") == "#ffffff"
    assert readable_text_color("#001f3f") == "#ffffff"
    assert readable_text_color("#4b0082") == "#ffffff"


def test_readable_text_color_picks_black_on_light_bg() -> None:
    # Pure white, pale yellow, light cyan -> black text.
    assert readable_text_color("#ffffff") == "#000000"
    assert readable_text_color("#ffffe0") == "#000000"
    assert readable_text_color("#e0ffff") == "#000000"


def test_readable_text_color_accepts_3_digit_shorthand() -> None:
    assert readable_text_color("#fff") == "#000000"
    assert readable_text_color("#000") == "#ffffff"


def test_readable_text_color_handles_malformed_input() -> None:
    # Bogus values should fall back to black (caller still gets *some*
    # valid CSS).
    assert readable_text_color("") == "#000000"
    assert readable_text_color("not-a-color") == "#000000"
    assert readable_text_color("#zzzzzz") == "#000000"
