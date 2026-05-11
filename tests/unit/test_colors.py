"""Tests for the random_hex_color helper."""

from __future__ import annotations

import random

from expense_analyzer.utils.colors import random_hex_color


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
