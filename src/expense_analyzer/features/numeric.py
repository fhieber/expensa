"""Numeric features derived from `Betrag (€)`."""

from __future__ import annotations

import math
from decimal import Decimal


def is_income(amount: Decimal) -> bool:
    return amount > 0


def is_round_amount(amount: Decimal, divisors: tuple[int, ...] = (5, 10, 50, 100)) -> bool:
    """True iff |amount| is divisible by any of the given divisors."""
    cents = abs(int(amount * 100))
    return any(cents % (d * 100) == 0 and cents != 0 for d in divisors)


def amount_bucket(amount: Decimal) -> str:
    """Coarse bucket on absolute amount, useful as a categorical feature."""
    a = abs(float(amount))
    if a < 10:
        return "<10"
    if a < 50:
        return "10-50"
    if a < 200:
        return "50-200"
    if a < 1000:
        return "200-1000"
    return ">1000"


def log_abs_amount(amount: Decimal) -> float:
    return math.log1p(abs(float(amount)))
