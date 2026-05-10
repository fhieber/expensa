"""Numeric-feature tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from expense_analyzer.features.numeric import (
    amount_bucket,
    is_income,
    is_round_amount,
    log_abs_amount,
)


def test_is_income() -> None:
    assert is_income(Decimal("100.00"))
    assert not is_income(Decimal("-50.00"))
    assert not is_income(Decimal("0.00"))


@pytest.mark.parametrize(
    "amount,expected",
    [
        (Decimal("100.00"), True),
        (Decimal("-50.00"), True),
        (Decimal("25.00"), True),
        (Decimal("12.34"), False),
        (Decimal("0.00"), False),  # zero should not count as round
        (Decimal("1000.00"), True),
    ],
)
def test_is_round_amount(amount: Decimal, expected: bool) -> None:
    assert is_round_amount(amount) is expected


@pytest.mark.parametrize(
    "amount,bucket",
    [
        (Decimal("3.50"), "<10"),
        (Decimal("-12.34"), "10-50"),
        (Decimal("99.99"), "50-200"),
        (Decimal("450.00"), "200-1000"),
        (Decimal("-2500.00"), ">1000"),
    ],
)
def test_amount_bucket(amount: Decimal, bucket: str) -> None:
    assert amount_bucket(amount) == bucket


def test_log_abs_amount_sign_independent() -> None:
    assert log_abs_amount(Decimal("100")) == log_abs_amount(Decimal("-100"))
    assert log_abs_amount(Decimal("0")) == 0
