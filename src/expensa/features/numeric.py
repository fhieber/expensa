"""Numeric features derived from `Betrag (€)` plus a few cheap shape /
categorical helpers (amount patterns, umsatztyp buckets, text shape,
cyclical calendar encodings)."""

from __future__ import annotations

import math
import re
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


# ---------------------------------------------------------------------------
# Amount-pattern flags. Cheap binary signals with real categorical power.
# All operate on integer cents so there's no float-rounding ambiguity.
# ---------------------------------------------------------------------------


def has_cents(betrag_cents: int) -> int:
    """1 iff the amount is NOT a whole euro. Whole-euro amounts skew toward
    transfers / rent / standing orders; retail tends to have odd cents."""
    return int(betrag_cents % 100 != 0)


def is_small_verification(betrag_cents: int) -> int:
    """1 iff 0 < |amount| <= 1.00 €. The classic card-verification / micro-
    charge pattern (also a fraud-probe signature)."""
    a = abs(int(betrag_cents))
    return int(0 < a <= 100)


def amount_ends_99(betrag_cents: int) -> int:
    """1 iff the amount ends in .99 (psychological retail pricing)."""
    return int(abs(int(betrag_cents)) % 100 == 99)


# ---------------------------------------------------------------------------
# Umsatztyp (German transaction-type) bucketing. The raw bank value is very
# predictive (Dauerauftrag -> rent/subs, Gehalt -> income, Bargeld -> cash)
# but free-text, so we fold it to a small fixed vocabulary that can be
# one-hot encoded into a stable set of columns.
# ---------------------------------------------------------------------------

# Canonical buckets, in a fixed order so the one-hot columns are stable
# across runs. ``other`` catches anything unmatched (incl. empty).
UMSATZTYP_BUCKETS: tuple[str, ...] = (
    "lastschrift",   # direct debit
    "dauerauftrag",  # standing order
    "ueberweisung",  # credit transfer
    "gehalt",        # salary / pension / wages
    "karte",         # card payment (POS / giro / debit)
    "bargeld",       # cash withdrawal
    "gutschrift",    # credit / incoming
    "entgelt",       # bank fee / charge
    "other",
)

# Substring -> bucket. First match in this order wins; longer / more
# specific patterns come first where they'd otherwise collide.
_UMSATZTYP_PATTERNS: tuple[tuple[str, str], ...] = (
    ("dauerauftrag", "dauerauftrag"),
    ("lastschrift", "lastschrift"),
    ("gehalt", "gehalt"),
    ("rente", "gehalt"),
    ("lohn", "gehalt"),
    ("bezuege", "gehalt"),
    ("bezüge", "gehalt"),
    ("bargeld", "bargeld"),
    ("geldautomat", "bargeld"),
    ("auszahlung", "bargeld"),
    ("abhebung", "bargeld"),
    ("kartenzahlung", "karte"),
    ("karte", "karte"),
    ("girocard", "karte"),
    ("giro", "karte"),
    ("debitk", "karte"),
    ("pos", "karte"),
    ("visa", "karte"),
    ("master", "karte"),
    ("gutschrift", "gutschrift"),
    ("entgelt", "entgelt"),
    ("gebuehr", "entgelt"),
    ("gebühr", "entgelt"),
    # Plain transfer last so the more specific salary / fee credit-transfers
    # above win first.
    ("ueberweisung", "ueberweisung"),
    ("überweisung", "ueberweisung"),
    ("uberweisung", "ueberweisung"),
    ("echtzeit", "ueberweisung"),
)


def umsatztyp_bucket(raw: str | None) -> str:
    """Fold a raw German umsatztyp string to one of :data:`UMSATZTYP_BUCKETS`.

    Case-insensitive substring match against :data:`_UMSATZTYP_PATTERNS`;
    unmatched / empty values fall to ``"other"``. Pure + deterministic so
    it can be unit-pinned."""
    if not raw:
        return "other"
    s = raw.strip().lower()
    for needle, bucket in _UMSATZTYP_PATTERNS:
        if needle in s:
            return bucket
    return "other"


# ---------------------------------------------------------------------------
# Text-shape features. Cheap structural signals over the normalised text
# that the embedding can smooth over (a terse "rewe" vs a long structured
# SEPA memo correlate with very different transaction types).
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")
_DIGIT_RE = re.compile(r"\d")


def text_length(s: str | None) -> int:
    return len(s) if s else 0


def token_count(s: str | None) -> int:
    return len(_TOKEN_RE.findall(s)) if s else 0


def digit_ratio(s: str | None) -> float:
    """Fraction of characters that are digits (0..1). Long-digit-heavy
    memos are typically machine references; word-heavy ones are human."""
    if not s:
        return 0.0
    n = len(s)
    if n == 0:
        return 0.0
    return len(_DIGIT_RE.findall(s)) / n


# ---------------------------------------------------------------------------
# Cyclical calendar encoding. Raw month/day-of-week/day-of-month integers
# tell a linear model that Dec(12) is far from Jan(1) and Mon(0) far from
# Sun(6) -- both false. (sin, cos) on the period makes the wrap-around
# distance correct. Trees keep using the raw integers; linear models lean
# on these.
# ---------------------------------------------------------------------------


def cyclical(value: float, period: float) -> tuple[float, float]:
    """Return ``(sin, cos)`` of ``value`` mapped onto a circle of the given
    ``period``. ``cyclical(0, 12) == cyclical(12, 12)`` (same point)."""
    angle = 2.0 * math.pi * (float(value) / period)
    return math.sin(angle), math.cos(angle)
