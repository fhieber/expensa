"""Tests for the feature-engineering batch (items 1-9):

  * amount-pattern flags (has_cents, is_small_verification, amount_ends_99)
  * umsatztyp bucketing + one-hot
  * text-shape features
  * cyclical calendar encodings + restored `week`
  * global amount z-score fallback
  * Gläubiger-ID / IBAN identity-keyed recurrence + counts
  * Gläubiger-ID label-distribution stage-1 fallback
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date

from expensa.features.numeric import (
    UMSATZTYP_BUCKETS,
    amount_ends_99,
    cyclical,
    digit_ratio,
    has_cents,
    is_small_verification,
    text_length,
    token_count,
    umsatztyp_bucket,
)
from expensa.features.temporal import compute_temporal_features_bulk
from expensa.ml.classifier import _NUMERIC_COLS, _vendor_exact_match
from expensa.storage.categories import (
    add_label,
    glaeubiger_label_distribution,
    upsert_category,
)


# ── Amount-pattern flags ──────────────────────────────────────────────


def test_has_cents() -> None:
    assert has_cents(1234) == 1      # 12.34 €
    assert has_cents(1200) == 0      # 12.00 €
    assert has_cents(-9999) == 1
    assert has_cents(0) == 0


def test_is_small_verification() -> None:
    assert is_small_verification(1) == 1      # 0.01 €
    assert is_small_verification(100) == 1    # 1.00 €
    assert is_small_verification(101) == 0    # 1.01 €
    assert is_small_verification(0) == 0
    assert is_small_verification(-50) == 1    # sign-agnostic


def test_amount_ends_99() -> None:
    assert amount_ends_99(1299) == 1    # 12.99
    assert amount_ends_99(-499) == 1
    assert amount_ends_99(1300) == 0


# ── Umsatztyp bucketing ───────────────────────────────────────────────


def test_umsatztyp_bucket_maps_german_types() -> None:
    cases = {
        "Lastschrift": "lastschrift",
        "SEPA-Lastschrift": "lastschrift",
        "Dauerauftrag": "dauerauftrag",
        "Überweisung": "ueberweisung",
        "Echtzeitüberweisung": "ueberweisung",
        "Gehalt/Rente": "gehalt",
        "Lohn/Gehalt": "gehalt",
        "Kartenzahlung": "karte",
        "Bargeldauszahlung": "bargeld",
        "Geldautomat": "bargeld",
        "Gutschrift": "gutschrift",
        "Entgelt": "entgelt",
        "": "other",
        None: "other",
        "Something weird": "other",
    }
    for raw, expected in cases.items():
        assert umsatztyp_bucket(raw) == expected, raw


def test_umsatztyp_buckets_are_exhaustive_and_unique() -> None:
    # Every pattern target must be a declared bucket.
    from expensa.features.numeric import _UMSATZTYP_PATTERNS

    for _needle, bucket in _UMSATZTYP_PATTERNS:
        assert bucket in UMSATZTYP_BUCKETS
    assert len(UMSATZTYP_BUCKETS) == len(set(UMSATZTYP_BUCKETS))


# ── Text-shape ────────────────────────────────────────────────────────


def test_text_shape_features() -> None:
    assert text_length("rewe") == 4
    assert text_length("") == 0
    assert text_length(None) == 0
    assert token_count("rewe markt berlin") == 3
    assert token_count("") == 0
    # 2 digits out of 4 chars.
    assert digit_ratio("ab12") == 0.5
    assert digit_ratio("abcd") == 0.0
    assert digit_ratio(None) == 0.0


# ── Cyclical encoding ─────────────────────────────────────────────────


def test_cyclical_wraps_around() -> None:
    # Month 12 and month 0 land on the same circle point.
    s0, c0 = cyclical(0, 12)
    s12, c12 = cyclical(12, 12)
    assert math.isclose(s0, s12, abs_tol=1e-9)
    assert math.isclose(c0, c12, abs_tol=1e-9)
    # Quarter-turn: value = period/4 -> sin=1, cos=0.
    s, c = cyclical(3, 12)
    assert math.isclose(s, 1.0, abs_tol=1e-9)
    assert math.isclose(c, 0.0, abs_tol=1e-9)


# ── _NUMERIC_COLS wiring (items 1, 2, 4, 7, 8) ────────────────────────


def test_week_is_restored_to_numeric_cols() -> None:
    # `week` was computed by basic_calendar_features but dropped before the
    # model -- same bug class as the earlier day_of_month fix.
    assert "week" in _NUMERIC_COLS


def test_new_features_in_numeric_cols() -> None:
    expected = {
        "has_cents", "is_small_verification", "amount_ends_99",
        "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos",
        "amount_zscore_global", "glaeubiger_count_before",
        "is_recurring_stable_key",
        "vz_length", "vz_token_count", "vz_digit_ratio",
        "umsatztyp_lastschrift", "umsatztyp_dauerauftrag", "umsatztyp_gehalt",
        "umsatztyp_karte", "umsatztyp_bargeld", "umsatztyp_other",
    }
    assert expected.issubset(set(_NUMERIC_COLS))


# ── Temporal: global z-score + identity-keyed recurrence/counts ───────


def _ins(conn, *, eid, cents, cpn, iban="", gid="", d):
    conn.execute(
        """
        INSERT INTO expenses(
            id, buchungsdatum, betrag_cents, iban, glaeubiger_id,
            counterparty, counterparty_normalized,
            verwendungszweck_normalized, combined_text, dedup_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (eid, d.isoformat(), cents, iban, gid, cpn, cpn, cpn, cpn, f"h{eid}"),
    )


def test_global_amount_zscore_exists_without_vendor_history(
    tmp_db: sqlite3.Connection,
) -> None:
    # Three DIFFERENT vendors -> amount_zscore_within_cp is None for all
    # (no per-vendor history), but the global z-score kicks in by the 3rd.
    base = date(2026, 1, 1)
    from datetime import timedelta

    # Priors must have spread (else variance 0 -> z 0); 1000 and 2000.
    _ins(tmp_db, eid=1, cents=-1000, cpn="a", d=base)
    _ins(tmp_db, eid=2, cents=-2000, cpn="b", d=base + timedelta(days=1))
    _ins(tmp_db, eid=3, cents=-9000, cpn="c", d=base + timedelta(days=2))
    feats = compute_temporal_features_bulk(tmp_db)
    assert feats[3]["amount_zscore_within_cp"] is None
    # Row 3 has 2 prior global rows -> global z is populated and positive
    # (9000 is well above the 1000/2000 prior mean).
    assert feats[3]["amount_zscore_global"] is not None
    assert feats[3]["amount_zscore_global"] > 0


def test_glaeubiger_and_iban_counts_before(tmp_db: sqlite3.Connection) -> None:
    from datetime import timedelta

    base = date(2026, 1, 1)
    # Same creditor id, but the display name drifts each month.
    for i in range(4):
        _ins(
            tmp_db, eid=i + 1, cents=-1999, cpn=f"netflix var {i}",
            iban="DE_NFLX", gid="DE98ZZZ0001", d=base + timedelta(days=30 * i),
        )
    feats = compute_temporal_features_bulk(tmp_db)
    # 4th charge has 3 priors sharing the creditor id AND the iban.
    assert feats[4]["glaeubiger_count_before"] == 3
    assert feats[4]["iban_count_before"] == 3


def test_recurring_stable_key_catches_name_drift(tmp_db: sqlite3.Connection) -> None:
    # Identical charge on the 1st of four consecutive months (so the priors
    # span >=3 DISTINCT calendar months), same creditor id, but a DIFFERENT
    # name each time -> name-keyed is_likely_recurring misses it; the
    # stable-key one catches it.
    for i, d in enumerate(
        [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)]
    ):
        _ins(
            tmp_db, eid=i + 1, cents=-1299, cpn=f"spotify ref {i}",
            gid="DE11ZZZ9999", d=d,
        )
    feats = compute_temporal_features_bulk(tmp_db)
    last = feats[4]
    assert last["is_likely_recurring"] == 0          # name drifted
    assert last["is_recurring_stable_key"] == 1       # creditor-id stable


# ── Stage-1 Gläubiger-ID fallback ─────────────────────────────────────


def test_glaeubiger_label_distribution(tmp_db: sqlite3.Connection) -> None:
    cat = upsert_category(tmp_db, "Abos")
    _ins(tmp_db, eid=1, cents=-1299, cpn="spotify a", gid="GID1", d=date(2026, 1, 1))
    _ins(tmp_db, eid=2, cents=-1299, cpn="spotify b", gid="GID1", d=date(2026, 2, 1))
    add_label(tmp_db, 1, cat, "user")
    add_label(tmp_db, 2, cat, "user")
    assert glaeubiger_label_distribution(tmp_db, "GID1") == {cat: 2}
    assert glaeubiger_label_distribution(tmp_db, "") == {}


def test_vendor_exact_match_falls_back_to_glaeubiger(
    tmp_db: sqlite3.Connection,
) -> None:
    cat = upsert_category(tmp_db, "Abos")
    # Two labelled rows under one creditor id, different names.
    _ins(tmp_db, eid=1, cents=-1299, cpn="spotify a", gid="GID1", d=date(2026, 1, 1))
    _ins(tmp_db, eid=2, cents=-1299, cpn="spotify b", gid="GID1", d=date(2026, 2, 1))
    add_label(tmp_db, 1, cat, "user")
    add_label(tmp_db, 2, cat, "user")
    # A brand-new name with no name-labels, but the same creditor id.
    assert _vendor_exact_match(tmp_db, "spotify premium neu", 0.8) is None
    hit = _vendor_exact_match(
        tmp_db, "spotify premium neu", 0.8, glaeubiger_id="GID1"
    )
    assert hit is not None and hit[0] == cat


def test_glaeubiger_tried_before_iban(tmp_db: sqlite3.Connection) -> None:
    # Creditor id points to cat A; the IBAN (shared with an unrelated
    # merchant) points to cat B. The creditor id must win.
    cat_a = upsert_category(tmp_db, "Abos")
    cat_b = upsert_category(tmp_db, "Sonstiges")
    _ins(tmp_db, eid=1, cents=-1000, cpn="m1", iban="DE_SHARED", gid="GID_A",
         d=date(2026, 1, 1))
    _ins(tmp_db, eid=2, cents=-1000, cpn="m2", iban="DE_SHARED", gid="GID_OTHER",
         d=date(2026, 1, 2))
    add_label(tmp_db, 1, cat_a, "user")   # creditor GID_A -> A
    add_label(tmp_db, 2, cat_b, "user")   # iban DE_SHARED also has a B label
    hit = _vendor_exact_match(
        tmp_db, "new name", 0.5, iban="DE_SHARED", glaeubiger_id="GID_A"
    )
    assert hit is not None and hit[0] == cat_a
