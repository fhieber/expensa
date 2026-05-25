"""Tests for the display-time PayPal rewrite (enrichment/display.py)."""

from __future__ import annotations

import pandas as pd

from expense_analyzer.enrichment.display import (
    apply_to_dataframe,
    display_counterparty,
    display_verwendungszweck,
)

# Real-world bank Verwendungszweck shapes captured from PayPal Lastschrift
# rows. Both with and without the merchant slot filled in.
_VZ_WITH_MERCHANT = "1050302908515/PP.2467.PP/. betterplace.org gGmbH, Ihr Einkauf bei betterplace.org gGmbH"
_VZ_EMPTY_SLOT = "1049005663578/PP.2467.PP/. , Ihr Einkauf bei"
_BANK_CP_PAYPAL = "PayPal Europe S.a.r.l. et Cie S.C.A 22-24 Boulevard Royal, 2449 Luxembourg"


# ─── display_counterparty ────────────────────────────────────────────


def test_counterparty_uses_paypal_prefix_when_enriched() -> None:
    """The user wants "PayPal {merchant}" so PayPal-routed transactions
    stay visually distinct from direct card purchases at the same merchant."""
    assert (
        display_counterparty(_BANK_CP_PAYPAL, "paypal", "Etsy Inc")
        == "PayPal Etsy Inc"
    )


def test_counterparty_falls_back_to_bank_when_not_enriched() -> None:
    """Non-enriched rows (or rows enriched from non-PayPal sources)
    keep the bank-side counterparty unchanged."""
    assert (
        display_counterparty(_BANK_CP_PAYPAL, None, None)
        == _BANK_CP_PAYPAL
    )
    assert (
        display_counterparty("REWE Markt GmbH", None, None)
        == "REWE Markt GmbH"
    )


def test_counterparty_ignores_empty_enriched_name() -> None:
    """An enrichment row that happened to land without a usable Name
    must NOT produce "PayPal " with a trailing space."""
    assert (
        display_counterparty(_BANK_CP_PAYPAL, "paypal", "   ")
        == _BANK_CP_PAYPAL
    )
    assert (
        display_counterparty(_BANK_CP_PAYPAL, "paypal", "")
        == _BANK_CP_PAYPAL
    )


def test_counterparty_other_enrichment_source_passes_through() -> None:
    """Future secondary sources (e.g. an Amazon export) must not get
    the "PayPal " prefix glued on top -- the prefix is intentionally
    paypal-specific."""
    assert (
        display_counterparty("Amazon EU SARL", "amazon", "Some Real Merchant")
        == "Amazon EU SARL"
    )


# ─── display_verwendungszweck ────────────────────────────────────────


def test_verwendungszweck_rebuilt_with_merchant_in_empty_slot() -> None:
    """The headline case: bank's Verwendungszweck has an empty merchant
    slot. After rewrite the reference prefix is preserved and the
    merchant from enrichment fills the trailing "Ihr Einkauf bei"."""
    out = display_verwendungszweck(_VZ_EMPTY_SLOT, "paypal", "Apple Services")
    assert out == "1049005663578/PP.2467.PP/ Ihr Einkauf bei Apple Services"


def test_verwendungszweck_rebuilt_overrides_partial_bank_merchant() -> None:
    """Even when the bank already wrote a merchant into the slot, we
    overwrite with the enriched one (PayPal's truth wins -- the bank
    sometimes truncates merchant names mid-word)."""
    out = display_verwendungszweck(_VZ_WITH_MERCHANT, "paypal", "betterplace.org gGmbH")
    assert out == "1050302908515/PP.2467.PP/ Ihr Einkauf bei betterplace.org gGmbH"


def test_verwendungszweck_unchanged_when_not_enriched() -> None:
    """No enrichment → keep the bank value verbatim."""
    assert (
        display_verwendungszweck(_VZ_EMPTY_SLOT, None, None) == _VZ_EMPTY_SLOT
    )


def test_verwendungszweck_left_alone_when_pattern_doesnt_match() -> None:
    """If the bank Verwendungszweck doesn't look like a PayPal
    Lastschrift (no `/PP.\\d+.PP/` prefix), don't pretend we know
    how to rebuild it -- safer to fall back to the original."""
    weird = "SEPA-Überweisung an Vermieter Schmidt Miete Januar"
    assert display_verwendungszweck(weird, "paypal", "Vermieter Schmidt") == weird


def test_verwendungszweck_handles_other_sources() -> None:
    """Same reasoning as the counterparty case: non-paypal enrichment
    sources don't trigger the PayPal-shaped rewrite."""
    assert (
        display_verwendungszweck(_VZ_WITH_MERCHANT, "amazon", "Anything")
        == _VZ_WITH_MERCHANT
    )


# ─── apply_to_dataframe ──────────────────────────────────────────────


def test_apply_to_dataframe_rewrites_only_paypal_rows() -> None:
    """End-to-end on a small DataFrame: paypal-enriched rows are
    rewritten; everything else stays put."""
    df = pd.DataFrame({
        "counterparty": [
            _BANK_CP_PAYPAL,    # enriched -> "PayPal Etsy Inc"
            _BANK_CP_PAYPAL,    # not enriched
            "REWE Markt GmbH",  # not enriched, non-paypal
        ],
        "verwendungszweck": [
            _VZ_EMPTY_SLOT,
            _VZ_WITH_MERCHANT,
            "REWE Lebensmittel",
        ],
        "enrichment_source": ["paypal", None, None],
        "enriched_counterparty": ["Etsy Inc", None, None],
    })
    apply_to_dataframe(df)
    assert df.loc[0, "counterparty"] == "PayPal Etsy Inc"
    assert df.loc[0, "verwendungszweck"] == (
        "1049005663578/PP.2467.PP/ Ihr Einkauf bei Etsy Inc"
    )
    # Untouched rows.
    assert df.loc[1, "counterparty"] == _BANK_CP_PAYPAL
    assert df.loc[1, "verwendungszweck"] == _VZ_WITH_MERCHANT
    assert df.loc[2, "counterparty"] == "REWE Markt GmbH"
    assert df.loc[2, "verwendungszweck"] == "REWE Lebensmittel"


def test_apply_to_dataframe_no_op_when_columns_missing() -> None:
    """Caller might be working with a SELECT that doesn't pull the
    enrichment columns yet. Helper must silently skip rather than
    raise so the UI doesn't break during a partial rollout."""
    df = pd.DataFrame({
        "counterparty": ["REWE"],
        "verwendungszweck": ["Lebensmittel"],
    })
    apply_to_dataframe(df)  # no enrichment_source / enriched_counterparty
    assert df.loc[0, "counterparty"] == "REWE"
    assert df.loc[0, "verwendungszweck"] == "Lebensmittel"
