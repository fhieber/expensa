"""Display-time rewriting of bank fields that secondary-source
enrichment has more information about.

Pure / Streamlit-free: takes the raw values produced by the SQL
SELECT and returns the strings that should actually surface in the
UI grids. The transformation is intentionally additive -- it never
mutates DB state, and it falls through to the bank values when no
enrichment is available, so non-enriched rows are unchanged.

Current scope: PayPal Lastschrift cleanup. The bank-side counterparty
for every PayPal-routed purchase is the same generic
``"PayPal Europe S.a.r.l. et Cie ..."`` string, which kills user
signal when scanning a list. The bank-side Verwendungszweck has a
slot for the merchant after ``Ihr Einkauf bei`` that's empty about
half the time. PayPal's own CSV always carries the real merchant,
so when we've matched the row we use that to surface
``"PayPal {Merchant}"`` as the counterparty and rebuild the
Verwendungszweck as ``"{reference} Ihr Einkauf bei {Merchant}"``.
"""

from __future__ import annotations

import re
from typing import Any

# Captures the PayPal Lastschrift reference prefix, e.g.
# ``"1049005663578/PP.2467.PP/"`` (with or without a trailing ``.``).
# This is the part the bank's own clearing emits before the merchant
# slot, so we preserve it verbatim in the rewritten Verwendungszweck.
_PAYPAL_REF_PREFIX = re.compile(r"^(\s*\d+\s*/\s*PP\.\d+\.PP/?\.?)")


def _is_paypal_enriched(enrichment_source: Any, enriched_counterparty: Any) -> bool:
    """True iff the row was matched against a PayPal source CSV AND
    the match carried a usable merchant name. Defensive on both nulls
    and empty strings (SQLite returns Python ``None`` for NULL but
    pandas sometimes hands back ``float('nan')``)."""
    if enrichment_source is None or not isinstance(enrichment_source, str):
        return False
    if enrichment_source.lower() != "paypal":
        return False
    if not enriched_counterparty or not isinstance(enriched_counterparty, str):
        return False
    return bool(enriched_counterparty.strip())


def display_counterparty(
    bank_counterparty: str | None,
    enrichment_source: str | None,
    enriched_counterparty: str | None,
) -> str:
    """Return the counterparty string that should surface in the UI.

    For PayPal-enriched rows: ``"PayPal {merchant}"`` -- the
    ``PayPal`` prefix is kept so the user can still tell at a glance
    which transactions went through PayPal vs were paid directly.
    Other rows fall through to the bank-side value unchanged.
    """
    if _is_paypal_enriched(enrichment_source, enriched_counterparty):
        return f"PayPal {(enriched_counterparty or '').strip()}"
    return (bank_counterparty or "").strip() or ""


def display_verwendungszweck(
    bank_verwendungszweck: str | None,
    enrichment_source: str | None,
    enriched_counterparty: str | None,
) -> str:
    """Rebuild the Verwendungszweck for PayPal-enriched rows.

    Format: ``"{reference-prefix} Ihr Einkauf bei {merchant}"``,
    where the reference-prefix is captured from the bank's own
    string (``1049005663578/PP.2467.PP/.``) so the user can still
    cross-reference the row against their bank statement. Falls
    back to the bank value when:
      * the row isn't paypal-enriched, OR
      * the reference prefix doesn't match the expected shape
        (which means it's not a PayPal Lastschrift the way we
        understand them -- safest to leave untouched).
    """
    bank_vz = (bank_verwendungszweck or "").strip()
    if not _is_paypal_enriched(enrichment_source, enriched_counterparty):
        return bank_vz
    m = _PAYPAL_REF_PREFIX.match(bank_vz)
    if not m:
        # Unfamiliar shape -- don't pretend we know how to rebuild it.
        return bank_vz
    prefix = m.group(1).rstrip(". ")
    merchant = (enriched_counterparty or "").strip()
    return f"{prefix} Ihr Einkauf bei {merchant}"


def apply_to_dataframe(df, *, counterparty_col: str = "counterparty",
                       verwendungszweck_col: str = "verwendungszweck") -> None:
    """Mutate ``df`` in place so the counterparty + Verwendungszweck
    columns reflect the enrichment-aware display values.

    Requires the DataFrame to also carry ``enrichment_source`` and
    ``enriched_counterparty`` columns (the UI SQL pulls them for this
    purpose). Missing columns: silently no-op so the helper is safe
    to call from contexts where the SQL hasn't been updated yet.
    """
    if "enrichment_source" not in df.columns or "enriched_counterparty" not in df.columns:
        return
    if counterparty_col in df.columns:
        df[counterparty_col] = [
            display_counterparty(cp, src, ecp)
            for cp, src, ecp in zip(
                df[counterparty_col],
                df["enrichment_source"],
                df["enriched_counterparty"],
                strict=True,
            )
        ]
    if verwendungszweck_col in df.columns:
        df[verwendungszweck_col] = [
            display_verwendungszweck(vz, src, ecp)
            for vz, src, ecp in zip(
                df[verwendungszweck_col],
                df["enrichment_source"],
                df["enriched_counterparty"],
                strict=True,
            )
        ]


__all__ = (
    "apply_to_dataframe",
    "display_counterparty",
    "display_verwendungszweck",
)
