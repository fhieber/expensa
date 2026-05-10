"""IBAN-derived features. Uses `schwifty` for validation/decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IbanInfo:
    raw: str
    country: str | None
    blz: str | None  # German bank code, when applicable
    is_valid: bool
    is_foreign: bool
    is_known_self: bool


def classify_iban(iban: str | None, own_ibans: Iterable[str] | None = None) -> IbanInfo:
    """Parse and classify an IBAN. Empty/invalid input returns an info with
    ``is_valid=False`` and ``country=None``.

    `own_ibans` is the set of the user's own IBANs (loaded from `own_ibans` table).
    Matching one means this transaction is an internal transfer.
    """
    own = {i.upper().replace(" ", "") for i in (own_ibans or [])}
    if not iban:
        return IbanInfo(raw="", country=None, blz=None, is_valid=False, is_foreign=False, is_known_self=False)

    cleaned = iban.replace(" ", "").upper()

    # Try schwifty first; fall back to a permissive parse if the lib is missing or rejects.
    country: str | None = None
    blz: str | None = None
    is_valid = False
    try:
        from schwifty import IBAN  # local import keeps top-level light

        try:
            obj = IBAN(cleaned)
            country = obj.country_code
            is_valid = True
            if country == "DE":
                blz = obj.bank_code  # 8-digit German BLZ
        except Exception:
            # malformed but maybe the prefix is still meaningful
            country = cleaned[:2] if len(cleaned) >= 2 and cleaned[:2].isalpha() else None
    except ImportError:
        country = cleaned[:2] if len(cleaned) >= 2 and cleaned[:2].isalpha() else None
        if country == "DE" and len(cleaned) >= 12:
            blz = cleaned[4:12]
            is_valid = True

    is_foreign = country is not None and country != "DE"
    is_known_self = cleaned in own
    return IbanInfo(
        raw=cleaned,
        country=country,
        blz=blz,
        is_valid=is_valid,
        is_foreign=is_foreign,
        is_known_self=is_known_self,
    )
