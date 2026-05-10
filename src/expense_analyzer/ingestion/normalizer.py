"""Text normalization for German bank-export records.

Two main outputs per record:
  * ``counterparty_normalized`` — the merchant/payer name, cleaned for
    consistent matching.
  * ``verwendungszweck_normalized`` — the purpose/memo, with transaction-id
    noise stripped so embeddings focus on semantics.
"""

from __future__ import annotations

import re
import unicodedata

# German legal-entity suffixes commonly appended to merchant names.
_LEGAL_SUFFIXES = {
    "gmbh",
    "ag",
    "kg",
    "ohg",
    "ug",
    "se",
    "co",
    "ltd",
    "limited",
    "inc",
    "bv",
    "sa",
    "mbh",
    "kgaa",
    "ev",
}

# Strings that often appear as trailing payment-processor noise.
_PROCESSOR_NOISE = re.compile(
    r"\b(?:visa|mastercard|maestro|paypal|klarna|sofort|sumup|adyen|stripe)\b",
    re.IGNORECASE,
)

# Long numeric blobs (IDs, transaction refs) — strip from text we feed to the embedder.
_LONG_DIGITS = re.compile(r"\b\d{6,}\b")
_IBAN_LIKE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_SEPA_REF = re.compile(r"\bSEPA[-\s]?(?:LASTSCHRIFT|UEBERWEISUNG)\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_MULTISPACE = re.compile(r"\s+")


def _fold_umlauts(s: str) -> str:
    return (
        s.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )


def _strip_punctuation(s: str) -> str:
    out_chars = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out_chars.append(ch)
        else:
            out_chars.append(" ")
    return "".join(out_chars)


def normalize_counterparty(raw: str) -> str:
    """Lowercase, fold umlauts, strip punctuation, drop legal suffixes and
    trailing IDs. Idempotent and stable: same input -> same output."""
    if not raw:
        return ""
    s = raw.strip()
    s = _fold_umlauts(s)
    s = _strip_punctuation(s.lower())
    s = _LONG_DIGITS.sub(" ", s)
    s = _PROCESSOR_NOISE.sub(" ", s)
    tokens = [t for t in s.split() if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_verwendungszweck(raw: str) -> str:
    """Lowercase + fold umlauts + mask transaction noise so the meaningful
    German wording remains for the embedder."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).strip()
    s = _fold_umlauts(s)
    s = _URL.sub(" ", s)
    s = _IBAN_LIKE.sub(" ", s)
    s = _SEPA_REF.sub(" ", s)
    s = _LONG_DIGITS.sub(" ", s)
    s = s.lower()
    s = _strip_punctuation(s)
    s = _MULTISPACE.sub(" ", s).strip()
    return s


def combined_text(counterparty_norm: str, verwendungszweck_norm: str) -> str:
    """Single string fed to the embedding model. The pipe separator gives the
    transformer a strong cue to treat the two halves distinctly."""
    cp = counterparty_norm or ""
    vz = verwendungszweck_norm or ""
    if cp and vz:
        return f"{cp} | {vz}"
    return cp or vz
