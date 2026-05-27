"""Opt-in vendor web lookup. **Privacy-critical.**

Rules:
  * Disabled unless ``config.vendor_lookup.enabled`` is True.
  * The ONLY field that leaves the machine is the normalized counterparty
    name (e.g. ``"rewe markt"``). The Verwendungszweck, IBAN, amount, and
    every other field MUST never appear in any outbound request.
  * Results are cached in the ``vendor_cache`` table keyed by
    ``counterparty_normalized``; subsequent lookups within
    ``cache_ttl_days`` re-use the cached result.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from expensa.config import VendorLookupConfig

# German industry labels. We feed these to a German embedding model and
# (when zeroshot prompting is on) into a German NLI premise -- the
# previous English labels ("supermarket", "transport") forced the model
# to cross-language match, weakening the boost.
_INDUSTRY_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("supermarkt", "lebensmittel", "rewe", "edeka", "aldi", "lidl", "penny", "netto", "kaufland"),
     "Supermarkt"),
    (("restaurant", "gaststaette", "imbiss", "pizzeria", "cafe", "bar"), "Restaurant"),
    (("apotheke", "drogerie", "klinik", "arzt", "praxis", "krankenhaus"), "Gesundheit"),
    (("bahn", "bvg", "vbb", "mvg", "tankstelle", "shell", "aral", "esso"), "Verkehr"),
    (("hotel", "pension", "airline", "lufthansa", "flughafen"), "Reisen"),
    (("netflix", "spotify", "amazon prime", "disney", "magenta tv", "sky"), "Streaming"),
    (("telekom", "vodafone", "o2", "1&1"), "Telekommunikation"),
    (("strom", "stadtwerke", "gas", "wasser"), "Versorgung"),
    (("versicherung", "allianz", "axa", "huk"), "Versicherung"),
    (("paypal", "klarna", "amazon", "ebay"), "Onlinehandel"),
    (("vermieter", "miete", "wohnung"), "Miete"),
]

# Sentinel returned when no hint matched. Consumers (cascade stages,
# CLI display) should treat this -- and the empty string -- as "no
# usable industry signal" and skip the enrichment.
INDUSTRY_OTHER = "Sonstige"

# Migration map: old English-labelled cache rows get translated on read
# so legacy entries don't show up untranslated in the UI/CLI or
# pollute the cascade with English tokens. Re-running ``expense vendor
# refresh`` would rewrite them, but most users won't bother.
_LEGACY_TO_GERMAN: dict[str, str] = {
    "supermarket": "Supermarkt",
    "restaurant": "Restaurant",
    "health": "Gesundheit",
    "transport": "Verkehr",
    "travel": "Reisen",
    "streaming": "Streaming",
    "telco": "Telekommunikation",
    "utilities": "Versorgung",
    "insurance": "Versicherung",
    "marketplace": "Onlinehandel",
    "rent": "Miete",
    "other": INDUSTRY_OTHER,
}


def is_meaningful_industry(industry: str | None) -> bool:
    """Return True iff this industry tag carries actionable signal.

    Empty strings and the ``Sonstige``/``other`` sentinel are filtered
    so they never make it into the cascade premise or category-
    similarity bonus -- they'd just dilute the signal.
    """
    if not industry:
        return False
    return industry.strip().lower() not in {"sonstige", "other", ""}


def normalize_industry(industry: str | None) -> str:
    """Translate legacy English labels to their German equivalents.

    Cheap O(1) lookup applied at read time so existing caches don't
    have to be wiped. Unknown values pass through untouched.
    """
    if not industry:
        return ""
    return _LEGACY_TO_GERMAN.get(industry.strip().lower(), industry)


def _heuristic_industry(counterparty: str, summary: str) -> str:
    blob = f"{counterparty} {summary}".lower()
    for keys, label in _INDUSTRY_HINTS:
        if any(k in blob for k in keys):
            return label
    return INDUSTRY_OTHER


@dataclass
class VendorInfo:
    counterparty_normalized: str
    summary: str
    industry: str


class VendorLookupDisabled(RuntimeError):
    """Raised when the caller tries to look up a vendor while the feature
    is disabled in config."""


def _cached(conn: sqlite3.Connection, counterparty: str, ttl_days: int) -> VendorInfo | None:
    r = conn.execute(
        "SELECT summary, industry, fetched_at FROM vendor_cache WHERE counterparty_normalized = ?",
        (counterparty,),
    ).fetchone()
    if r is None:
        return None
    fetched = r["fetched_at"]
    if isinstance(fetched, str):
        fetched_dt = datetime.fromisoformat(fetched.replace(" ", "T"))
    else:
        fetched_dt = fetched
    # Treat naive timestamps from sqlite as UTC.
    if fetched_dt.tzinfo is None:
        fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - fetched_dt > timedelta(days=ttl_days):
        return None
    # Migrate legacy English industry labels to German on read so the
    # UI/CLI and cascade never have to think about either dialect.
    industry = normalize_industry(r["industry"]) or INDUSTRY_OTHER
    return VendorInfo(counterparty, r["summary"] or "", industry)


def _cache_put(
    conn: sqlite3.Connection, info: VendorInfo
) -> None:
    conn.execute(
        """
        INSERT INTO vendor_cache(counterparty_normalized, summary, industry)
        VALUES (?, ?, ?)
        ON CONFLICT(counterparty_normalized) DO UPDATE SET
            summary=excluded.summary,
            industry=excluded.industry,
            fetched_at=CURRENT_TIMESTAMP
        """,
        (info.counterparty_normalized, info.summary, info.industry),
    )


def _ddg_search(query: str, max_results: int = 3) -> str:
    """Send `query` to DuckDuckGo. **Only `query` leaves the machine.**"""
    from duckduckgo_search import DDGS  # local import keeps it optional

    snippets: list[str] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results, safesearch="moderate"):
            body = (hit.get("body") or "").strip()
            if body:
                snippets.append(body)
    return " ".join(snippets)[:600]


def _searxng_search(base_url: str, query: str, max_results: int = 3) -> str:
    """Send `query` to a SearxNG instance. Only `query` leaves the machine."""
    import json as _json
    import urllib.parse
    import urllib.request

    if not base_url:
        return ""
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"}
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    snippets = [(r.get("content") or "").strip() for r in data.get("results", [])[:max_results]]
    return " ".join(s for s in snippets if s)[:600]


def lookup_vendor(
    conn: sqlite3.Connection,
    counterparty_normalized: str,
    config: VendorLookupConfig,
) -> VendorInfo:
    """Look up a vendor. Returns cached result if fresh.

    PRIVACY: only ``counterparty_normalized`` is sent over the wire. The
    Verwendungszweck, IBAN, amount, and every other field stay local.
    """
    if not config.enabled:
        raise VendorLookupDisabled(
            "vendor_lookup.enabled is False. Enable it in config to use this feature."
        )
    if not counterparty_normalized:
        return VendorInfo("", "", INDUSTRY_OTHER)
    cached = _cached(conn, counterparty_normalized, config.cache_ttl_days)
    if cached is not None:
        return cached

    # Build the outbound query out of ONLY the counterparty name. We do not
    # interpolate any other field. This is the privacy invariant.
    query = counterparty_normalized

    summary = ""
    try:
        if config.backend == "duckduckgo":
            summary = _ddg_search(query)
        elif config.backend == "searxng":
            summary = _searxng_search(config.searxng_url, query)
    except Exception:
        # Network failure shouldn't kill the pipeline; we cache an empty miss.
        summary = ""

    industry = _heuristic_industry(counterparty_normalized, summary)
    info = VendorInfo(counterparty_normalized, summary, industry)
    _cache_put(conn, info)
    return info
