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

from expense_analyzer.config import VendorLookupConfig

_INDUSTRY_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("supermarkt", "lebensmittel", "rewe", "edeka", "aldi", "lidl", "penny", "netto", "kaufland"),
     "supermarket"),
    (("restaurant", "gaststaette", "imbiss", "pizzeria", "cafe", "bar"), "restaurant"),
    (("apotheke", "drogerie", "klinik", "arzt", "praxis", "krankenhaus"), "health"),
    (("bahn", "bvg", "vbb", "mvg", "tankstelle", "shell", "aral", "esso"), "transport"),
    (("hotel", "pension", "airline", "lufthansa", "flughafen"), "travel"),
    (("netflix", "spotify", "amazon prime", "disney", "magenta tv", "sky"), "streaming"),
    (("telekom", "vodafone", "o2", "1&1"), "telco"),
    (("strom", "stadtwerke", "gas", "wasser"), "utilities"),
    (("versicherung", "allianz", "axa", "huk"), "insurance"),
    (("paypal", "klarna", "amazon", "ebay"), "marketplace"),
    (("vermieter", "miete", "wohnung"), "rent"),
]


def _heuristic_industry(counterparty: str, summary: str) -> str:
    blob = f"{counterparty} {summary}".lower()
    for keys, label in _INDUSTRY_HINTS:
        if any(k in blob for k in keys):
            return label
    return "other"


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
    return VendorInfo(counterparty, r["summary"] or "", r["industry"] or "other")


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
        return VendorInfo("", "", "other")
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
