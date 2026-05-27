"""Date-range preset radio used by both the Dashboard and Data tabs.

The mapping is table-driven: adding a preset is one line in
``_PRESET_RANGES``. Typos raise ``KeyError`` instead of silently
mapping to "no filter".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as _date_cls
from datetime import timedelta

# Each preset returns a (since, until) tuple given today's date. The
# labels say "Past N days" because the actual semantics are "today - N
# days" -- not previous-calendar-month boundaries. Truth in labelling.
_PRESET_RANGES: dict[
    str, Callable[[_date_cls], tuple[_date_cls | None, _date_cls | None]]
] = {
    "Past 30 days":   lambda today: (today - timedelta(days=30), today),
    "Past 90 days":   lambda today: (today - timedelta(days=90), today),
    "Past 180 days":  lambda today: (today - timedelta(days=180), today),
    "YTD":            lambda today: (_date_cls(today.year, 1, 1), today),
    "Past 12 months": lambda today: (today - timedelta(days=365), today),
    "Past 24 months": lambda today: (today - timedelta(days=730), today),
    "All-time":       lambda today: (None, None),
}
PRESETS: list[str] = list(_PRESET_RANGES) + ["Custom"]
DEFAULT_PRESET = "Past 90 days"


def resolve_range(
    preset: str,
    custom_from: _date_cls | None = None,
    custom_to: _date_cls | None = None,
) -> tuple[_date_cls | None, _date_cls | None]:
    """Resolve a preset name to ``(since, until)``.

    Raises ``KeyError`` on unknown preset names so typos surface
    immediately rather than silently disabling the date filter.
    """
    if preset == "Custom":
        return custom_from, custom_to
    return _PRESET_RANGES[preset](_date_cls.today())
