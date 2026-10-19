"""Heuristic parsing of free-text travel plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_RE_MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_RE_DAY_INLINE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*\s*)?day\s*(\d+)\b")
_RE_TIME = re.compile(r"\b(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?)\b")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[\u2013\u2014\u2022•·,;:!?()-]")
_RE_NUMBER = re.compile(r"\b(\d{1,3})(?:,(\d{3}))+\b")


@dataclass
class ParsedPlan:
    raw: str
    total_usd: float | None = None
    line_costs_usd: list[float] = field(default_factory=list)
    day_sections: int = 0
    mentions_hotel: bool = False
    mentions_flight: bool = False
    times_hhmm: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize plan text for metric computation."""
    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = _RE_NUMBER.sub(lambda m: m.group(0).replace(",", ""), normalized)
    normalized = normalized.replace("\n", " ")
    normalized = _RE_PUNCT.sub(" ", normalized)
    normalized = _RE_MULTI_SPACE.sub(" ", normalized)
    return normalized.strip()


def parse_plan_text(text: str) -> ParsedPlan:
    if not text:
        return ParsedPlan(raw="")
    line_costs = [float(x.replace(",", "")) for x in _RE_MONEY.findall(text)]
    total = float(sum(line_costs)) if line_costs else None

    day_sections = len(_RE_DAY_INLINE.findall(text))

    lower = normalize_text(text)
    mentions_hotel = any(
        w in lower for w in ("hotel", "inn", "resort", "airbnb", "lodging", "accommodation")
    )
    mentions_flight = any(
        w in lower for w in ("flight", "airline", "depart", "arrival", "airport", "plane")
    )
    times = _RE_TIME.findall(text)

    return ParsedPlan(
        raw=text,
        total_usd=total,
        line_costs_usd=line_costs,
        day_sections=day_sections,
        mentions_hotel=mentions_hotel,
        mentions_flight=mentions_flight,
        times_hhmm=times,
    )


def sum_all_money(text: str) -> float | None:
    costs = [float(x.replace(",", "")) for x in _RE_MONEY.findall(text)]
    if not costs:
        return None
    return float(sum(costs))
