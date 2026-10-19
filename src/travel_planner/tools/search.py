"""Tavily web search wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from travel_planner.config import TAVILY_API_KEY

_CACHE_DIR = Path("artifacts/cache/tavily")


def _cache_path(query: str, max_results: int) -> Path:
    digest = hashlib.sha1(f"{max_results}:{query}".encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{digest}.txt"


def tavily_search(query: str, *, api_key: str | None = None, max_results: int = 5) -> str:
    key = api_key or TAVILY_API_KEY
    if not key:
        return "[Tavily] TAVILY_API_KEY not set."
    cache_file = _cache_path(query, max_results)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
    }
    r = httpx.post("https://api.tavily.com/search", json=payload, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    answer = data.get("answer") or ""
    lines: list[str] = []
    if answer:
        lines.append(f"Answer: {answer}")
    for i, res in enumerate(data.get("results") or [], 1):
        title = res.get("title", "")
        content = res.get("content", "")
        url = res.get("url", "")
        lines.append(f"{i}. {title}\n   {content}\n   {url}")
    result = "\n".join(lines)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result, encoding="utf-8")
    return result


def web_search(query: str) -> str:
    """Search the web via Tavily."""
    if TAVILY_API_KEY:
        try:
            return tavily_search(query)
        except Exception as e:
            return f"[Tavily error] {e}"
    return "No search API configured (set TAVILY_API_KEY)."
