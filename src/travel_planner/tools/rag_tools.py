"""LangChain-compatible tool wrappers for POI retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from travel_planner.rag.retrieve import get_retriever

_CACHE_DIR = Path("artifacts/cache/rag")


def make_rag_tool(
    persist_dir: str | None = None,
    collection_name: str = "travel_pois",
):
    """Build a @tool function for POI retrieval."""
    retriever = get_retriever(persist_dir=persist_dir, collection_name=collection_name)

    @tool
    def search_pois(query: str, city: str | None = None, k: int = 5) -> str:
        """Retrieve city-specific points of interest and reference travel info from the vector DB."""
        return retriever(query, city=city, k=k)

    return search_pois


def rag_lookup(query: str, city: str | None = None, k: int = 5, **kwargs: Any) -> str:
    """Imperative helper (no LangChain tool decorator)."""
    digest = hashlib.sha1(f"{query}|{city}|{k}".encode("utf-8")).hexdigest()
    cache_file = _CACHE_DIR / f"{digest}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    persist = kwargs.get("persist_dir")
    collection = kwargs.get("collection_name", "travel_pois")
    retriever = get_retriever(persist_dir=persist, collection_name=collection)
    result = retriever(query, city=city, k=k)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result, encoding="utf-8")
    return result
