"""Retrieve city-scoped POI chunks from Chroma."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from travel_planner.config import CHROMA_PERSIST_DIR


def get_client(persist_dir: str | None = None) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_dir or CHROMA_PERSIST_DIR))


def get_collection(
    persist_dir: str | None = None,
    collection_name: str = "travel_pois",
):
    client = get_client(persist_dir)
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb,
    )


def retrieve_pois(
    query: str,
    *,
    city: str | None = None,
    k: int = 5,
    persist_dir: str | None = None,
    collection_name: str = "travel_pois",
) -> str:
    """Return top-k passages as a single string for the LLM."""
    coll = get_collection(persist_dir=persist_dir, collection_name=collection_name)
    where: dict[str, Any] | None = None
    if city:
        where = {"dest": {"$eq": city}}

    if where:
        try:
            res = coll.query(
                query_texts=[query],
                n_results=k,
                where=where,
            )
        except Exception:
            res = coll.query(query_texts=[query], n_results=k)
    else:
        res = coll.query(query_texts=[query], n_results=k)

    docs = (res.get("documents") or [[]])[0]
    if not docs:
        return ""
    parts = [f"[{i+1}] {d}" for i, d in enumerate(docs)]
    return "\n\n".join(parts)


def get_retriever(
    persist_dir: str | None = None,
    collection_name: str = "travel_pois",
):
    """Return a callable (query, city) -> str for tools."""

    def _fn(q: str, city: str | None = None, k: int = 5) -> str:
        return retrieve_pois(q, city=city, k=k, persist_dir=persist_dir, collection_name=collection_name)

    return _fn
