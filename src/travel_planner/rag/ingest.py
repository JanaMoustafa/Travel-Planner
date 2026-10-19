"""Chunk TravelPlanner `reference_information` and persist to ChromaDB."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from travel_planner.config import CHROMA_PERSIST_DIR


def _chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += max_chars
    return chunks


def _reference_to_str(ref: Any) -> str:
    if ref is None:
        return ""
    if isinstance(ref, str):
        return ref
    try:
        return json.dumps(ref, ensure_ascii=False)
    except Exception:
        return str(ref)


def ingest_dataset_to_chroma(
    dataset_rows: list[dict[str, Any]],
    *,
    persist_dir: str | None = None,
    collection_name: str = "travel_pois",
    reset: bool = False,
) -> None:
    """Embed and upsert chunks from each row's reference_information."""
    persist = str(persist_dir or CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(path=persist)
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb,
        metadata={"description": "TravelPlanner reference_information chunks"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, row in enumerate(dataset_rows):
        ref = _reference_to_str(row.get("reference_information"))
        chunks = _chunk_text(ref)
        for j, chunk in enumerate(chunks):
            uid = hashlib.sha256(f"{i}-{j}-{chunk[:64]}".encode()).hexdigest()[:16]
            ids.append(uid)
            documents.append(chunk)
            metadatas.append(
                {
                    "row_index": i,
                    "split": str(row.get("_split", "")),
                    "org": str(row.get("org", ""))[:200],
                    "dest": str(row.get("dest", ""))[:200],
                    "days": int(row.get("days") or 0) or 0,
                }
            )

    if not ids:
        return
    batch = 64
    for start in range(0, len(ids), batch):
        coll.add(
            ids=ids[start : start + batch],
            documents=documents[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )
