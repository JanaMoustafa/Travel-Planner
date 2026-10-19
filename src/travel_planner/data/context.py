"""Prompt context helpers (no heavy dataset imports)."""

from __future__ import annotations

from typing import Any


def row_to_prompt_context(row: dict[str, Any]) -> str:
    """Build a short context string for the Logician from a HF row or metadata dict."""
    parts = [
        f"Query: {row.get('query', '')}",
        f"From: {row.get('org', '')} To: {row.get('dest', '')}",
        f"Days: {row.get('days', '')} Cities to visit: {row.get('visiting_city_number', '')}",
        f"Date: {row.get('date', '')} People: {row.get('people_number', '')}",
        f"Local constraint: {row.get('local_constraint', '')}",
        f"Level: {row.get('level', '')}",
    ]
    return "\n".join(parts)
