"""Shared chat LLM factory: Groq (primary) with OpenAI-compatible fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from travel_planner.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    PREFER_OPENAI,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_llm(
    model: str | None = None,
    temperature: float = 0.2,
    *,
    top_p: float | None = None,
    max_tokens: int | None = None,
    frequency_penalty: float | None = None,
) -> "BaseChatModel":
    """Return Groq ChatGroq when `GROQ_API_KEY` is set; else ChatOpenAI if `OPENAI_API_KEY` is set."""
    kwargs: dict[str, Any] = {
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty

    def _openai_chat():
        from langchain_openai import ChatOpenAI

        openai_kwargs: dict[str, Any] = {
            "model": model or OPENAI_MODEL,
            "temperature": temperature,
            "api_key": OPENAI_API_KEY,
            **kwargs,
        }
        if OPENAI_BASE_URL:
            openai_kwargs["base_url"] = OPENAI_BASE_URL
        return ChatOpenAI(**openai_kwargs)

    if OPENAI_API_KEY and PREFER_OPENAI:
        return _openai_chat()

    if GROQ_API_KEY:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or GROQ_MODEL,
            api_key=GROQ_API_KEY,
            **kwargs,
        )

    if OPENAI_API_KEY:
        return _openai_chat()

    raise RuntimeError(
        "No LLM API configured. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY in the environment."
    )
