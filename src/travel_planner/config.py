"""Environment-driven configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()           # load .env
load_dotenv(".env.local", override=True)  # .env.local overrides .env


def _get(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return v


# Primary LLM: Groq (free tier). Fallback: OpenAI-compatible API.
GROQ_API_KEY: str | None = _get("GROQ_API_KEY")
GROQ_MODEL: str = _get("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant"

OPENAI_API_KEY: str | None = _get("OPENAI_API_KEY")
OPENAI_BASE_URL: str | None = _get("OPENAI_BASE_URL")
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
PREFER_OPENAI: bool = (_get("PREFER_OPENAI", "0") or "0").lower() in ("1", "true", "yes")

TAVILY_API_KEY: str | None = _get("TAVILY_API_KEY")

HF_TOKEN: str | None = _get("HF_TOKEN")

def has_chat_llm_credentials() -> bool:
    """True if Groq or OpenAI key is configured for agent LLM calls."""
    return bool(GROQ_API_KEY or OPENAI_API_KEY)

CHROMA_PERSIST_DIR: Path = Path(_get("CHROMA_PERSIST_DIR", "./chroma_db") or "./chroma_db")
SOLVER_LORA_PATH: Path = Path(_get("SOLVER_LORA_PATH", "./artifacts/models/solver-lora") or "./artifacts/models/solver-lora")

MAX_AUDIT_ITERATIONS: int = int(_get("MAX_AUDIT_ITERATIONS", "3") or "3")

# Fine-tuning defaults (override via CLI)
# Prefer Unsloth 4-bit checkpoints for QLoRA; override via CLI / env.
DEFAULT_SFT_MODEL_NAME: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
DEFAULT_OUTPUT_DIR: Path = Path("./artifacts/solver-lora")
