"""Load and cache local LoRA-fine-tuned model or fall back to API.

This module provides the critical bridge between the trained LoRA model
and the inference pipeline. It was added to fix the fundamental issue
where training code existed but was never used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

_cached_models: dict[tuple[str, float, float | None, int | None, float | None], "BaseLanguageModel"] = {}


def should_use_local_model() -> bool:
    """Check if USE_LOCAL_SOLVER_MODEL env var is set.
    
    Returns:
        True if local LoRA model should be used, False to use Groq/OpenAI API.
    """
    return os.environ.get("USE_LOCAL_SOLVER_MODEL", "0").lower() in ("1", "true", "yes")


def load_local_solver_model(
    model_path: Path | str,
    temperature: float = 0.4,
    *,
    top_p: float | None = None,
    max_tokens: int | None = None,
    frequency_penalty: float | None = None,
) -> "BaseLanguageModel":
    """Load LoRA-fine-tuned Llama model from disk.
    
    This is the KEY FIX for the training integration issue. Previously,
    the LoRA model was trained but never loaded. This function:
    1. Loads base Llama-3.1-8B model with 4-bit quantization
    2. Loads LoRA adapter from disk
    3. Prepares for inference
    4. Wraps in LangChain-compatible interface
    
    Args:
        model_path: Path to directory with model weights and LoRA adapter.
                   Must contain: adapter_config.json, adapter_model.bin, etc.
        temperature: Sampling temperature for generation (0.0-1.0)
        
    Returns:
        LangChain BaseLanguageModel ready for inference
        
    Raises:
        ImportError: If unsloth, peft, or transformers not installed
        FileNotFoundError: If model_path doesn't exist
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path {model_path} does not exist. "
            f"First run: bash scripts/train_solver.sh"
        )
    
    cache_key = (
        str(model_path.resolve()),
        temperature,
        top_p,
        max_tokens,
        frequency_penalty,
    )
    if cache_key in _cached_models:
        logger.info("Using cached LoRA model for config %s", cache_key[1:])
        return _cached_models[cache_key]
    
    logger.info(f"Loading LoRA model from {model_path}")
    
    logger.info("Attempting local LoRA load via Unsloth path")

    try:
        from unsloth import FastLanguageModel

        # Load base model with LoRA adapter already integrated
        # Unsloth auto-detects and merges LoRA from the saved directory
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=4096,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )

        # Prepare for inference (no more training)
        model = FastLanguageModel.for_inference(model)

    except Exception as unsloth_error:
        logger.warning(
            "Unsloth local load failed, falling back to Transformers+PEFT: %s",
            unsloth_error,
        )

        # Fallback path avoids Unsloth runtime kernels that can require system Python headers.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        # Infer base model from adapter metadata if present.
        base_model_name = "unsloth/Llama-3.2-3B-bnb-4bit"
        try:
            import json
            adapter_cfg = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_name = adapter_cfg.get("base_model_name_or_path") or base_model_name
        except Exception:
            pass

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model.eval()

    logger.info("Wrapping model in HuggingFace pipeline for LangChain")

    # Wrap in HuggingFace pipeline
    from transformers import pipeline

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens or 512,
        temperature=temperature,
        do_sample=True,
        top_p=top_p or 0.9,
        repetition_penalty=1.0 + max(0.0, frequency_penalty or 0.0),
        return_full_text=False,
    )

    # Wrap in LangChain interface
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

    # langchain-community changed constructor kwarg across versions.
    # Try the modern `pipeline=` first, then fall back to legacy `model_pipeline=`.
    try:
        llm = HuggingFacePipeline(pipeline=text_pipeline)
    except Exception:
        llm = HuggingFacePipeline(model_pipeline=text_pipeline)
    _cached_models[cache_key] = llm
    
    logger.info(
        "LoRA model loaded and cached (temperature=%s top_p=%s max_tokens=%s frequency_penalty=%s). Ready for inference.",
        temperature,
        top_p,
        max_tokens,
        frequency_penalty,
    )
    return llm


def get_solver_llm(
    temperature: float = 0.4,
    *,
    top_p: float | None = None,
    max_tokens: int | None = None,
    frequency_penalty: float | None = None,
) -> "BaseLanguageModel":
    """Get solver LLM: local LoRA model OR Groq/OpenAI chat API.
    
    This is the key integration point. It decides whether to use:
    - Local LoRA-fine-tuned Llama model (if USE_LOCAL_SOLVER_MODEL=1)
    - Groq (default when GROQ_API_KEY is set) or OpenAI-compatible API
    
    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
        
    Returns:
        LangChain-compatible language model
    """
    if should_use_local_model():
        lora_path = Path(os.environ.get("SOLVER_LORA_PATH", "./artifacts/models/solver-lora"))
        
        logger.info(f"Using local LoRA model from training (temperature={temperature})")
        return load_local_solver_model(
            lora_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
    else:
        from travel_planner.graph.llm import get_chat_llm

        logger.info(
            "Using Groq/OpenAI chat API for solver (temperature=%s top_p=%s max_tokens=%s)",
            temperature,
            top_p,
            max_tokens,
        )
        return get_chat_llm(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
