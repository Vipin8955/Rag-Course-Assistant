"""
generator.py — LLM text generation using HuggingFace transformers.
Supports FLAN-T5-Base (fast, CPU-friendly) and FLAN-T5-Large (more accurate).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.prompt import NOT_FOUND_MSG

logger = logging.getLogger(__name__)

# ── Available models ──────────────────────────────────────────────────────────
AVAILABLE_LLMS: dict[str, str] = {
    "FLAN-T5-Base (Fast)": "google/flan-t5-base",
    "FLAN-T5-Large": "google/flan-t5-large",
}

DEFAULT_LLM = "FLAN-T5-Base (Fast)"

# Lazy pipeline cache — loaded on first use to keep startup fast
_pipeline = None
_current_model: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_pipeline(model_name: str):
    """
    Load (or return cached) HuggingFace text2text-generation pipeline.

    Raises:
        ImportError:  If transformers / torch is not installed.
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    global _pipeline, _current_model

    if _pipeline is not None and _current_model == model_name:
        return _pipeline

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required. "
            "Run: pip install transformers torch"
        ) from e

    model_id = AVAILABLE_LLMS.get(model_name, model_name)
    logger.info("Loading LLM '%s' (%s)…", model_name, model_id)

    try:
        from src.config import HF_TOKEN
        token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

        tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **token_kwargs)

        _pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,       # gentle penalty — stops runaway loops without cutting sentences short
            no_repeat_ngram_size=3,        # blocks repeated 3-word sequences
        )
        _current_model = model_name
        logger.info("LLM '%s' loaded successfully.", model_name)
        return _pipeline

    except OSError as e:
        raise RuntimeError(
            f"Could not download/load model '{model_id}'. "
            "Ensure you have an internet connection on first run. "
            f"Detail: {e}"
        ) from e
    except MemoryError:
        raise RuntimeError(
            f"Out of memory loading '{model_id}'. "
            "Try FLAN-T5-Base (Fast) instead of Large, or free up RAM."
        )
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading LLM '{model_id}': {e}"
        ) from e


def unload_model() -> None:
    """Unload the cached pipeline to free memory."""
    global _pipeline, _current_model
    _pipeline = None
    _current_model = None
    logger.info("LLM pipeline unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
#  Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    prompt: str,
    model_name: str = DEFAULT_LLM,
    max_tokens: int = 512,
    retrieved_chunks: Optional[list[dict]] = None,
) -> str:
    """
    Generate an answer from the LLM using the given prompt.

    Args:
        prompt:           Full RAG prompt string.
        model_name:       Key from AVAILABLE_LLMS.
        max_tokens:       Max new tokens to generate.
        retrieved_chunks: If provided and empty, returns NOT_FOUND_MSG immediately
                          without loading the model.

    Returns:
        Generated answer string, or an error/fallback message.
    """
    # Fast path: no relevant chunks were retrieved
    if retrieved_chunks is not None and len(retrieved_chunks) == 0:
        logger.info("generate_answer: no chunks retrieved — returning NOT_FOUND_MSG.")
        return NOT_FOUND_MSG

    if not prompt or not prompt.strip():
        logger.warning("generate_answer: empty prompt received.")
        return NOT_FOUND_MSG

    if max_tokens < 1:
        logger.warning("max_tokens=%d is < 1 — using default 256.", max_tokens)
        max_tokens = 256

    try:
        pipe = _load_pipeline(model_name)
        truncated_prompt = _truncate_prompt(prompt, max_chars=2048)
        output = pipe(truncated_prompt, max_new_tokens=max_tokens)
        answer = output[0]["generated_text"].strip()

        if not answer or len(answer) < 2:
            logger.warning("LLM returned empty answer — using fallback.")
            return NOT_FOUND_MSG

        logger.debug("generate_answer: %d chars generated.", len(answer))
        return answer

    except ImportError as e:
        logger.error("LLM dependency missing: %s", e)
        return f"⚠️ Missing dependency: {e}"
    except RuntimeError as e:
        logger.error("LLM load/generation error: %s", e)
        return f"⚠️ Model error: {e}\n\nPlease check your internet connection or try FLAN-T5-Base."
    except Exception as e:
        logger.error("Unexpected generation error: %s", e)
        return f"⚠️ Unexpected error during generation: {e}"


def warmup_model(model_name: str = DEFAULT_LLM) -> bool:
    """
    Pre-load the model so the first real query is faster.

    Returns:
        True if loaded successfully, False otherwise.
    """
    try:
        _load_pipeline(model_name)
        logger.info("Model warmup complete for '%s'.", model_name)
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning("Model warmup failed for '%s': %s", model_name, e)
        return False
    except Exception as e:
        logger.error("Unexpected warmup error: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_prompt(prompt: str, max_chars: int = 2048) -> str:
    """
    Truncate prompt to fit within the model's practical context window.
    Preserves lines from the end (question + recent context) where possible.
    """
    if not isinstance(prompt, str):
        logger.warning("_truncate_prompt: non-string input (%s).", type(prompt))
        return str(prompt)[:max_chars]

    if len(prompt) <= max_chars:
        return prompt

    logger.debug("Truncating prompt from %d to %d chars.", len(prompt), max_chars)
    lines = prompt.split("\n")
    result_lines: list[str] = []
    budget = max_chars

    for line in reversed(lines):
        line_len = len(line) + 1  # +1 for newline
        if budget - line_len > 0:
            result_lines.insert(0, line)
            budget -= line_len
        else:
            break

    return "\n".join(result_lines)
