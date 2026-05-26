"""
generator.py — LLM text generation using HuggingFace transformers
Supports FLAN-T5 (lightweight, fast) and optionally Mistral-small
"""

from __future__ import annotations
import re
from src.prompt import NOT_FOUND_MSG

# Lazy imports to avoid slow startup
_pipeline = None
_current_model = None

AVAILABLE_LLMS = {
    "FLAN-T5-Base (Fast)": "google/flan-t5-base",
    "FLAN-T5-Large": "google/flan-t5-large",
}

DEFAULT_LLM = "FLAN-T5-Base (Fast)"


def _load_pipeline(model_name: str):
    """Load the HuggingFace text2text pipeline."""
    global _pipeline, _current_model
    if _pipeline is not None and _current_model == model_name:
        return _pipeline

    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = AVAILABLE_LLMS.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    _pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    _current_model = model_name
    return _pipeline


def generate_answer(
    prompt: str,
    model_name: str = DEFAULT_LLM,
    max_tokens: int = 512,
    retrieved_chunks: list[dict] | None = None,
) -> str:
    """
    Generate an answer using the selected LLM.

    Falls back gracefully if context is missing.
    """
    # If no chunks were retrieved, return the standard not-found message
    if retrieved_chunks is not None and len(retrieved_chunks) == 0:
        return NOT_FOUND_MSG

    try:
        pipe = _load_pipeline(model_name)
        # FLAN-T5 works best with shorter prompts; truncate context if needed
        truncated_prompt = _truncate_prompt(prompt, max_chars=2048)
        output = pipe(truncated_prompt, max_new_tokens=max_tokens)
        answer = output[0]["generated_text"].strip()

        # Post-process: if model returned empty or gibberish, use fallback
        if not answer or len(answer) < 5:
            return NOT_FOUND_MSG

        return answer

    except Exception as e:
        return f"⚠️ Generation error: {str(e)}\n\nPlease check model availability."


def _truncate_prompt(prompt: str, max_chars: int = 2048) -> str:
    """Truncate prompt to fit within model's context window."""
    if len(prompt) <= max_chars:
        return prompt
    # Keep the question part (last ~200 chars) + truncated context
    lines = prompt.split("\n")
    # Find the question line
    result_lines = []
    char_budget = max_chars
    for line in reversed(lines):
        if char_budget - len(line) - 1 > 0:
            result_lines.insert(0, line)
            char_budget -= len(line) + 1
        else:
            break
    return "\n".join(result_lines)


def warmup_model(model_name: str = DEFAULT_LLM):
    """Pre-load the model so first query is fast."""
    try:
        _load_pipeline(model_name)
        return True
    except Exception:
        return False
