"""
prompt.py — Prompt templates for the RAG system.

Uses a FLAN-T5-optimised extractive QA format.
FLAN-T5 is a seq2seq encoder-decoder trained for extraction tasks —
it works best with short "Answer based on context" prompts.
Do NOT use summarisation-style prompts — FLAN-T5 hallucinates in that mode.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = "Answer not found in uploaded documents."

# ── FLAN-T5 extractive QA prompt ─────────────────────────────────────────────
# Keep the prompt short and direct — this matches FLAN-T5 training distribution.
_QA_TEMPLATE = """\
Answer the question based only on the context below. \
If the answer is not in the context, say "{not_found}".

Context:
{context}

Question: {question}
Answer:"""

_NO_CONTEXT_TEMPLATE = """\
Answer the question based only on the context below. \
If the answer is not in the context, say "{not_found}".

Context: [No relevant content found in the uploaded documents]

Question: {question}
Answer:"""

# ── Context budget ────────────────────────────────────────────────────────────
# FLAN-T5 has a 512-token limit (~1500 chars total).
# We spread the budget across all retrieved chunks so no single chunk dominates.
# 600 chars per chunk × top_k=2–3 chunks → ~1200 chars total, safely within budget.
_CHARS_PER_CHUNK = 600


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a FLAN-T5 optimised extractive QA prompt.

    Context is distributed evenly across retrieved chunks so that content
    from later chunks (e.g. projects at the bottom of a resume) is also
    reachable.

    Args:
        question:         The user's question.
        retrieved_chunks: List of chunk dicts with 'text' and 'source'.

    Returns:
        Formatted prompt string ready for FLAN-T5.
    """
    if not isinstance(question, str):
        logger.warning("build_prompt: question is not a string (%s) — coercing.", type(question))
        question = str(question)

    question = question.strip() or "No question provided."

    if not retrieved_chunks:
        logger.debug("build_prompt: no chunks provided — using no-context template.")
        return _NO_CONTEXT_TEMPLATE.format(
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    # Take the first _CHARS_PER_CHUNK chars from each retrieved chunk.
    # This ensures every retrieved chunk contributes context, not just the first.
    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            logger.warning("build_prompt: chunk %d is not a dict — skipping.", i)
            continue
        text = chunk.get("text", "").strip()
        if not text:
            logger.debug("build_prompt: chunk %d has empty text — skipping.", i)
            continue
        context_parts.append(text[:_CHARS_PER_CHUNK])

    if not context_parts:
        logger.warning("build_prompt: all chunks were empty — using no-context template.")
        return _NO_CONTEXT_TEMPLATE.format(
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    context_str = "\n\n".join(context_parts)
    prompt = _QA_TEMPLATE.format(
        context=context_str,
        question=question,
        not_found=NOT_FOUND_MSG,
    )
    logger.debug("build_prompt: %d chars, %d chunk(s).", len(prompt), len(context_parts))
    return prompt


def format_chat_history(history: list[dict]) -> str:
    """
    Format recent chat history for optional multi-turn context injection.

    Args:
        history: List of {'role': 'user'|'assistant', 'content': str} dicts.

    Returns:
        Formatted string of the last 4 turns.
    """
    if not history:
        return ""

    formatted: list[str] = []
    for turn in history[-4:]:  # last 4 turns only
        if not isinstance(turn, dict):
            continue
        role = "Student" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
