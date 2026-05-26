"""
prompt.py — Prompt templates for the RAG system.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = "Answer not found in uploaded documents."

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful university course assistant. Your job is to answer student \
questions STRICTLY based on the provided context from uploaded course documents.

Rules:
- Answer ONLY from the provided context below
- Do NOT use any external knowledge
- If the answer is not in the context, reply exactly: "{not_found}"
- Be clear, structured, and educational in your response
- Use bullet points or numbered lists when explaining multi-step concepts
- Keep answers concise but complete

Context from uploaded documents:
{context}

Student Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the full prompt for the LLM.

    Args:
        question:         The student's question.
        retrieved_chunks: List of chunk dicts with 'text' and 'source'.

    Returns:
        Formatted prompt string.
    """
    if not isinstance(question, str):
        logger.warning("build_prompt: question is not a string (%s) — coercing.", type(question))
        question = str(question)

    question = question.strip() or "No question provided."

    if not retrieved_chunks:
        logger.debug("build_prompt: no chunks provided — using no-context placeholder.")
        return _SYSTEM_PROMPT_TEMPLATE.format(
            context="[No relevant context found in uploaded documents]",
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    # Build context string from chunks
    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            logger.warning("build_prompt: chunk %d is not a dict — skipping.", i)
            continue
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "").strip()
        if not text:
            logger.debug("build_prompt: chunk %d has empty text — skipping.", i)
            continue
        context_parts.append(f"[Source {i}: {source}]\n{text}")

    if not context_parts:
        logger.warning("build_prompt: all chunks were empty — using no-context placeholder.")
        return _SYSTEM_PROMPT_TEMPLATE.format(
            context="[No relevant context found in uploaded documents]",
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    context_str = "\n\n---\n\n".join(context_parts)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question,
        not_found=NOT_FOUND_MSG,
    )


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
