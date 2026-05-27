"""
prompt.py — Prompt templates for the RAG system.

Uses a FLAN-T5-optimised extractive QA format.
FLAN-T5 is a seq2seq model trained on short extractive tasks —
it works best with "Answer based on context" style prompts,
NOT long role-play / instruction-following prompts (those cause loops).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = "Answer not found in uploaded documents."

# ── FLAN-T5 optimised prompt ──────────────────────────────────────────────────
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

# FLAN-T5 has a 512 token limit (~1500 chars total).
# 900 chars for context leaves headroom for the template + question.
_MAX_CONTEXT_CHARS = 900

# ── Summary prompt (for 'what is this about / summarise' questions) ───────────
_SUMMARY_TEMPLATE = """\
Summarise the following document in 2-3 sentences.

Document:
{context}

Summary:"""

# Keywords that signal the user wants a summary, not a fact lookup
_SUMMARY_KEYWORDS = (
    "what is this", "what is the document", "what is this document",
    "what is this pdf", "what is this file", "what is this about",
    "summarise", "summarize", "give me a summary", "overview",
    "tell me about this", "describe this", "what does this contain",
    "is this a resume", "is this a cv", "what type of document",
    "what kind of document",
)


def _is_summary_question(question: str) -> bool:
    """Return True if the question is asking for a document overview/summary."""
    q = question.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a FLAN-T5 optimised extractive QA prompt.

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

    # Build context — concatenate chunk texts up to character budget
    context_parts: list[str] = []
    total_chars = 0
    for i, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            logger.warning("build_prompt: chunk %d is not a dict — skipping.", i)
            continue
        text = chunk.get("text", "").strip()
        if not text:
            logger.debug("build_prompt: chunk %d has empty text — skipping.", i)
            continue
        if total_chars + len(text) > _MAX_CONTEXT_CHARS:
            # Trim last chunk to fit within budget
            remaining = _MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:          # only add if meaningful
                context_parts.append(text[:remaining])
            break
        context_parts.append(text)
        total_chars += len(text)

    if not context_parts:
        logger.warning("build_prompt: all chunks were empty — using no-context template.")
        return _NO_CONTEXT_TEMPLATE.format(
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    context_str = "\n\n".join(context_parts)

    # Route summary-intent questions to the summarisation template
    if _is_summary_question(question):
        logger.debug("build_prompt: summary-intent detected — using summarisation prompt.")
        return _SUMMARY_TEMPLATE.format(context=context_str[:_MAX_CONTEXT_CHARS])

    prompt = _QA_TEMPLATE.format(
        context=context_str,
        question=question,
        not_found=NOT_FOUND_MSG,
    )
    logger.debug("build_prompt: %d chars, %d chunks.", len(prompt), len(context_parts))
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
