"""
prompt.py — Prompt templates for the RAG system
"""

from __future__ import annotations

NOT_FOUND_MSG = "Answer not found in uploaded documents."

SYSTEM_PROMPT_TEMPLATE = """You are a helpful university course assistant. Your job is to answer student questions STRICTLY based on the provided context from uploaded course documents. 

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
        question: the student's question
        retrieved_chunks: list of chunk dicts with 'text' and 'source'

    Returns:
        Formatted prompt string
    """
    if not retrieved_chunks:
        return SYSTEM_PROMPT_TEMPLATE.format(
            context="[No relevant context found in uploaded documents]",
            question=question,
            not_found=NOT_FOUND_MSG,
        )

    # Format each chunk with source attribution
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "").strip()
        context_parts.append(f"[Source {i}: {source}]\n{text}")

    context_str = "\n\n---\n\n".join(context_parts)

    return SYSTEM_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question,
        not_found=NOT_FOUND_MSG,
    )


def format_chat_history(history: list[dict]) -> str:
    """Format chat history for context (optional, for multi-turn)."""
    formatted = []
    for turn in history[-4:]:  # Last 4 turns for context
        role = "Student" if turn["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {turn['content']}")
    return "\n".join(formatted)
