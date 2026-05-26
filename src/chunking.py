"""
chunking.py — Text splitting into overlapping word-based chunks
"""

from __future__ import annotations
import re


def word_tokenize(text: str) -> list[str]:
    """Split text into words (simple whitespace split)."""
    return text.split()


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 75,
    source: str = "unknown",
) -> list[dict]:
    """
    Split `text` into overlapping chunks of `chunk_size` words.

    Returns a list of dicts:
        {
            "chunk_id": int,
            "source": str,
            "text": str,
            "word_count": int,
            "start_word": int,
        }
    """
    words = word_tokenize(text)
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": chunk_text_str,
                "word_count": len(chunk_words),
                "start_word": start,
            }
        )

        # Advance by chunk_size - overlap so next chunk overlaps
        advance = chunk_size - overlap
        if advance <= 0:
            advance = 1  # safety guard
        start += advance
        chunk_id += 1

    return chunks


def chunk_documents(
    docs: dict[str, str],
    chunk_size: int = 300,
    overlap: int = 75,
) -> list[dict]:
    """
    Chunk all documents.

    Args:
        docs: {filename: text}
        chunk_size: words per chunk (300 or 700)
        overlap: word overlap between consecutive chunks

    Returns:
        Flat list of chunk dicts from all documents.
    """
    all_chunks = []
    for source, text in docs.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=source)
        all_chunks.extend(chunks)
    return all_chunks


def get_chunk_stats(chunks: list[dict]) -> dict:
    """Return basic stats about a list of chunks."""
    if not chunks:
        return {"total": 0, "avg_words": 0, "min_words": 0, "max_words": 0}
    word_counts = [c["word_count"] for c in chunks]
    return {
        "total": len(chunks),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }
