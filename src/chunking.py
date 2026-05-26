"""
chunking.py — Word-based text splitting with overlapping windows.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Acceptable chunk size bounds
_MIN_CHUNK_SIZE = 10
_MAX_CHUNK_SIZE = 2000
_MIN_OVERLAP = 0
_MAX_OVERLAP_RATIO = 0.9  # overlap must be < 90% of chunk_size


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def word_tokenize(text: str) -> list[str]:
    """Split text into words by whitespace."""
    if not isinstance(text, str):
        logger.warning("word_tokenize expected str, got %s — returning [].", type(text).__name__)
        return []
    return text.split()


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 75,
    source: str = "unknown",
) -> list[dict]:
    """
    Split `text` into overlapping word-based chunks.

    Args:
        text:       Raw text to split.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.
        source:     Source filename/label attached to each chunk.

    Returns:
        List of chunk dicts with keys:
            chunk_id, source, text, word_count, start_word.

    Raises:
        ValueError: If chunk_size or overlap are out of valid range.
    """
    # ── Validate params ───────────────────────────────────────────────────────
    if not isinstance(chunk_size, int) or chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be an int >= {_MIN_CHUNK_SIZE}, got {chunk_size!r}."
        )
    if chunk_size > _MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be <= {_MAX_CHUNK_SIZE}, got {chunk_size}."
        )
    if not isinstance(overlap, int) or overlap < _MIN_OVERLAP:
        raise ValueError(
            f"overlap must be a non-negative int, got {overlap!r}."
        )
    if overlap >= chunk_size * _MAX_OVERLAP_RATIO:
        logger.warning(
            "overlap=%d is >= %.0f%% of chunk_size=%d — adjusting to avoid tiny advances.",
            overlap, _MAX_OVERLAP_RATIO * 100, chunk_size,
        )
        overlap = max(0, int(chunk_size * _MAX_OVERLAP_RATIO) - 1)

    if not text or not text.strip():
        return []

    words = word_tokenize(text)
    if not words:
        return []

    advance = chunk_size - overlap
    if advance <= 0:
        advance = 1  # safety guard — always move forward

    chunks: list[dict] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": str(source),
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words),
                "start_word": start,
            }
        )
        start += advance
        chunk_id += 1

    logger.debug(
        "chunk_text: %d words → %d chunks (size=%d, overlap=%d, source=%s)",
        len(words), len(chunks), chunk_size, overlap, source,
    )
    return chunks


def chunk_documents(
    docs: dict[str, str],
    chunk_size: int = 300,
    overlap: int = 75,
) -> list[dict]:
    """
    Chunk all documents in a dict.

    Args:
        docs:       {filename: text} mapping.
        chunk_size: Words per chunk.
        overlap:    Word overlap between consecutive chunks.

    Returns:
        Flat list of chunk dicts from all documents.
        Documents that fail are skipped with a warning.
    """
    if not docs:
        logger.warning("chunk_documents received an empty docs dict.")
        return []

    all_chunks: list[dict] = []

    for source, text in docs.items():
        try:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=source)
            all_chunks.extend(chunks)
            logger.debug("'%s' → %d chunks.", source, len(chunks))
        except ValueError as e:
            logger.error("Chunking config error for '%s': %s", source, e)
            raise  # re-raise config errors — caller should fix params
        except Exception as e:
            logger.warning("Unexpected error chunking '%s': %s — skipping.", source, e)

    logger.info(
        "chunk_documents: %d docs → %d total chunks (size=%d, overlap=%d).",
        len(docs), len(all_chunks), chunk_size, overlap,
    )
    return all_chunks


def get_chunk_stats(chunks: list[dict]) -> dict:
    """
    Return basic statistics about a list of chunks.

    Returns:
        Dict with keys: total, avg_words, min_words, max_words.
    """
    if not chunks:
        return {"total": 0, "avg_words": 0, "min_words": 0, "max_words": 0}

    word_counts = [c.get("word_count", 0) for c in chunks]
    return {
        "total": len(chunks),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }
