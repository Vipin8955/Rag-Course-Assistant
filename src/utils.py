"""
utils.py — Shared helpers and formatting utilities.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def format_score(score: float) -> str:
    """
    Format a similarity score (0.0–1.0) as a percentage string.

    Returns '0.0%' for invalid inputs instead of raising.
    """
    try:
        return f"{float(score) * 100:.1f}%"
    except (TypeError, ValueError) as e:
        logger.warning("format_score: invalid input %r (%s) — returning '0.0%%'.", score, e)
        return "0.0%"


def truncate_text(text: str, max_words: int = 60) -> str:
    """
    Truncate text to at most max_words words, appending '…' if truncated.

    Returns an empty string for non-string input.
    """
    if not isinstance(text, str):
        logger.warning("truncate_text: non-string input (%s) — returning ''.", type(text))
        return ""
    if max_words < 1:
        logger.warning("truncate_text: max_words=%d < 1 — returning ''.", max_words)
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def file_hash(filepath: str | Path) -> Optional[str]:
    """
    Compute MD5 hash of a file for change detection.

    Returns:
        Hex digest string, or None if the file is unreadable.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("file_hash: file not found: %s", filepath)
        return None
    if not filepath.is_file():
        logger.warning("file_hash: path is not a file: %s", filepath)
        return None

    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.error("file_hash: cannot read '%s': %s", filepath, e)
        return None


def timestamp() -> str:
    """Return current local time as HH:MM."""
    return time.strftime("%H:%M")


def format_chunk_for_display(chunk: dict, show_score: bool = True) -> str:
    """
    Format a chunk dict for human-readable display.

    Returns an empty string for non-dict input.
    """
    if not isinstance(chunk, dict):
        logger.warning("format_chunk_for_display: expected dict, got %s.", type(chunk))
        return ""

    source = chunk.get("source", "unknown")
    text = chunk.get("text", "")
    score = chunk.get("score", 0.0)
    wc = chunk.get("word_count", 0)

    lines = [f"📄 **{source}** | {wc} words"]
    if show_score:
        lines.append(f"🎯 Relevance: {format_score(score)}")
    lines.append("")
    lines.append(str(text))
    return "\n".join(lines)


def get_index_key(
    model_name: str,
    store_type: str,
    chunk_size: int,
    overlap: int,
    doc_names: list[str],
) -> str:
    """
    Generate a unique cache key for the current index configuration.
    Used to detect when re-indexing is needed.
    """
    try:
        doc_str = ",".join(sorted(str(n) for n in doc_names))
        return f"{model_name}_{store_type}_{chunk_size}_{overlap}_{doc_str}"
    except Exception as e:
        logger.error("get_index_key failed: %s — returning empty key.", e)
        return ""


def estimate_index_time(num_docs: int, chunk_size: int) -> str:
    """
    Rough human-readable estimate of indexing time based on doc count and chunk size.
    """
    try:
        base = int(num_docs) * (5 if int(chunk_size) <= 300 else 10)
    except (TypeError, ValueError):
        return "unknown"

    if base < 10:
        return "< 10 seconds"
    elif base < 30:
        return "10–30 seconds"
    else:
        return "30–60 seconds"
