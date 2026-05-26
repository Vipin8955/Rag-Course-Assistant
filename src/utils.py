"""
utils.py — Shared helpers and session state management
"""

from __future__ import annotations
import time
import hashlib
from pathlib import Path


def format_score(score: float) -> str:
    """Format similarity score as percentage."""
    return f"{score * 100:.1f}%"


def truncate_text(text: str, max_words: int = 60) -> str:
    """Truncate text to max_words for display."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def file_hash(filepath: str | Path) -> str:
    """Compute MD5 hash of a file (for change detection)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp() -> str:
    """Return current time as HH:MM."""
    return time.strftime("%H:%M")


def format_chunk_for_display(chunk: dict, show_score: bool = True) -> str:
    """Format a chunk dict for display in the UI."""
    source = chunk.get("source", "unknown")
    text = chunk.get("text", "")
    score = chunk.get("score", 0)
    wc = chunk.get("word_count", 0)

    lines = [f"📄 **{source}** | {wc} words"]
    if show_score:
        lines.append(f"🎯 Relevance: {format_score(score)}")
    lines.append("")
    lines.append(text)
    return "\n".join(lines)


def get_index_key(
    model_name: str,
    store_type: str,
    chunk_size: int,
    overlap: int,
    doc_names: list[str],
) -> str:
    """
    Generate a unique key for the current index configuration.
    Used to detect when re-indexing is needed.
    """
    doc_str = ",".join(sorted(doc_names))
    return f"{model_name}_{store_type}_{chunk_size}_{overlap}_{doc_str}"


def estimate_index_time(num_docs: int, chunk_size: int) -> str:
    """Rough estimate of indexing time."""
    base = num_docs * (5 if chunk_size == 300 else 10)
    if base < 10:
        return "< 10 seconds"
    elif base < 30:
        return "10–30 seconds"
    else:
        return "30–60 seconds"
