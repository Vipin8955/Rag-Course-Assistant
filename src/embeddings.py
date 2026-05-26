"""
embeddings.py — Sentence embedding model wrappers.
Supports: all-MiniLM-L6-v2 (fast) and BAAI/bge-base-en-v1.5 (accurate).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Available models ──────────────────────────────────────────────────────────
AVAILABLE_MODELS: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}

# Model cache: avoid re-loading across calls
_model_cache: dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def get_model(model_name: str):
    """
    Load (or return cached) SentenceTransformer model.

    Args:
        model_name: Key from AVAILABLE_MODELS, or a raw HuggingFace model ID.

    Returns:
        Loaded SentenceTransformer instance.

    Raises:
        ImportError:  If sentence-transformers is not installed.
        RuntimeError: If the model cannot be loaded (network error, bad ID, etc.).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from e

    if model_name in _model_cache:
        return _model_cache[model_name]

    model_id = AVAILABLE_MODELS.get(model_name, model_name)
    logger.info("Loading embedding model '%s' (%s)…", model_name, model_id)

    try:
        # Pass HuggingFace token if available (for gated models)
        from src.config import HF_TOKEN
        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        model = SentenceTransformer(model_id, **kwargs)
        _model_cache[model_name] = model
        logger.info("Model '%s' loaded (dim=%d).", model_name, model.get_sentence_embedding_dimension())
        return model

    except OSError as e:
        raise RuntimeError(
            f"Could not load model '{model_id}'. "
            "Check your internet connection (required for first download) "
            f"or verify the model name is correct.\nDetail: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading embedding model '{model_id}': {e}"
        ) from e


def clear_model_cache() -> None:
    """Remove all cached models to free memory."""
    _model_cache.clear()
    logger.info("Embedding model cache cleared.")


# ══════════════════════════════════════════════════════════════════════════════
#  Embedding
# ══════════════════════════════════════════════════════════════════════════════

def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute L2-normalised embeddings for a list of texts.

    Args:
        texts:         List of strings to embed.
        model_name:    Key from AVAILABLE_MODELS.
        batch_size:    Batch size for encoding (reduce if OOM).
        show_progress: Show tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalised.

    Raises:
        ValueError:   If texts is empty.
        RuntimeError: On model loading or encoding failure.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list of texts.")

    # Filter out blank strings, warn about them
    clean_texts = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            logger.warning("Skipping empty/non-string text at index %d.", i)
            clean_texts.append(" ")  # placeholder to preserve index alignment
        else:
            clean_texts.append(t)

    model = get_model(model_name)

    try:
        embeddings = model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise for cosine sim via dot product
        )
        logger.debug(
            "embed_texts: %d texts → shape %s (model=%s).",
            len(clean_texts), embeddings.shape, model_name,
        )
        return embeddings

    except MemoryError:
        raise RuntimeError(
            "Out of memory while embedding texts. "
            "Try reducing batch_size, using a smaller model (MiniLM), "
            "or uploading fewer / shorter documents."
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed with model '{model_name}': {e}") from e


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query:      The search query.
        model_name: Key from AVAILABLE_MODELS.

    Returns:
        1-D numpy array of shape (embedding_dim,).

    Raises:
        ValueError:   If query is empty.
        RuntimeError: On encoding failure.
    """
    if not query or not query.strip():
        raise ValueError("embed_query received an empty query string.")
    return embed_texts([query], model_name=model_name)[0]


def get_embedding_dim(model_name: str) -> int:
    """
    Return the embedding dimension for a model.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    model = get_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError(f"Model '{model_name}' returned None for embedding dimension.")
    return int(dim)
