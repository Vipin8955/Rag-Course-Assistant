"""
config.py — Centralised configuration loader for University Course Assistant.

Loads settings from the .env file (via python-dotenv) with safe fallback
defaults so the app works even if no .env file is present.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/ -> root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=False)  # Don't override existing env vars

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("config")


# ── Helper ────────────────────────────────────────────────────────────────────
def _get_int(key: str, default: int) -> int:
    """Read an integer env var safely, falling back to default on bad values."""
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r — using default %d", key, raw, default)
        return default


# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR: Path = _PROJECT_ROOT / os.getenv("DATA_DIR", "data")
CHROMA_DIR: Path = _PROJECT_ROOT / os.getenv("CHROMA_DIR", "data/chroma_db")

# Ensure directories exist at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Default App Settings ──────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = _get_int("DEFAULT_CHUNK_SIZE", 300)
DEFAULT_OVERLAP: int = _get_int("DEFAULT_OVERLAP", 75)
DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_VECTOR_STORE: str = os.getenv("DEFAULT_VECTOR_STORE", "FAISS")
DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "FLAN-T5-Base (Fast)")
DEFAULT_TOP_K: int = _get_int("DEFAULT_TOP_K", 5)

# ── HuggingFace (optional) ────────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

# ── Windows / Runtime Fixes ───────────────────────────────────────────────────
# Set before torch/faiss are imported to avoid duplicate OpenMP library warning.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", os.getenv("KMP_DUPLICATE_LIB_OK", "TRUE"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))

if logger.isEnabledFor(logging.DEBUG):
    logger.debug(
        "Config loaded | DATA_DIR=%s | CHROMA_DIR=%s | "
        "chunk=%d | overlap=%d | embed=%s | store=%s | llm=%s | top_k=%d",
        DATA_DIR, CHROMA_DIR,
        DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
        DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_STORE,
        DEFAULT_LLM, DEFAULT_TOP_K,
    )
