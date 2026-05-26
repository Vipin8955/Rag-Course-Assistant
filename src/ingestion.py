"""
ingestion.py — PDF upload, text extraction, and cleaning.
Uses PyMuPDF (fitz) for robust PDF parsing with full error handling.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  File Management
# ══════════════════════════════════════════════════════════════════════════════

def save_uploaded_file(uploaded_file) -> Optional[Path]:
    """
    Save a Streamlit UploadedFile to the data directory.

    Returns:
        Path to the saved file, or None if saving failed.
    """
    if uploaded_file is None:
        logger.warning("save_uploaded_file called with None.")
        return None

    dest = DATA_DIR / uploaded_file.name
    try:
        data = uploaded_file.getbuffer()
        if len(data) == 0:
            logger.warning("Uploaded file '%s' is empty — skipping.", uploaded_file.name)
            return None

        with open(dest, "wb") as f:
            f.write(data)

        logger.info("Saved '%s' (%d bytes) to %s", uploaded_file.name, len(data), dest)
        return dest

    except OSError as e:
        logger.error("Failed to save '%s': %s", uploaded_file.name, e)
        return None


def list_uploaded_files() -> list[str]:
    """Return names of all PDFs currently in the data directory."""
    try:
        return sorted(f.name for f in DATA_DIR.glob("*.pdf"))
    except OSError as e:
        logger.error("Cannot list data directory '%s': %s", DATA_DIR, e)
        return []


def delete_file(filename: str) -> bool:
    """
    Delete a PDF from the data directory.

    Returns:
        True if deleted, False if not found or deletion failed.
    """
    if not filename or ".." in filename:
        logger.warning("Unsafe or empty filename rejected: %r", filename)
        return False

    target = DATA_DIR / filename
    try:
        if not target.exists():
            logger.warning("File not found for deletion: %s", target)
            return False
        target.unlink()
        logger.info("Deleted '%s'", target)
        return True
    except OSError as e:
        logger.error("Failed to delete '%s': %s", target, e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Text Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract and clean raw text from a PDF file.

    Returns:
        Cleaned text string, or empty string on failure.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return ""

    if pdf_path.stat().st_size == 0:
        logger.warning("PDF is empty (0 bytes): %s", pdf_path)
        return ""

    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError as e:
        logger.error("Corrupt or unreadable PDF '%s': %s", pdf_path.name, e)
        return ""
    except Exception as e:
        logger.error("Unexpected error opening PDF '%s': %s", pdf_path.name, e)
        return ""

    pages_text: list[str] = []
    try:
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                if text:
                    pages_text.append(text)
            except Exception as e:
                logger.warning(
                    "Could not extract text from page %d of '%s': %s",
                    page_num + 1, pdf_path.name, e,
                )
    finally:
        doc.close()

    if not pages_text:
        logger.warning("No text extracted from '%s' — may be a scanned/image PDF.", pdf_path.name)
        return ""

    raw = "\n".join(pages_text)
    cleaned = clean_text(raw)
    logger.info(
        "Extracted %d chars from '%s' (%d pages).",
        len(cleaned), pdf_path.name, len(pages_text),
    )
    return cleaned


def clean_text(text: str) -> str:
    """
    Remove noise from extracted PDF text:
    excessive whitespace, lone page numbers, non-printable characters.
    """
    if not isinstance(text, str):
        logger.warning("clean_text received non-string input (%s) — returning empty.", type(text))
        return ""

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that contain only digits (page numbers)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove non-printable characters (keep newlines and standard ASCII)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_all_documents() -> dict[str, str]:
    """
    Load and extract text from all PDFs in the data directory.

    Returns:
        Dict mapping filename → extracted text.
        Files that fail to parse are skipped with a warning.
    """
    docs: dict[str, str] = {}
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDFs found in %s.", DATA_DIR)
        return docs

    for pdf_file in sorted(pdf_files):
        text = extract_text_from_pdf(pdf_file)
        if text:
            docs[pdf_file.name] = text
        else:
            logger.warning("Skipping '%s' — no usable text extracted.", pdf_file.name)

    logger.info("Loaded %d/%d PDFs successfully.", len(docs), len(pdf_files))
    return docs
