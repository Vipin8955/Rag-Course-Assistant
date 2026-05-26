"""
ingestion.py — PDF text extraction using PyMuPDF
"""

import os
import re
import fitz  # PyMuPDF
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile to the data directory."""
    dest = DATA_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract and clean raw text from a PDF file."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
    doc.close()
    raw = "\n".join(pages_text)
    return clean_text(raw)


def clean_text(text: str) -> str:
    """Remove noise: excessive whitespace, special chars, page numbers."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone page numbers (lines with only digits)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove non-printable characters except newlines
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_all_documents() -> dict[str, str]:
    """Load and extract text from all PDFs in the data directory."""
    docs = {}
    for pdf_file in DATA_DIR.glob("*.pdf"):
        docs[pdf_file.name] = extract_text_from_pdf(pdf_file)
    return docs


def list_uploaded_files() -> list[str]:
    """Return names of all PDFs currently in data directory."""
    return [f.name for f in DATA_DIR.glob("*.pdf")]


def delete_file(filename: str) -> bool:
    """Delete a PDF from data directory."""
    target = DATA_DIR / filename
    if target.exists():
        target.unlink()
        return True
    return False
