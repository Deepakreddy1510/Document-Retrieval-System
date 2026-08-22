from __future__ import annotations

import re

import pymupdf

from .models import PageText


def clean_text(text: str) -> str:
    """Apply conservative cleanup while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words split by PDF line wrapping, e.g. "regu-\nlarization".
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    # Preserve blank lines as paragraph boundaries, join ordinary line wraps.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_bytes(pdf_bytes: bytes) -> list[PageText]:
    """Extract text page-by-page from a PDF byte string using PyMuPDF."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    pages: list[PageText] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for index, page in enumerate(document):
            # sort=True generally gives a more natural top-left to bottom-right order.
            text = clean_text(page.get_text("text", sort=True))
            pages.append(PageText(page_number=index + 1, text=text))

    if not any(page.text for page in pages):
        raise ValueError(
            "No extractable text was found. This v1 supports text-based PDFs; "
            "OCR for scanned PDFs is intentionally not enabled yet."
        )
    return pages
