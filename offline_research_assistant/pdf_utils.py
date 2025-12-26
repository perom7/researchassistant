from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader


class PDFExtractionError(RuntimeError):
    pass


def extract_pages_from_pdf(pdf_path: str | Path, ocr_fallback: bool = False, max_pages: Optional[int] = None) -> list[str]:
    """Extract per-page text; optionally fall back to OCR.

    Notes (Windows):
    - OCR fallback requires Poppler (for pdf2image) and Tesseract installed.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    page_texts: list[str] = []

    try:
        reader = PdfReader(str(pdf_path))
        pages = reader.pages[:max_pages] if max_pages else reader.pages
        for page in pages:
            page_text = page.extract_text() or ""
            page_texts.append(page_text)
    except Exception as e:
        raise PDFExtractionError(f"PyPDF2 failed to read PDF: {e}") from e

    text = "\n".join(page_texts).strip()
    if len(text) >= 200:
        # embedded text worked
        return page_texts

    if not ocr_fallback:
        # may be empty or tiny; still return what we have
        return page_texts

    # OCR fallback
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        raise PDFExtractionError(
            "OCR fallback requested but pdf2image is not available. Install it with pip install pdf2image."
        ) from e

    if shutil.which("tesseract") is None:
        raise PDFExtractionError(
            "OCR fallback requested but Tesseract is not installed or not on PATH. "
            "Install Tesseract OCR and ensure 'tesseract' is available in PATH."
        )

    try:
        import pytesseract
    except Exception as e:
        raise PDFExtractionError(
            "OCR fallback requested but pytesseract is not available. Install it with pip install pytesseract."
        ) from e

    try:
        images = convert_from_path(str(pdf_path))
    except Exception as e:
        raise PDFExtractionError(
            "pdf2image failed to convert pages. On Windows you typically need Poppler installed and added to PATH. "
            f"Underlying error: {e}"
        ) from e

    ocr_pages: list[str] = []
    for img in images[:max_pages] if max_pages else images:
        ocr_pages.append(pytesseract.image_to_string(img))

    return ocr_pages


def extract_text_from_pdf(pdf_path: str | Path, ocr_fallback: bool = False, max_pages: Optional[int] = None) -> str:
    """Backward-compatible API that returns a single string."""

    pages = extract_pages_from_pdf(pdf_path, ocr_fallback=ocr_fallback, max_pages=max_pages)
    return "\n".join(pages).strip()