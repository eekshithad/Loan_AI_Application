"""
PDF Ingestion Module
Handles loading, reading, and extracting text from PDF files.
Supports both text-based and scanned PDFs (via OCR).
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF 2.x
import pdfplumber
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int          # 1-based
    text: str
    has_text: bool
    ocr_applied: bool = False
    image_count: int = 0


@dataclass
class PDFDocument:
    file_path: str
    file_name: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


def _try_ocr(page_image: Image.Image) -> str:
    """Attempt OCR on a page image. Returns empty string if pytesseract unavailable."""
    try:
        import pytesseract
        return pytesseract.image_to_string(page_image)
    except Exception as exc:
        logger.debug(f"OCR skipped: {exc}")
        return ""


def load_pdf(file_path: str, ocr_fallback: bool = True) -> PDFDocument:
    """
    Load a PDF file and extract text from all pages.
    Falls back to OCR for scanned/image-only pages.
    """
    path = Path(file_path)
    if not path.exists():
        return PDFDocument(
            file_path=file_path,
            file_name=path.name,
            total_pages=0,
            error=f"File not found: {file_path}"
        )

    doc = PDFDocument(
        file_path=str(path),
        file_name=path.name,
        total_pages=0,
    )

    try:
        pdf = fitz.open(file_path)
        doc.total_pages = len(pdf)
        doc.metadata = {k: v for k, v in pdf.metadata.items() if v}

        all_text_parts = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text").strip()
            has_text = bool(text and len(text) > 20)
            ocr_applied = False

            if not has_text and ocr_fallback:
                # Render page to image and attempt OCR
                try:
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    clip = page.get_pixmap(matrix=mat)
                    img_data = clip.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    text = _try_ocr(img)
                    if text.strip():
                        ocr_applied = True
                        has_text = True
                        logger.debug(f"OCR applied to page {page_num+1} of {path.name}")
                except Exception as e:
                    logger.debug(f"OCR image render failed: {e}")

            # Count images on page
            image_list = page.get_images(full=True)

            pc = PageContent(
                page_number=page_num + 1,
                text=text,
                has_text=has_text,
                ocr_applied=ocr_applied,
                image_count=len(image_list),
            )
            doc.pages.append(pc)
            if text:
                all_text_parts.append(f"[Page {page_num+1}]\n{text}")

        doc.full_text = "\n\n".join(all_text_parts)
        pdf.close()
        logger.info(f"Loaded '{path.name}': {doc.total_pages} pages, "
                    f"{sum(p.has_text for p in doc.pages)} with text")

    except Exception as exc:
        doc.error = str(exc)
        logger.error(f"Failed to load PDF '{file_path}': {exc}")

    return doc


def load_pdfs_from_folder(folder_path: str, ocr_fallback: bool = True) -> list[PDFDocument]:
    """Load all PDFs from a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"Folder does not exist: {folder_path}")
        return []

    pdfs = []
    for pdf_file in sorted(folder.glob("*.pdf")):
        doc = load_pdf(str(pdf_file), ocr_fallback=ocr_fallback)
        pdfs.append(doc)
    return pdfs


def get_page_text(doc: PDFDocument, page_number: int) -> str:
    """Get text from a specific page (1-based)."""
    for page in doc.pages:
        if page.page_number == page_number:
            return page.text
    return ""


def get_text_window(doc: PDFDocument, start_page: int, end_page: int) -> str:
    """Get combined text from a range of pages."""
    parts = []
    for page in doc.pages:
        if start_page <= page.page_number <= end_page:
            parts.append(f"[Page {page.page_number}]\n{page.text}")
    return "\n\n".join(parts)

