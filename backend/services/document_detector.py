"""
Detects whether a PDF is digital (has extractable text), scanned (image-only),
or mixed (some pages digital, some scanned). Images are always treated as
requiring OCR.
"""

import logging
import fitz  # PyMuPDF

from models.schemas import DocumentType

logger = logging.getLogger("raguard.detector")

# If a page has fewer than this many extractable characters, we treat it as
# "no usable text" and assume it needs OCR.
MIN_TEXT_CHARS_PER_PAGE = 20


def detect_pdf_type(file_path: str) -> DocumentType:
    """
    Open a PDF and inspect each page's extractable text to decide
    if the document is digital, scanned, or mixed.
    """
    doc = fitz.open(file_path)
    digital_pages = 0
    scanned_pages = 0

    for page in doc:
        text = page.get_text().strip()
        if len(text) >= MIN_TEXT_CHARS_PER_PAGE:
            digital_pages += 1
        else:
            scanned_pages += 1

    doc.close()
    logger.info(f"{file_path}: {digital_pages} digital pages, {scanned_pages} scanned pages")

    if scanned_pages == 0:
        return DocumentType.DIGITAL_PDF
    if digital_pages == 0:
        return DocumentType.SCANNED_PDF
    return DocumentType.MIXED_PDF


def detect_document_type(file_path: str, file_extension: str) -> DocumentType:
    """
    Entry point used by the processor. Routes to PDF-specific detection,
    or treats known image formats as always needing OCR.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    if file_extension.lower() == ".pdf":
        return detect_pdf_type(file_path)

    if file_extension.lower() in image_extensions:
        return DocumentType.IMAGE

    raise ValueError(f"Unsupported file extension: {file_extension}")