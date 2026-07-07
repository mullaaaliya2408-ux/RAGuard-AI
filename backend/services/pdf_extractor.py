"""
Extracts text from digital PDF pages using PyMuPDF.
Also handles page rotation so extracted text reads in the correct order.
"""

import logging
import fitz

from models.schemas import PageResult

logger = logging.getLogger("raguard.pdf_extractor")


def extract_digital_page(page: "fitz.Page", page_number: int) -> PageResult:
    """Extract text from a single digital PDF page, correcting rotation first."""
    if page.rotation != 0:
        logger.info(f"Page {page_number} rotated {page.rotation}°, normalizing")
        page.set_rotation(0)

    text = page.get_text()
    return PageResult(page_number=page_number, text=text, used_ocr=False)


def extract_digital_pdf(file_path: str) -> list[PageResult]:
    """Extract all pages from a fully digital PDF."""
    doc = fitz.open(file_path)
    results = [extract_digital_page(page, i + 1) for i, page in enumerate(doc)]
    doc.close()
    return results


def render_page_as_image(file_path: str, page_number: int, zoom: float = 2.0):
    """
    Render a PDF page to a PIL-compatible image (used when a page needs OCR,
    e.g. in scanned or mixed PDFs). zoom=2.0 roughly doubles resolution,
    which noticeably improves OCR accuracy on small text.
    """
    doc = fitz.open(file_path)
    page = doc[page_number - 1]
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    doc.close()
    return pixmap.tobytes("png")