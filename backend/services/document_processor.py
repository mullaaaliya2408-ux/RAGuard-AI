"""
Top-level orchestrator: given a file, detects its type, routes to the right
extractor(s), normalizes the text, and returns structured metadata + pages.
This is the single entry point the API layer calls.
"""

import logging
import os
import uuid

from models.schemas import DocumentMetadata, DocumentType, PageResult
from services.document_detector import detect_document_type
from services.pdf_extractor import extract_digital_pdf, render_page_as_image
from services.text_normalizer import normalize_text, remove_likely_headers_footers
from ocr.ocr_engine import run_ocr_on_image_bytes

import fitz

logger = logging.getLogger("raguard.processor")


def process_image_file(file_path: str) -> list[PageResult]:
    """A standalone image file is treated as a single OCR page."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    return [run_ocr_on_image_bytes(image_bytes, page_number=1)]


def process_scanned_or_mixed_pdf(file_path: str, doc_type: DocumentType) -> list[PageResult]:
    """
    For scanned PDFs, OCR every page. For mixed PDFs, use digital extraction
    where possible and fall back to OCR only for image-only pages.
    """
    doc = fitz.open(file_path)
    results: list[PageResult] = []

    for i, page in enumerate(doc):
        page_number = i + 1
        digital_text = page.get_text().strip()

        if doc_type == DocumentType.MIXED_PDF and len(digital_text) >= 20:
            results.append(PageResult(page_number=page_number, text=page.get_text(), used_ocr=False))
        else:
            image_bytes = render_page_as_image(file_path, page_number)
            results.append(run_ocr_on_image_bytes(image_bytes, page_number))

    doc.close()
    return results


def process_document(file_path: str, filename: str) -> tuple[DocumentMetadata, list[PageResult]]:
    """
    Main entry point: detect → extract → normalize → return metadata + pages.
    """
    _, extension = os.path.splitext(filename)
    warnings: list[str] = []

    doc_type = detect_document_type(file_path, extension)
    logger.info(f"Detected {filename} as {doc_type.value}")

    if doc_type == DocumentType.DIGITAL_PDF:
        pages = extract_digital_pdf(file_path)
    elif doc_type == DocumentType.IMAGE:
        pages = process_image_file(file_path)
    else:  # SCANNED_PDF or MIXED_PDF
        pages = process_scanned_or_mixed_pdf(file_path, doc_type)

    # Normalize each page's text
    for page in pages:
        page.text = normalize_text(page.text)

    # Remove repeated headers/footers across pages
    cleaned_texts = remove_likely_headers_footers([p.text for p in pages])
    for page, cleaned in zip(pages, cleaned_texts):
        page.text = cleaned

    ocr_confidences = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
    if ocr_confidences and (sum(ocr_confidences) / len(ocr_confidences)) < 0.5:
        warnings.append("Low average OCR confidence — extracted text may contain errors.")

    total_chars = sum(len(p.text) for p in pages)
    if total_chars == 0:
        warnings.append("No text could be extracted from this document.")

    metadata = DocumentMetadata(
        document_id=str(uuid.uuid4()),
        filename=filename,
        document_type=doc_type,
        total_pages=len(pages),
        total_characters=total_chars,
        average_ocr_confidence=(
            round(sum(ocr_confidences) / len(ocr_confidences), 3) if ocr_confidences else None
        ),
        processing_warnings=warnings,
    )

    return metadata, pages