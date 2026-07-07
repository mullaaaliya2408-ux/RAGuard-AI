"""
Wraps EasyOCR behind a simple interface. EasyOCR is initialized once and
reused across requests, since loading the model is expensive.
"""

import logging
import io
import numpy as np
from PIL import Image
import easyocr

from models.schemas import PageResult

logger = logging.getLogger("raguard.ocr")

# Loaded lazily so importing this module doesn't trigger model download at startup
_reader: easyocr.Reader | None = None


def get_ocr_reader() -> easyocr.Reader:
    """Return a cached EasyOCR reader, creating it on first use."""
    global _reader
    if _reader is None:
        logger.info("Loading EasyOCR model (first use only)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def run_ocr_on_image_bytes(image_bytes: bytes, page_number: int) -> PageResult:
    """
    Run OCR on raw image bytes (PNG/JPEG) and return extracted text plus
    an average confidence score across all detected text regions.
    """
    reader = get_ocr_reader()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    results = reader.readtext(image_array)  # list of (bbox, text, confidence)

    if not results:
        logger.warning(f"No text detected on page {page_number}")
        return PageResult(page_number=page_number, text="", ocr_confidence=0.0, used_ocr=True)

    texts = [r[1] for r in results]
    confidences = [r[2] for r in results]
    average_confidence = sum(confidences) / len(confidences)

    return PageResult(
        page_number=page_number,
        text=" ".join(texts),
        ocr_confidence=round(average_confidence, 3),
        used_ocr=True,
    )