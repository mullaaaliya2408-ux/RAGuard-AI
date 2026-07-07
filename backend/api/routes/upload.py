"""
Upload endpoint: accepts a file, saves it, runs it through the document
processor, and returns metadata + page-level results to the frontend.
"""

import logging
import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.document_processor import process_document

logger = logging.getLogger("raguard.upload")
router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "documents"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a document, process it, and return extracted metadata."""
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_filename = f"{uuid.uuid4()}{extension}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        os.remove(saved_path)
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        metadata, pages = process_document(saved_path, file.filename)
        chunks = chunk_document(pages, metadata.document_id, file.filename)
        was_indexed = index_chunks(chunks)
    except Exception as e:
        logger.exception(f"Failed to process {file.filename}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {
        "metadata": metadata.model_dump(),
        "pages": [p.model_dump() for p in pages],
         "chunks": [c.model_dump() for c in chunks],
         "indexed": was_indexed,
    }
from services.document_processor import process_document
from services.chunker import chunk_document
from services.chunker import chunk_document
from services.indexing_service import index_chunks