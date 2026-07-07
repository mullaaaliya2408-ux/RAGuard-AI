"""
Standalone script to sanity-check chunking quality on a local file
without going through the API. Useful for quickly iterating on
TARGET_CHUNK_CHARS or overlap logic.

Usage:
    python scripts/test_chunking.py path/to/file.pdf
"""

import sys
import uuid

from services.document_processor import process_document
from services.chunker import chunk_document


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_chunking.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    filename = file_path.split("/")[-1]

    metadata, pages = process_document(file_path, filename)
    chunks = chunk_document(pages, str(uuid.uuid4()), filename)

    print(f"\nDocument type: {metadata.document_type}")
    print(f"Total pages: {metadata.total_pages}")
    print(f"Total chunks: {len(chunks)}\n")

    for c in chunks:
        print(f"--- Chunk {c.chunk_index} | Page {c.page_number} | Section: {c.section} | Len: {c.chunk_length} ---")
        print(c.text[:200] + ("..." if len(c.text) > 200 else ""))
        print()


if __name__ == "__main__":
    main()