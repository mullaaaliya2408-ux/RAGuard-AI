"""
Smart chunking service.

Design principles (why we don't use fixed-size chunking):
- Fixed-size chunking cuts sentences and paragraphs in half, destroying
  meaning at chunk boundaries. This hurts retrieval quality badly.
- Instead, we build chunks out of whole sentences, grouped into paragraphs,
  and only split when a target size is reached — always at a sentence
  boundary.
- Overlap is adaptive: dense/technical text (long sentences) gets a smaller
  overlap window (in sentence count) since each sentence already carries
  more context; short/choppy text gets a larger overlap so context isn't lost.
"""

import logging
import re
import uuid

import nltk

from models.schemas import Chunk, PageResult

logger = logging.getLogger("raguard.chunker")

# Target chunk size in characters. Chosen to comfortably fit inside
# typical embedding model context windows while staying focused enough
# for precise retrieval.
TARGET_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 200


def split_into_sentences(text: str) -> list[str]:
    """Split a block of text into sentences using NLTK's Punkt tokenizer."""
    if not text.strip():
        return []
    return nltk.sent_tokenize(text)


def split_into_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs on blank lines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def detect_section_heading(paragraph: str) -> str | None:
    """
    Heuristic: short, title-cased or all-caps lines with no ending
    punctuation are likely section headings (e.g. "ADMISSION POLICY").
    """
    stripped = paragraph.strip()
    if len(stripped) < 60 and not stripped.endswith((".", ",", ";")):
        if stripped.isupper() or stripped.istitle():
            return stripped
    return None


def compute_adaptive_overlap_sentences(sentences: list[str]) -> int:
    """
    Decide how many trailing sentences to carry over into the next chunk.
    Longer average sentence length -> less overlap needed (1 sentence).
    Shorter/choppier sentences -> more overlap needed (up to 3 sentences).
    """
    if not sentences:
        return 1

    average_length = sum(len(s) for s in sentences) / len(sentences)

    if average_length > 120:
        return 1
    if average_length > 60:
        return 2
    return 3


def chunk_paragraph_group(
    sentences: list[str],
    document_id: str,
    document_name: str,
    page_number: int,
    section: str | None,
    ocr_confidence: float | None,
    start_index: int,
) -> list[Chunk]:
    """
    Greedily pack sentences into chunks close to TARGET_CHUNK_CHARS,
    never splitting mid-sentence, with adaptive overlap between chunks.
    """
    chunks: list[Chunk] = []
    overlap_size = compute_adaptive_overlap_sentences(sentences)

    current: list[str] = []
    current_len = 0
    chunk_index = start_index

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        current.append(sentence)
        current_len += len(sentence)

        is_last_sentence = i == len(sentences) - 1
        if current_len >= TARGET_CHUNK_CHARS or is_last_sentence:
            text = " ".join(current)
            if len(text) >= MIN_CHUNK_CHARS or is_last_sentence:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        section=section,
                        text=text,
                        chunk_length=len(text),
                        ocr_confidence=ocr_confidence,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                # Adaptive overlap: carry the last `overlap_size` sentences
                # into the next chunk so context isn't lost at the boundary.
                current = current[-overlap_size:] if overlap_size < len(current) else current[:]
                current_len = sum(len(s) for s in current)
            # else: chunk too small (only happens on non-last iterations
            # in edge cases) — keep accumulating

        i += 1

    return chunks


def chunk_page(
    page: PageResult,
    document_id: str,
    document_name: str,
    start_index: int,
) -> list[Chunk]:
    """Chunk a single page's text, respecting paragraph and section boundaries."""
    paragraphs = split_into_paragraphs(page.text)
    if not paragraphs:
        return []

    all_chunks: list[Chunk] = []
    current_section: str | None = None
    index = start_index

    for paragraph in paragraphs:
        heading = detect_section_heading(paragraph)
        if heading:
            current_section = heading
            continue  # headings aren't chunked as content themselves

        sentences = split_into_sentences(paragraph)
        if not sentences:
            continue

        new_chunks = chunk_paragraph_group(
            sentences=sentences,
            document_id=document_id,
            document_name=document_name,
            page_number=page.page_number,
            section=current_section,
            ocr_confidence=page.ocr_confidence,
            start_index=index,
        )
        all_chunks.extend(new_chunks)
        index += len(new_chunks)

    return all_chunks


def chunk_document(
    pages: list[PageResult],
    document_id: str,
    document_name: str,
) -> list[Chunk]:
    """
    Main entry point: turn all pages of a processed document into
    an ordered list of chunks with full metadata.
    """
    all_chunks: list[Chunk] = []
    index = 0

    for page in pages:
        page_chunks = chunk_page(page, document_id, document_name, start_index=index)
        all_chunks.extend(page_chunks)
        index += len(page_chunks)

    logger.info(f"Chunked '{document_name}' into {len(all_chunks)} chunks across {len(pages)} pages")
    return all_chunks