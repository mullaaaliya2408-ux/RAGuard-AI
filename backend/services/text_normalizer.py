"""
Cleans up extracted text: fixes spacing, removes duplicate blank lines,
and strips likely headers/footers (short repeated lines at page edges).
"""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single, readable spacing."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merge_broken_paragraphs(text: str) -> str:
    """
    OCR and PDF extraction often break sentences across lines mid-paragraph.
    This merges lines that don't end in sentence-ending punctuation.
    """
    lines = text.split("\n")
    merged: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue

        if merged and merged[-1] and not merged[-1].rstrip()[-1:] in ".!?:":
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

    return "\n".join(merged)


def remove_likely_headers_footers(pages_text: list[str]) -> list[str]:
    """
    If the exact same short line (<60 chars) appears on more than half the
    pages, it's almost certainly a header/footer (e.g. page numbers, doc title)
    and gets stripped from every page.
    """
    if len(pages_text) < 3:
        return pages_text  # not enough pages to detect a repeating pattern reliably

    line_counts: dict[str, int] = {}
    for text in pages_text:
        for line in set(text.split("\n")):
            clean = line.strip()
            if clean and len(clean) < 60:
                line_counts[clean] = line_counts.get(clean, 0) + 1

    threshold = len(pages_text) / 2
    repeated_lines = {line for line, count in line_counts.items() if count > threshold}

    cleaned_pages = []
    for text in pages_text:
        lines = [ln for ln in text.split("\n") if ln.strip() not in repeated_lines]
        cleaned_pages.append("\n".join(lines))

    return cleaned_pages


def normalize_text(text: str) -> str:
    """Full normalization pipeline applied to a single page's text."""
    text = normalize_whitespace(text)
    text = merge_broken_paragraphs(text)
    return text