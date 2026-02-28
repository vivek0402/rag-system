# app/ingestion/chunker.py

import logging

logger = logging.getLogger(__name__)


def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split page-level text into overlapping chunks.

    Args:
        pages: Output from load_pdf() - list of page dicts
        chunk_size: Number of characters per chunk
        overlap: Number of characters shared between consecutive chunks

    Returns:
        List of chunk dicts:
        [{"chunk_id": 0, "text": "...", "source": "file.pdf", "page": 1}, ...]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Skip empty or whitespace-only chunks
            if chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": source,
                    "page": page_num
                })
                chunk_id += 1

            # Move forward by (chunk_size - overlap)
            start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages "
                f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks