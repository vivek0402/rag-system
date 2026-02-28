# tests/test_chunker.py

import pytest
from app.ingestion.chunker import chunk_text


# Sample pages mimicking load_pdf() output
SAMPLE_PAGES = [
    {"page": 1, "text": "A" * 1000, "source": "test.pdf"}
]


def test_chunk_count():
    """Correct number of chunks produced."""
    chunks = chunk_text(SAMPLE_PAGES, chunk_size=500, overlap=50)
    # 1000 chars, step=450 → chunks at 0, 450 → 2 chunks
    assert len(chunks) == 3


def test_chunk_size():
    """No chunk exceeds chunk_size."""
    chunks = chunk_text(SAMPLE_PAGES, chunk_size=500, overlap=50)
    for c in chunks:
        assert len(c["text"]) <= 500


def test_overlap_exists():
    """Consecutive chunks share overlapping text."""
    pages = [{"page": 1, "text": "ABCDEFGHIJ", "source": "test.pdf"}]
    chunks = chunk_text(pages, chunk_size=5, overlap=2)
    # chunk 0: ABCDE, chunk 1: DEFGH — 'DE' appears in both
    assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]


def test_metadata_preserved():
    """Each chunk carries source and page metadata."""
    chunks = chunk_text(SAMPLE_PAGES, chunk_size=500, overlap=50)
    for c in chunks:
        assert c["source"] == "test.pdf"
        assert c["page"] == 1
        assert "chunk_id" in c


def test_invalid_overlap_raises():
    """overlap >= chunk_size should raise ValueError."""
    with pytest.raises(ValueError):
        chunk_text(SAMPLE_PAGES, chunk_size=100, overlap=100)


def test_empty_pages_returns_empty():
    """Empty input returns empty list."""
    chunks = chunk_text([], chunk_size=500, overlap=50)
    assert chunks == []