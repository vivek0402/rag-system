# tests/test_retriever.py

import pytest
from app.retrieval.retriever import Retriever


def test_search_before_build_raises():
    """Searching before building index should raise RuntimeError."""
    retriever = Retriever()
    with pytest.raises(RuntimeError):
        retriever.search("test query")


def test_build_and_search(tmp_path):
    """Build index from a real PDF and search returns results."""
    import shutil
    from pathlib import Path

    # Copy test PDF to temp directory
    src = Path("data/raw/Vivek_s_Cover_Letter_Jan (1).pdf")
    if not src.exists():
        pytest.skip("Test PDF not available")

    dst = tmp_path / "test.pdf"
    shutil.copy(src, dst)

    retriever = Retriever(chunk_size=500, overlap=50)
    retriever.build_index([str(dst)])

    results = retriever.search("CGPA", top_k=2)
    assert len(results) == 2
    assert "text" in results[0]
    assert "score" in results[0]
    assert "source" in results[0]


def test_top_k_respected(tmp_path):
    """search() returns exactly top_k results."""
    import shutil
    from pathlib import Path

    src = Path("data/raw/Vivek_s_Cover_Letter_Jan (1).pdf")
    if not src.exists():
        pytest.skip("Test PDF not available")

    dst = tmp_path / "test.pdf"
    shutil.copy(src, dst)

    retriever = Retriever(chunk_size=500, overlap=50)
    retriever.build_index([str(dst)])

    for k in [1, 2, 3]:
        results = retriever.search("experience", top_k=k)
        assert len(results) == k