# tests/test_faiss_store.py

import pytest
import numpy as np
from app.vectorstore.faiss_store import FAISSVectorStore


def make_store_with_data(n=5, dim=384):
    """Helper: create a store with n random vectors."""
    store = FAISSVectorStore(embedding_dim=dim)
    chunks = [
        {"chunk_id": i, "text": f"chunk {i}", "source": "test.pdf", "page": 1}
        for i in range(n)
    ]
    embeddings = np.random.rand(n, dim).astype("float32")
    store.add_chunks(chunks, embeddings)
    return store


def test_total_chunks():
    """Store reports correct chunk count."""
    store = make_store_with_data(n=5)
    assert store.total_chunks() == 5


def test_search_returns_top_k():
    """Search returns exactly top_k results."""
    store = make_store_with_data(n=10)
    query = np.random.rand(1, 384).astype("float32")
    results = store.search(query, top_k=3)
    assert len(results) == 3


def test_search_results_have_score():
    """Each result includes a score field."""
    store = make_store_with_data(n=5)
    query = np.random.rand(1, 384).astype("float32")
    results = store.search(query, top_k=2)
    for r in results:
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.0


def test_search_empty_store_raises():
    """Searching empty store raises ValueError."""
    store = FAISSVectorStore(embedding_dim=384)
    query = np.random.rand(1, 384).astype("float32")
    with pytest.raises(ValueError):
        store.search(query, top_k=3)


def test_mismatched_chunks_embeddings_raises():
    """Mismatched chunks and embeddings raises ValueError."""
    store = FAISSVectorStore(embedding_dim=384)
    chunks = [{"chunk_id": 0, "text": "test", "source": "test.pdf", "page": 1}]
    embeddings = np.random.rand(3, 384).astype("float32")
    with pytest.raises(ValueError):
        store.add_chunks(chunks, embeddings)