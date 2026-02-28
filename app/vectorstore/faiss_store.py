# app/vectorstore/faiss_store.py

import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Wraps a FAISS index with chunk metadata storage.
    
    FAISS stores vectors and returns integer indices.
    We maintain a separate list of chunk dicts to map
    those indices back to actual text and metadata.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Args:
            embedding_dim: Size of each embedding vector.
                          Must match your embedding model's output.
                          all-MiniLM-L6-v2 produces 384-dim vectors.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []  # stores chunk dicts in insertion order
        logger.info(f"Initialized FAISS IndexFlatL2 | dim={embedding_dim}")

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: List of chunk dicts from chunker.py
            embeddings: numpy array of shape (len(chunks), embedding_dim)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have same length. "
                f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        # FAISS requires float32 - normalize for cosine-equivalent search
        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks | Total in store: {len(self.chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most similar chunks to the query.

        Args:
            query_embedding: numpy array of shape (1, embedding_dim)
            top_k: Number of results to return

        Returns:
            List of chunk dicts, ordered by similarity (most similar first)
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        if top_k > self.index.ntotal:
            top_k = self.index.ntotal
            logger.warning(f"top_k reduced to {top_k} (total chunks in store)")

        # Normalize query vector
        query_vector = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_vector)

        # D = distances, I = indices of nearest neighbors
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(1 - dist / 2)  # convert L2 to similarity score
            results.append(chunk)

        logger.info(f"Retrieved {len(results)} chunks for query")
        return results

    def total_chunks(self) -> int:
        return self.index.ntotal