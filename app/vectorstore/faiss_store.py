# app/vectorstore/faiss_store.py

import logging
import json
from pathlib import Path
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []
        logger.info(f"Initialized FAISS IndexFlatL2 | dim={embedding_dim}")

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have same length. "
                f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )
        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks | Total in store: {len(self.chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")
        if top_k > self.index.ntotal:
            top_k = self.index.ntotal
            logger.warning(f"top_k reduced to {top_k}")
        query_vector = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(1 - dist / 2)
            results.append(chunk)
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results

    def total_chunks(self) -> int:
        return self.index.ntotal

    def save(self, directory: str | Path) -> None:
        """
        Save FAISS index and chunk metadata to disk.

        Args:
            directory: Folder to save files into.
                      Creates two files:
                      - faiss.index  (binary FAISS index)
                      - chunks.json  (chunk metadata)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "faiss.index"
        chunks_path = directory / "chunks.json"

        faiss.write_index(self.index, str(index_path))

        with open(chunks_path, "w") as f:
            json.dump(self.chunks, f)

        logger.info(f"Saved index ({self.index.ntotal} vectors) to {directory}")

    def load(self, directory: str | Path) -> bool:
        """
        Load FAISS index and chunk metadata from disk.

        Args:
            directory: Folder containing faiss.index and chunks.json

        Returns:
            True if loaded successfully, False if files don't exist
        """
        directory = Path(directory)
        index_path = directory / "faiss.index"
        chunks_path = directory / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            logger.info("No saved index found — starting fresh")
            return False

        self.index = faiss.read_index(str(index_path))

        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)

        logger.info(f"Loaded index ({self.index.ntotal} vectors) from {directory}")
        return True