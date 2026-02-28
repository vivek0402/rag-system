# app/retrieval/retriever.py

import logging
import numpy as np

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_texts, embed_query
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Orchestrates the full retrieval pipeline:
    PDF → chunks → embeddings → FAISS index → search
    
    The index is built once via build_index() and then
    search() can be called any number of times efficiently.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, embedding_dim: int = 384):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = FAISSVectorStore(embedding_dim=embedding_dim)
        self._is_built = False
        logger.info("Retriever initialized")

    def build_index(self, pdf_paths: list[str]) -> None:
        """
        Load PDFs, chunk them, embed them, and store in FAISS.
        Call this once before calling search().

        Args:
            pdf_paths: List of paths to PDF files
        """
        all_chunks = []

        for path in pdf_paths:
            logger.info(f"Processing: {path}")
            pages = load_pdf(path)
            chunks = chunk_text(pages, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks were extracted from the provided PDFs.")

        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)
        self.store.add_chunks(all_chunks, embeddings)
        self._is_built = True
        logger.info(f"Index built with {len(all_chunks)} total chunks")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search the index for chunks relevant to the query.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            List of chunk dicts with similarity scores
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = embed_query(query)
        results = self.store.search(query_embedding, top_k=top_k)
        return results