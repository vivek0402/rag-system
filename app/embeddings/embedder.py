# app/embeddings/embedder.py

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# We load the model once at module level - not inside the function
# Loading a model is expensive (seconds). If it were inside the function,
# it would reload every time you embed a chunk. That's wasteful.
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
logger.info(f"Loaded embedding model: {MODEL_NAME}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim)
        For all-MiniLM-L6-v2, embedding_dim = 384
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")

    logger.info(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Kept separate from embed_texts for clarity - queries are always single strings.

    Args:
        query: The user's question

    Returns:
        numpy array of shape (1, embedding_dim)
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    return model.encode([query])