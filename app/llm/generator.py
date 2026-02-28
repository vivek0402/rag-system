# app/llm/generator.py

import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env file into environment variables

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Generate a grounded answer using retrieved context chunks.

    Args:
        query: The user's question
        context_chunks: Retrieved chunks from the vector store

    Returns:
        Dict with answer, sources, and model used
    """
    if not context_chunks:
        return {
            "answer": "I could not find relevant information to answer your question.",
            "sources": [],
            "model": MODEL
        }

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join([
        f"Source: {c['source']} | Page: {c['page']}\n{c['text']}"
        for c in context_chunks
    ])

    # This prompt is critical - it grounds the LLM to only use provided context
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Do NOT use your own knowledge or make up information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    logger.info(f"Sending query to {MODEL}: {query[:50]}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature = more focused, less creative
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()

    sources = [
        {"source": c["source"], "page": c["page"], "score": c.get("score", 0)}
        for c in context_chunks
    ]

    logger.info(f"Answer generated ({len(answer)} chars)")

    return {
        "answer": answer,
        "sources": sources,
        "model": MODEL
    }