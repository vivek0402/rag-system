# app/api/routes.py

import os
import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.retrieval.retriever import Retriever
from app.llm.generator import generate_answer
from config import DATA_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

# Single retriever instance — built once, reused for all queries
# This is the "stateful" part of our API
retriever = Retriever()
is_index_built = False


# ── Request/Response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str


# ── Endpoints ────────────────────────────────────────────────────────
@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and build the vector index.
    Call this before querying.
    """
    global retriever, is_index_built

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to data/raw/
    save_path = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"PDF saved: {save_path}")

    # Build fresh index with this PDF
    retriever = Retriever()
    retriever.build_index([str(save_path)])
    is_index_built = True

    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_indexed": retriever.store.total_chunks()
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question against the ingested document.
    """
    if not is_index_built:
        raise HTTPException(
            status_code=400,
            detail="No document ingested yet. Call /ingest first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Query received: {request.question}")

    results = retriever.search(request.question, top_k=request.top_k)
    response = generate_answer(request.question, results)

    return QueryResponse(
        answer=response["answer"],
        sources=response["sources"],
        model=response["model"]
    )


@router.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "index_built": is_index_built}