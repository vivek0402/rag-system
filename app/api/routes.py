# app/api/routes.py

import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from app.retrieval.retriever import Retriever
from app.llm.generator import generate_answer
from config import DATA_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

retriever = Retriever()
is_index_built = False
ingested_files = []  # track which files are in the index


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str


@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs and build the vector index.
    Each call ADDS to the existing index — does not replace it.
    """
    global retriever, is_index_built, ingested_files

    saved_paths = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF. Only PDF files are supported."
            )

        save_path = DATA_DIR / file.filename
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved_paths.append(str(save_path))
        ingested_files.append(file.filename)
        logger.info(f"PDF saved: {save_path}")

    # Build index with ALL files ingested so far
    retriever = Retriever()
    all_paths = [str(DATA_DIR / f) for f in ingested_files]
    retriever.build_index(all_paths)
    is_index_built = True

    return {
        "message": f"Successfully ingested {len(saved_paths)} file(s)",
        "files_ingested": saved_paths,
        "total_files_in_index": len(ingested_files),
        "total_chunks": retriever.store.total_chunks()
    }


@router.delete("/ingest/reset")
async def reset_index():
    """
    Clear the index and start fresh.
    """
    global retriever, is_index_built, ingested_files

    retriever = Retriever()
    is_index_built = False
    ingested_files = []

    return {"message": "Index reset successfully."}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question against all ingested documents.
    """
    if not is_index_built:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call /ingest first."
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
    return {
        "status": "ok",
        "index_built": is_index_built,
        "files_ingested": ingested_files,
        "total_chunks": retriever.store.total_chunks() if is_index_built else 0
    }