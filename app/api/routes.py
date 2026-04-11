# app/api/routes.py

import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from app.retrieval.retriever import Retriever
from app.llm.generator import generate_answer
from app.langchain_rag import LangChainRAG
from config import DATA_DIR, INDEX_DIR, LC_INDEX_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize retriever and attempt to load saved index
retriever = Retriever()
ingested_files = []

# Try to load persisted index on startup
if retriever.store.load(INDEX_DIR):
    is_index_built = True
    logger.info("Restored index from disk on startup")
else:
    is_index_built = False
    logger.info("No saved index found — starting fresh")

# Initialize LangChain RAG and attempt to load saved index
lc_rag = LangChainRAG()

if lc_rag.load_index(LC_INDEX_DIR):
    lc_index_built = True
    logger.info("Restored LangChain index from disk on startup")
else:
    lc_index_built = False
    logger.info("No saved LangChain index found — starting fresh")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str


@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    global retriever, is_index_built, ingested_files, lc_rag, lc_index_built

    saved_paths = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF."
            )
        save_path = DATA_DIR / file.filename
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(save_path))
        ingested_files.append(file.filename)
        logger.info(f"PDF saved: {save_path}")

    retriever = Retriever()
    all_paths = [str(DATA_DIR / f) for f in ingested_files]
    retriever.build_index(all_paths)
    is_index_built = True

    # Save index to disk immediately after building
    retriever.store.save(INDEX_DIR)
    logger.info("Index persisted to disk")

    # Keep LangChain pipeline in sync
    lc_rag = LangChainRAG()
    lc_rag.build_index(all_paths)
    lc_rag.save_index(LC_INDEX_DIR)
    lc_index_built = True
    logger.info("LangChain index built and persisted to disk")

    return {
        "message": f"Successfully ingested {len(saved_paths)} file(s)",
        "files_ingested": saved_paths,
        "total_files_in_index": len(ingested_files),
        "total_chunks": retriever.store.total_chunks()
    }


@router.delete("/ingest/reset")
async def reset_index():
    global retriever, is_index_built, ingested_files, lc_rag, lc_index_built

    retriever = Retriever()
    is_index_built = False
    ingested_files = []

    # Delete saved index files
    index_path = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.json"
    if index_path.exists():
        index_path.unlink()
    if chunks_path.exists():
        chunks_path.unlink()

    # Reset LangChain pipeline and delete its index files
    lc_rag = LangChainRAG()
    lc_index_built = False
    lc_faiss = LC_INDEX_DIR / "index.faiss"
    lc_pkl = LC_INDEX_DIR / "index.pkl"
    if lc_faiss.exists():
        lc_faiss.unlink()
    if lc_pkl.exists():
        lc_pkl.unlink()
    logger.info("LangChain index reset")

    return {"message": "Index reset successfully."}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
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
        "lc_index_built": lc_index_built,
        "files_ingested": ingested_files,
        "total_chunks": retriever.store.total_chunks() if is_index_built else 0
    }


@router.post("/langchain/query", response_model=QueryResponse)
async def langchain_query(request: QueryRequest):
    if not lc_index_built:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call /ingest first."
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"LangChain query received: {request.question}")
    response = lc_rag.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=response["answer"],
        sources=response["sources"],
        model=response["model"]
    )