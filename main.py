# main.py

import logging
from fastapi import FastAPI
from app.api.routes import router
from config import LOG_LEVEL

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation over PDF documents",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "RAG System is running. Visit /docs for API documentation."}