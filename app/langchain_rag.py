# app/langchain_rag.py

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import load_pdf

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.1-8b-instant"

_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Do NOT use your own knowledge or make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class LangChainRAG:
    """
    LangChain-based RAG pipeline.

    Mirrors the interface of Retriever + generate_answer using LangChain internals:
    - PDF loading:  app/ingestion/pdf_loader.py (reused as-is via PyMuPDF)
    - Chunking:     RecursiveCharacterTextSplitter (500 chars, 50 overlap)
    - Embeddings:   HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    - Vector store: langchain_community.vectorstores.FAISS
    - LLM:          ChatGroq (llama-3.1-8b-instant, temp=0.1, max_tokens=500)
    - Chain:        LCEL retrieval chain with source document tracking

    Public interface:
        build_index(pdf_paths)         -> None
        query(question, top_k=3)       -> {"answer", "sources", "model"}
        save_index(directory)          -> None
        load_index(directory)          -> bool
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self.llm = ChatGroq(
            model_name=MODEL_NAME,
            temperature=0.1,
            max_tokens=500,
        )
        self.vectorstore = None
        self._is_built = False
        logger.info("LangChainRAG initialized")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, pdf_paths: list[str]) -> None:
        """
        Load PDFs, chunk, embed, and store in FAISS.

        Args:
            pdf_paths: List of paths to PDF files.
        """
        documents: list[Document] = []

        for path in pdf_paths:
            logger.info(f"Processing: {path}")
            pages = load_pdf(path)
            for page in pages:
                documents.append(
                    Document(
                        page_content=page["text"],
                        metadata={"source": page["source"], "page": page["page"]},
                    )
                )

        if not documents:
            raise ValueError("No content extracted from the provided PDFs.")

        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self._is_built = True
        logger.info(f"LangChain FAISS index built with {len(chunks)} chunks")

    def save_index(self, directory: str | Path) -> None:
        """
        Save FAISS index to disk using LangChain's built-in method.

        Saves two files: index.faiss and index.pkl

        Args:
            directory: Folder to save index files into.
        """
        if not self._is_built:
            raise RuntimeError("No index to save. Call build_index() first.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(directory))
        logger.info(f"LangChain index saved to {directory}")

    def load_index(self, directory: str | Path) -> bool:
        """
        Load FAISS index from disk.

        Args:
            directory: Folder containing index.faiss and index.pkl

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        directory = Path(directory)
        index_file = directory / "index.faiss"

        if not index_file.exists():
            logger.info("No saved LangChain index found — starting fresh")
            return False

        self.vectorstore = FAISS.load_local(
            str(directory),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._is_built = True
        logger.info(f"LangChain index loaded from {directory}")
        return True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Retrieve relevant chunks and generate a grounded answer.

        Args:
            question: The user's question.
            top_k:    Number of chunks to retrieve.

        Returns:
            {"answer": str, "sources": list[dict], "model": str}
            Each source dict: {"source": str, "page": int, "score": float}
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Fetch source docs separately to expose them alongside the answer
        source_docs = retriever.invoke(question)

        prompt = PromptTemplate(
            template=_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        def _format_docs(docs: list[Document]) -> str:
            return "\n\n---\n\n".join(
                f"Source: {d.metadata.get('source', '')} | Page: {d.metadata.get('page', '')}\n{d.page_content}"
                for d in docs
            )

        chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"LangChain query: {question[:50]}...")
        answer = chain.invoke(question)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "score": 0.0,  # LCEL retriever does not expose similarity scores
            }
            for doc in source_docs
        ]

        logger.info(f"LangChain answer generated ({len(answer)} chars)")

        return {
            "answer": answer,
            "sources": sources,
            "model": MODEL_NAME,
        }
