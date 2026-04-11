# RAG System — LangChain Variant: Architecture & Progress

_Last updated: 2026-04-11_

---

## Project State Audit

### Git
- **Current branch**: `langchain-variant` (created 2026-04-11)
- **Base branch**: `main`
- **Uncommitted on branch at creation**: `Dockerfile`, `Dockerfile.streamlit` (modified), `requirements.docker.txt` (untracked)

### Docker (INCOMPLETE — not yet run)
The Docker setup files are complete and correct but were never executed. No containers exist.
- `Dockerfile` — builds FastAPI backend, uses `requirements.docker.txt`
- `Dockerfile.streamlit` — builds Streamlit UI
- `docker-compose.yml` — wires both services, mounts `./data`, reads `.env`
- `requirements.docker.txt` — minimal deps for container (no torch/numpy bloat)
- **Fix applied in Dockerfile**: `libgl1-mesa-glx` → `libgl1` (correct package for Debian Bookworm)

**To run Docker** (manual step, when ready):
```bash
# From e:/Projects/rag-system
docker compose up --build
# API → http://localhost:8000/docs
# UI  → http://localhost:8501
```
**Requires**: Docker Desktop running on Windows.

### API Key
- `.env` file **exists** at project root and contains `GROQ_API_KEY`
- Key is valid (not lost) — check `.env` directly
- Never commit `.env`

---

## Existing Architecture (from-scratch pipeline)

```
PDF files
  └─ load_pdf()               app/ingestion/pdf_loader.py   (PyMuPDF, page-by-page)
       └─ chunk_text()         app/ingestion/chunker.py      (500 chars, 50 overlap)
            └─ embed_texts()   app/embeddings/embedder.py    (all-MiniLM-L6-v2 via sentence-transformers)
                 └─ FAISSVectorStore  app/vectorstore/faiss_store.py  (IndexFlatL2, L2-normalised)
                      └─ Retriever.search()  app/retrieval/retriever.py
                           └─ generate_answer()  app/llm/generator.py  (Groq SDK, llama-3.1-8b-instant)
```

**API layer** (`app/api/routes.py`):
- `POST /api/v1/ingest` — upload PDFs, build + persist index
- `POST /api/v1/query` — retrieve + generate answer
- `DELETE /api/v1/ingest/reset` — wipe index
- `GET /api/v1/health` — status check

**Persistence**: `data/index/faiss.index` + `data/index/chunks.json`

**Config** (`config.py`):
- `DATA_DIR = data/raw`
- `INDEX_DIR = data/index`

---

## LangChain Variant Plan (langchain-variant branch)

### Component Swaps

| Original | LangChain replacement |
|---|---|
| custom `chunk_text()` | `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` |
| `FAISSVectorStore` (custom) | `langchain_community.vectorstores.FAISS` |
| `embed_texts` / `embed_query` | `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` |
| `generate_answer()` (Groq SDK) | `ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=500)` |
| `Retriever.search()` | `RetrievalQA` chain with `return_source_documents=True` |

### Files to create/modify

| File | Action |
|---|---|
| `app/langchain_rag.py` | **Create** — `LangChainRAG` class |
| `config.py` | **Edit** — add `LC_INDEX_DIR = data/lc_index` |
| `requirements.txt` | **Edit** — add 4 langchain packages |
| `app/api/routes.py` | **Edit** — add LC instance, dual ingest, new endpoint, health field |

### Files that must NOT be touched
- `app/ingestion/pdf_loader.py`
- `app/ingestion/chunker.py`
- `app/embeddings/embedder.py`
- `app/vectorstore/faiss_store.py`
- `app/retrieval/retriever.py`
- `app/llm/generator.py`
- `main.py`
- `streamlit_app.py`
- `conftest.py`
- `tests/test_retriever.py`

---

## Progress Checklist

- [x] Read all source files
- [x] Create `langchain-variant` branch
- [x] Write `tasks/todo.md` (this file)
- [x] Add LangChain deps to `requirements.txt` (pinned actual versions)
- [x] Install new deps in venv (fresh Python 3.14 venv — see lessons.md)
- [x] Update `config.py` — added `LC_INDEX_DIR`
- [x] Create `app/langchain_rag.py`
- [x] Update `app/api/routes.py`
- [x] Verify: `from app.langchain_rag import LangChainRAG` → `import ok`
- [x] Verify: `from app.api.routes import router` → `routes ok`
- [x] Verify: `pytest tests/ -x -q` → 14 passed, 0 failed
- [ ] Commit

---

## Lessons Learned

_(Updated as issues arise — see also tasks/lessons.md)_
