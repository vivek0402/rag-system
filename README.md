# RAG System — Production-Style Retrieval-Augmented Generation

A production-style RAG (Retrieval-Augmented Generation) pipeline built from scratch in Python.
Upload any PDF and ask questions — the system retrieves relevant context and generates grounded answers using an LLM.

## System Architecture

```
PDF Input
   ↓
PDF Loader (PyMuPDF)        — extracts text page by page
   ↓
Text Chunker                — splits text into overlapping chunks (500 chars, 50 overlap)
   ↓
Embedding Model             — converts chunks to 384-dim vectors (all-MiniLM-L6-v2)
   ↓
FAISS Vector Store          — indexes vectors for fast similarity search
   ↓
Retriever                   — embeds query, finds top-k relevant chunks via cosine similarity
   ↓
LLM Generator (Groq)        — generates grounded answer from retrieved context only
   ↓
FastAPI                     — exposes everything as a REST API
```

## Tech Stack

| Component | Technology |
|---|---|
| PDF Parsing | PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (IndexFlatL2) |
| LLM | Llama 3.1 via Groq API |
| API Framework | FastAPI + Uvicorn |
| Language | Python 3.11 |

## Project Structure

```
rag-system/
├── app/
│   ├── ingestion/       # PDF loading and chunking
│   ├── embeddings/      # Sentence transformer embeddings
│   ├── vectorstore/     # FAISS index management
│   ├── retrieval/       # Retrieval pipeline orchestration
│   ├── llm/             # LLM integration (Groq)
│   └── api/             # FastAPI routes
├── data/raw/            # PDF storage (gitignored)
├── config.py            # Centralized configuration
├── main.py              # Application entry point
└── requirements.txt     # Dependencies
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/vivek0402/rag-system.git
cd rag-system
```

**2. Create and activate virtual environment**

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**5. Run the API**

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## API Endpoints

### POST /api/v1/ingest

Upload a PDF to build the vector index.

### POST /api/v1/query

Ask a question against the ingested document.

**Request:**

```json
{
  "question": "What are the key findings in this document?",
  "top_k": 3
}
```

**Response:**

```json
{
  "answer": "Based on the document, the key findings are...",
  "sources": [{"source": "document.pdf", "page": 1, "score": 0.85}],
  "model": "llama-3.1-8b-instant"
}
```

### GET /api/v1/health

Health check endpoint.

## Key Design Decisions

**Why chunk with overlap?**
Overlap preserves context at chunk boundaries. Without it, sentences split across chunks lose meaning and retrieval quality drops significantly.

**Why FAISS over a database?**
FAISS is optimized specifically for vector similarity search at scale. It uses approximate nearest neighbor algorithms that are orders of magnitude faster than brute-force comparison.

**Why ground the LLM with a strict prompt?**
LLMs hallucinate — they generate plausible but factually wrong answers when relying on training data. By constraining the LLM to answer only from retrieved chunks, we eliminate hallucination and make answers auditable.

**Why return sources with every answer?**
Provenance — knowing where an answer came from — is critical for trust in production systems. Users can verify answers, and the system becomes auditable.

## Interview Prep

This project demonstrates:

- End-to-end RAG pipeline design
- Vector similarity search (cosine similarity, FAISS)
- Embedding models and semantic search
- LLM prompt engineering for grounded generation
- Production API design with FastAPI
- Clean modular architecture with separation of concerns

```

Save it.

---

## Step 2: Push to GitHub

**First, create a repository on GitHub:**

1. Go to https://github.com
2. Click the `+` button → "New repository"
3. Name it `rag-system`
4. Set it to **Public**
5. Do NOT check "Add README" or "Add .gitignore" — we already have these
6. Click "Create repository"

GitHub will show you a page with commands. You need the URL of your repo — it looks like:
```
<https://github.com/vivek0402/rag-system.git>
