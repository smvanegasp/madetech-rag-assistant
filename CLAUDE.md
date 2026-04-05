# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack RAG chatbot over the MadeTech company handbook. Uses Groq for chat completion, ChromaDB Cloud for semantic search, and OpenAI for embeddings.

## Development Commands

### Backend (from `backend/` directory)
```bash
uv venv                          # Create virtual environment
source .venv/bin/activate        # Unix/Mac
.venv\Scripts\activate           # Windows
uv pip install -e .              # Install dependencies
uvicorn src.app:app --reload --port 9481   # Start dev server
```

### Frontend (from `frontend/` directory)
```bash
npm install        # Install dependencies
npm run dev        # Start Vite dev server on port 3000
npm run build      # Production build to dist/
```

### Vector DB ingestion (first-time only, from `backend/`)
```bash
python -m scripts.ingest
```

### Docker (production)
```bash
docker build -t madetech-rag-assistant .
docker run -p 9481:9481 --env-file backend/.env madetech-rag-assistant
```

### No test framework or linting tools are configured.

## Architecture

**Frontend:** React 19 + TypeScript + Vite. Dev server on port 3000 proxies API calls to backend on port 9481. In production (Docker), FastAPI serves the built frontend as static files.

**Backend:** FastAPI + Uvicorn on port 9481. Python 3.12+, dependencies managed with UV.

**RAG Pipeline** (`backend/src/rag/`):
1. **Query rewriting** (`query_rewriting.py`) — rewrites follow-up questions into standalone queries using chat history
2. **Retrieval** (`retrieval.py`) — embeds query with OpenAI `text-embedding-3-large`, queries ChromaDB Cloud for top-k chunks
3. **Reranking** (`reranking.py`) — LLM reorders retrieved chunks by relevance
4. **Generation** (`pipeline.py`) — builds prompt with context + history, calls LLM via LiteLLM

RAG behavior is configured in `backend/config.yaml` (query rewriting, reranking, k values, model selection). The embedding model in config must match what was used during ingestion.

**LLM routing:** LiteLLM calls Groq (`groq/openai/gpt-oss-20b`) as primary, with OpenAI as fallback.

**Data flow:** `POST /api/chat` → `rag_service.py` → `pipeline.answer_question()` → returns answer + source chunks.

**Other services:**
- Chat logging to Supabase PostgreSQL (`chat_logger.py`)
- Contact/feedback emails via Resend (`contact_service.py`)

## Key Files

- `backend/src/app.py` — FastAPI routes, CORS, startup
- `backend/src/rag/pipeline.py` — Core RAG orchestration
- `backend/config.yaml` — RAG pipeline configuration
- `backend/utils/models.py` — Pydantic models (shared data contracts)
- `backend/utils/prompts.py` — System prompts for LLM calls
- `frontend/App.tsx` — Root component with all state management
- `frontend/services/apiService.ts` — API client (uses `VITE_BACKEND_URL` env var)

## Environment Variables

Required in `backend/.env` (see `backend/.env.example`):
- `GROQ_API_KEY`, `OPENAI_API_KEY` — LLM providers
- `CHROMA_API_KEY`, `TENANT_CHROMA` — ChromaDB Cloud
- `DATABASE_URL` — Supabase PostgreSQL for chat logging
- `RESEND_API_KEY`, `CONTACT_EMAIL` — Email service

## Handbook Data

Source markdown files live in `backend/data/handbook/` organized by category (benefits, guides, roles, etc.). The ingestion script (`backend/scripts/01_llm_chunking_embedding/`) chunks these files, generates headlines/summaries via LLM, embeds them, and stores in ChromaDB Cloud.
