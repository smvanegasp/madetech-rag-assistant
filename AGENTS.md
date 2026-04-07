# AGENTS.md

This file provides guidance to AI coding agents (Cursor, Codex, etc.) when working in this repository.

## Project Overview

Full-stack RAG chatbot over the MadeTech company handbook. Uses Groq for chat completion, ChromaDB Cloud for semantic search, and OpenAI for embeddings.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + TypeScript + Vite (port 3000) |
| Backend | FastAPI + Uvicorn (port 9481) |
| Embeddings | OpenAI `text-embedding-3-large` |
| LLM | LiteLLM → Groq primary, OpenAI fallback |
| Vector DB | ChromaDB Cloud |
| Chat logging | Supabase PostgreSQL |
| Email | Resend |

## Development Commands

### Backend (run from `backend/`)

```powershell
uv venv
.venv\Scripts\activate          # Windows
uv pip install -e .
uvicorn src.app:app --reload --port 9481
```

### Frontend (run from `frontend/`)

```bash
npm install
npm run dev       # Dev server on port 3000
npm run build     # Production build to dist/
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

> No test framework or linting tools are configured.

## Architecture

### RAG Pipeline (`backend/src/rag/`)

1. **Query rewriting** (`query_rewriting.py`) — rewrites follow-up questions into standalone queries using chat history
2. **Retrieval** (`retrieval.py`) — embeds query with OpenAI, queries ChromaDB Cloud for top-k chunks
3. **Reranking** (`reranking.py`) — LLM reorders retrieved chunks by relevance
4. **Generation** (`pipeline.py`) — builds prompt with context + history, calls LLM via LiteLLM

RAG behaviour is configured in `backend/config.yaml` (query rewriting, reranking, k values, model selection). The embedding model in config **must match** what was used during ingestion.

### Data Flow

```
POST /api/chat → rag_service.py → pipeline.answer_question() → answer + source chunks
```

## Key Files

| File | Purpose |
|------|---------|
| `backend/src/app.py` | FastAPI routes, CORS, startup |
| `backend/src/rag/pipeline.py` | Core RAG orchestration |
| `backend/config.yaml` | RAG pipeline configuration |
| `backend/utils/models.py` | Pydantic models (shared data contracts) |
| `backend/utils/prompts.py` | System prompts for LLM calls |
| `frontend/App.tsx` | Root component with all state management |
| `frontend/services/apiService.ts` | API client (`VITE_BACKEND_URL` env var) |

## Environment Variables

Required in `backend/.env` (see `backend/.env.example`):

```
GROQ_API_KEY        # Primary LLM provider
OPENAI_API_KEY      # Embeddings + fallback LLM
CHROMA_API_KEY      # ChromaDB Cloud
TENANT_CHROMA       # ChromaDB Cloud tenant
DATABASE_URL        # Supabase PostgreSQL (chat logging)
RESEND_API_KEY      # Email service
CONTACT_EMAIL       # Recipient for contact/feedback emails
```

## Handbook Data

Source markdown files: `backend/data/handbook/` (organised by category: benefits, guides, roles, etc.)

Ingestion script: `backend/scripts/01_llm_chunking_embedding/` — chunks files, generates headlines/summaries via LLM, embeds them, and stores in ChromaDB Cloud.
