# Backend

FastAPI backend for the RAG handbook chatbot. Serves the chat API, highlights API, and (in production) the frontend.

## Quick Start

```bash
cd backend
uv run uvicorn src.app:app --reload --port 9481
```

Ensure `backend/.env` has `GROQ_API_KEY` and `OPENAI_API_KEY`. The vector database must exist at `data/vector_db` (see [Ingestion](#ingestion)).

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      src/app.py                          │
                    │  FastAPI app · CORS · startup · API routes              │
                    └───────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ /api/chat     │         │ /api/highlights  │         │ /api/handbook     │
│ ChatRequest   │         │ HighlightsRequest│         │ (static data)     │
└───────┬───────┘         └────────┬─────────┘         └────────┬──────────┘
        │                          │                            │
        ▼                          ▼                            │
┌───────────────────┐    ┌───────────────────┐                  │
│ RAGService        │    │ highlights_service │                  │
│ get_rag_response()│    │ get_relevance_    │                  │
└─────────┬─────────┘    │ highlights()      │                  │
          │              └───────────────────┘                  │
          │  (delegates to rag pipeline)                         │
          ▼                                                      ▼
┌───────────────────────────────────────┐              ┌──────────────────┐
│ src/rag/pipeline.py                   │              │ handbook_loader  │
│ answer_question()                      │              │ load_handbook_   │
│   → fetch_context()                   │              │ documents()      │
│   → make_rag_messages()               │              └──────────────────┘
│   → litellm.completion()              │
└───────────────┬───────────────────────┘
                │
    ┌───────────┼───────────┬───────────────┐
    ▼           ▼           ▼               ▼
retrieval  query_rewriting  reranking   config_loader
(ChromaDB) (LLM)            (LLM)       (config.yaml)
```

---

## Directory Structure

```
backend/
├── config.yaml           # RAG config (approach, retrieval, model)
├── data/
│   ├── handbook/         # Markdown source files (loaded at startup)
│   └── vector_db/         # ChromaDB (created by ingest script)
├── src/                   # Application code
│   ├── app.py             # FastAPI app, routes, startup
│   ├── config_loader.py   # Load and resolve config.yaml
│   ├── rag_service.py     # RAG orchestration (delegates to rag/)
│   ├── handbook_loader.py # Load .md files → HandbookDoc
│   ├── highlights_service.py  # "Highlight with AI" (litellm)
│   └── rag/               # RAG pipeline
│       ├── pipeline.py    # fetch_context, answer_question
│       ├── retrieval.py  # ChromaDB + OpenAI embeddings
│       ├── query_rewriting.py  # LLM query expansion
│       └── reranking.py  # LLM chunk reordering
├── utils/                 # Shared models and prompts
│   ├── models.py          # HandbookDoc, Result, SourceChunk, API models
│   └── prompts.py         # RAG_SYSTEM_PROMPT, etc.
└── scripts/               # Ingestion (chunking, embedding)
    └── 01_llm_chunking_embedding/
```

---

## Request Flows

### Chat (`POST /api/chat`)

1. **Request**: `ChatRequest` with `query` and `history`
2. **RAGService.get_rag_response()**:
   - Converts `history` to `list[dict]`
   - Calls `rag.answer_question()`
3. **rag.pipeline.answer_question()**:
   - **fetch_context()**:
     - Embeds question (OpenAI), queries ChromaDB for `retrieval_k` chunks
     - If `use_query_rewriting`: rewrites question, fetches again, merges
     - If `use_reranking`: LLM reorders chunks, take top `final_k`
   - **make_rag_messages()**: System prompt + context + history + question
   - **litellm.completion()**: Groq model from config
4. **RAGService**: Maps `Result` chunks to `SourceChunk`, returns `{content, sources}`

### Highlights (`POST /api/highlights`)

1. **Request**: `HighlightsRequest` with `answer` and `document_content`
2. **get_relevance_highlights()**: Litellm completion (Groq primary, OpenAI fallback) with JSON mode
3. Returns verbatim phrases from the document that support the answer; frontend wraps them in `<mark>`

---

## Configuration

`config.yaml` (or `RAG_CONFIG_PATH`) controls:

| Key | Description |
|-----|-------------|
| `vector_db.path` | ChromaDB directory (relative to backend root) |
| `vector_db.collection_name` | Collection name (default `docs`) |
| `embedding_model` | Must match ingestion (default `text-embedding-3-large`) |
| `retrieval.retrieval_k` | Chunks per query (default 20) |
| `retrieval.final_k` | Chunks passed to LLM after reranking (default 10) |
| `model` | LLM for rewriting, reranking, answer (default `groq/openai/gpt-oss-20b`) |
| `approach.use_query_rewriting` | Expand follow-ups for better retrieval |
| `approach.use_reranking` | LLM reorders chunks before generation |

**Approach variants:**
- `basic_rag`: both `false`
- `with_reranking`: rewriting `false`, reranking `true`
- `with_rewriting`: rewriting `true`, reranking `false`
- `with_rewriting_and_reranking`: both `true` (recommended)

---

## Key Files Explained

| File | Role |
|------|------|
| **app.py** | Entry point. Loads handbook, initialises RAG, defines routes. CORS for local dev; static serving in production. |
| **rag_service.py** | Thin wrapper: calls `answer_question`, extracts sources. No retrieval logic. |
| **rag/pipeline.py** | Core RAG flow. `fetch_context` does retrieval + optional rewrite + optional rerank. `answer_question` builds messages and calls the LLM. |
| **rag/retrieval.py** | ChromaDB connection and `fetch_context_unranked` (embed → query → Result list). |
| **rag/query_rewriting.py** | Single function `rewrite_query`: LLM turns a follow-up into a standalone search query. |
| **rag/reranking.py** | `rerank`: LLM returns permutation of chunk IDs; we reorder. `merge_chunks`: dedupe when combining original + rewritten results. |
| **config_loader.py** | Reads YAML, resolves paths, merges `approach` flags, provides defaults. |
| **handbook_loader.py** | Scans `data/handbook/*.md`, parses frontmatter, builds `HandbookDoc` list. |
| **highlights_service.py** | Litellm JSON completion to extract supporting phrases from a document. |
| **utils/models.py** | Canonical Pydantic models (HandbookDoc, Result, SourceChunk, ChatRequest, etc.). |
| **utils/prompts.py** | System prompts (RAG_SYSTEM_PROMPT, etc.). |

---

## Ingestion

The vector database is created separately. From the repo root:

```bash
python -m backend.scripts.01_llm_chunking_embedding.main
```

(Adjust to your project’s ingest entrypoint if different.) This chunks handbook markdown and embeds with OpenAI `text-embedding-3-large`, storing results in ChromaDB at `data/vector_db`.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq API key (chat, highlights) |
| `OPENAI_API_KEY` | OpenAI API key (embeddings, fallback) |
| `RAG_CONFIG_PATH` | Override path to config YAML |
| `FRONTEND_PATH` | Static frontend dir (Docker default: `/app/frontend/dist`) |

---

## Running

**Local:**
```bash
uv run uvicorn src.app:app --reload --port 9481
```
Frontend (Vite) typically runs on 3000; CORS is configured for that origin.

**Docker:** The image serves both API and static frontend. API on `/api/*`, frontend on other paths.
