# Advanced RAG Pipeline

Interactive CLI for retrieval-augmented generation using the Chroma Cloud collection created by `01_llm_chunking_embedding`. Implements query rewriting, dual retrieval (original + rewritten query), LLM-based reranking, and answer generation.

## Overview

This pipeline answers questions about the Made Tech handbook using an advanced RAG flow:

1. **Query rewriting** — LLM rewrites the user's question for better retrieval
2. **Dual retrieval** — Fetch chunks for both original and rewritten questions
3. **Merge & rerank** — Deduplicate and LLM-rerank by relevance
4. **Answer** — Generate response from top-k chunks

## Requirements

- Python 3.10+
- Dependencies from `experiments/requirements.txt` (litellm, chromadb, openai, pyyaml, python-dotenv, pydantic)
- `backend/.env` with the following keys set:
  - `OPENAI_API_KEY` — for embeddings
  - `GROQ_API_KEY` — for LLM calls (rewriting, reranking, answering)
  - `CHROMA_API_KEY` — Chroma Cloud API key
  - `TENANT_CHROMA` — Chroma Cloud tenant ID
- Run `01_llm_chunking_embedding` first to populate the Chroma Cloud collection

## Usage

From the **repo root**, activate the experiments virtual environment and run:

```powershell
.\experiments\.venv\Scripts\Activate.ps1
python -m experiments.scripts.02_advanced_rag_interactive_cli.main
```

The script starts an interactive loop — type your question and press Enter. Type `quit` or `exit` to stop.

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `vector_db.database` | Chroma Cloud database name | `madetech_handbook` |
| `vector_db.collection_name` | Chroma Cloud collection | `docs` |
| `embedding_model` | OpenAI embedding model (must match `01_llm_chunking_embedding`) | `text-embedding-3-large` |
| `retrieval.retrieval_k` | Chunks per query (original + rewritten) | `10` |
| `retrieval.final_k` | Chunks passed to LLM after reranking | `7` |
| `model` | LLM for rewriting, reranking, and answering | `groq/openai/gpt-oss-20b` |

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `retrieval.py` | Chroma Cloud connection and semantic search |
| `query_rewriting.py` | LLM-based query expansion |
| `reranking.py` | LLM-based chunk reranking |
| `rag.py` | Pipeline orchestration (fetch_context, answer_question) |
| `main.py` | CLI entry point |
