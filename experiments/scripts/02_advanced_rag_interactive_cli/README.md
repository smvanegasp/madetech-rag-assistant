# Advanced RAG Pipeline

Runs retrieval-augmented generation using the ChromaDB created by `01_llm_chunking_embedding`. Implements query rewriting, dual retrieval (original + rewritten query), LLM-based reranking, and answer generation.

## Overview

This pipeline answers questions about the Made Tech handbook using an advanced RAG flow:

1. **Query rewriting** — LLM rewrites the user's question for better retrieval
2. **Dual retrieval** — Fetch chunks for both original and rewritten questions
3. **Merge & rerank** — Deduplicate and LLM-rerank by relevance
4. **Answer** — Generate response from top-k chunks

## Requirements

- Python 3.10+
- Dependencies from `experiments/requirements.txt` (litellm, chromadb, openai, pyyaml, python-dotenv, pydantic)
- API keys in `backend/.env` (OpenAI for embeddings; LiteLLM/Groq for LLM calls)
- Run `01_llm_chunking_embedding` first to create the ChromaDB

## Usage

Run from the repo root:

```bash
# Default question: "What cycling benefits do I have?"
python -m experiments.scripts.02_advanced_rag.main

# Custom question
python -m experiments.scripts.02_advanced_rag.main "How many days of annual leave do I get?"
```

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `vector_db.path` | ChromaDB path (relative to this script) | `../01_llm_chunking_embedding/output/preprocessed_db` |
| `vector_db.collection_name` | ChromaDB collection | `docs` |
| `embedding_model` | OpenAI embedding model (must match 01) | `text-embedding-3-large` |
| `retrieval.retrieval_k` | Chunks per query (original + rewritten) | `10` |
| `retrieval.final_k` | Chunks passed to LLM after reranking | `7` |
| `model` | LLM for rewriting, reranking, and answering | `groq/openai/gpt-oss-20b` |

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `retrieval.py` | ChromaDB connection and semantic search |
| `query_rewriting.py` | LLM-based query expansion |
| `reranking.py` | LLM-based chunk reranking |
| `rag.py` | Pipeline orchestration (fetch_context, answer_question) |
| `main.py` | CLI entry point |
