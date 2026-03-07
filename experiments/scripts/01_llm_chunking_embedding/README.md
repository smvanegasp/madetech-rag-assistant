# LLM Chunking and Embedding Pipeline

Loads handbook documents from the Made Tech handbook, splits them into semantic chunks via an LLM (with headlines and summaries), generates embeddings, and stores them in ChromaDB for RAG retrieval.

## Overview

This pipeline prepares handbook content for retrieval-augmented generation (RAG):

1. **Load** — Reads markdown documents from the handbook directory
2. **Chunk** — Uses an LLM to split each document into semantic chunks with headlines and summaries
3. **Embed** — Generates vector embeddings via the OpenAI API
4. **Store** — Persists chunks and embeddings in ChromaDB

## Requirements

- Python 3.10+
- Dependencies from `experiments/requirements.txt` (litellm, chromadb, openai, tenacity, tqdm, pyyaml, python-dotenv, pydantic)
- API keys in `backend/.env` (OpenAI for embeddings; LiteLLM/Groq for chunking if using `groq/` models)

## Usage

Run from the repo root:

```bash
python -m experiments.scripts.01_llm_chunking_embedding.main
```

Ensure your virtual environment is activated and dependencies are installed:

```bash
pip install -r experiments/requirements.txt
pip install -e backend/
```

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `handbook_path` | Path to handbook markdown files (relative to repo root) | `backend/data/handbook` |
| `document_limit` | Cap documents for testing (`null` = process all) | `5` |
| `average_chunk_size` | Target chunk size in characters for estimating chunk count | `400` |
| `chunking_workers` | Parallel workers for chunking (1 to avoid rate limits) | `3` |
| `chunking_model` | LLM for chunk generation (e.g. `groq/openai/gpt-oss-20b`) | — |
| `max_tokens` | Max tokens per LLM response | `65536` |
| `vector_db.path` | ChromaDB store directory (relative to this script) | `preprocessed_db` |
| `vector_db.collection_name` | ChromaDB collection name | `docs` |
| `embedding_model` | OpenAI embedding model | `text-embedding-3-large` |

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `chunking.py` | LLM-based semantic chunking (sequential and parallel) |
| `embeddings.py` | Embedding generation and ChromaDB storage |
| `main.py` | Pipeline orchestration |

## Output

- **ChromaDB** — A persistent vector store in `preprocessed_db/` (or the path in config)
- Each chunk includes: headline, summary, and original text, plus document metadata (id, title, category)
