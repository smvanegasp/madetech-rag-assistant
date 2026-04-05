# RAG Experiment: Model, Rerank & Rewrite Comparison

Evaluates RAG variants across multiple LLMs, reranking, and query rewriting strategies. Each experiment is scored by an LLM-as-judge (accuracy, completeness, relevance) with end-to-end latency tracking.

## Experiment Design

Each experiment declares three parameters:

| Parameter | Description |
|-----------|-------------|
| `model` | LLM for RAG (query rewriting, reranking, answer generation) |
| `use_query_rewriting` | Expand follow-up questions into standalone queries |
| `use_reranking` | LLM reorders retrieved chunks by relevance |

A separate `judge_model` scores all experiments for fair comparison.

### Current Models

| Model | Provider | Notes |
|-------|----------|-------|
| `groq/openai/gpt-oss-20b` | Groq (free) | Current production model |
| `openai/gpt-4o-mini` | OpenAI | Uses credits |
| `anthropic/claude-3-haiku-20240307` | Anthropic | Requires `ANTHROPIC_API_KEY` |

## Requirements

Install experiment dependencies (from repo root):

```bash
pip install -r experiments/requirements.txt
```

Environment variables needed in `backend/.env`:
- `OPENAI_API_KEY` — embeddings + OpenAI models
- `CHROMA_API_KEY`, `TENANT_CHROMA` — ChromaDB Cloud
- `GROQ_API_KEY` — Groq models
- `ANTHROPIC_API_KEY` — Anthropic models (if testing Claude)

## Prerequisites

1. **00_questions_generation** — Run first to generate eval questions (`validation.jsonl`).
2. **01_llm_chunking_embedding** — ChromaDB vector store must exist.

## Usage

From the repo root:

```bash
python -m experiments.scripts.03_experiment_rerank_rewrite.main
```

To reload a previous run (view results or regenerate plots):

```bash
# Edit config.yaml and set load_from to the timestamp folder
load_from: "output/2026-03-08_12-00-00"
```

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `questions.source` | Path to `00_questions_generation/output` | `../00_questions_generation/output` |
| `questions.use_latest` | Use most recent timestamp folder | `true` |
| `questions.files` | JSONL files to load | `["validation.jsonl"]` |
| `questions.n_questions` | Limit number of questions (`null` = all) | `null` |
| `questions.random_seed` | Seed for reproducible sampling | `42` |
| `vector_db.database` | ChromaDB Cloud database | `madetech_handbook` |
| `vector_db.collection_name` | ChromaDB collection | `docs` |
| `embedding_model` | Must match 01_llm_chunking_embedding | `text-embedding-3-large` |
| `retrieval.retrieval_k` | Chunks per query | `20` |
| `retrieval.final_k` | Chunks passed to LLM after reranking | `10` |
| `judge_model` | LLM for scoring (separate from RAG model) | `groq/llama-3.3-70b-versatile` |
| `experiments` | List of variants (each with `name`, `model`, `use_query_rewriting`, `use_reranking`) | See config.yaml |
| `workers` | Parallel processes | `3` |
| `output` | Output directory | `output` |
| `load_from` | Load previous run (`null` = fresh run) | `null` |
| `save_plots` | Generate plots after run | `true` |

## Output

Results are written to `output/{timestamp}/`:

| File | Contents |
|------|----------|
| `results.json` | Per-question results: question, expected/generated answer, accuracy, completeness, relevance, feedback, latency_seconds |
| `summary.csv` | Per-experiment: mean accuracy, completeness, relevance, overall score, mean/p50/p95 latency |
| `summary_scores.png` | Bar chart of mean scores by experiment |
| `by_question_type.png` | Mean scores by question type for the best experiment |

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `config.yaml` | All configuration (models, experiments, retrieval params) |
| `evaluation.py` | LLM-as-judge with latency tracking |
| `main.py` | Entry point: load questions, run experiments, save results |
| `plots.py` | Summary visualizations |

RAG logic (retrieval, query rewriting, reranking, answer generation) is reused from `02_advanced_rag_interactive_cli`.
