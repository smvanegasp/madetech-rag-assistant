# Rerank/Rewrite Experiment

Runs the four RAG variants (basic, +reranking, +rewriting, +both) against evaluation questions and scores each answer with an LLM-as-judge (accuracy, completeness, relevance).

## Experiment Variants

| Name | Query Rewriting | Re-ranking |
|------|-----------------|------------|
| `basic_rag` | No | No |
| `with_reranking` | No | Yes |
| `with_rewriting` | Yes | No |
| `with_rewriting_and_reranking` | Yes | Yes |

## Requirements

Install experiment dependencies (from repo root):

```bash
pip install -r experiments/requirements.txt
```

Or ensure `pandas`, `numpy`, `matplotlib`, `seaborn`, `tabulate`, `tqdm`, `litellm`, `chromadb`, `openai`, `tenacity`, `pyyaml`, `python-dotenv`, `pydantic` are available.

## Prerequisites

1. **00_questions_generation** — Run first to generate eval questions. Output goes to `00_questions_generation/output/{timestamp}/` with `validation.jsonl`.

2. **01_llm_chunking_embedding** — Run to create the ChromaDB vector store (or use existing `backend/data/vector_db`).

## Usage

From the repo root:

```bash
python -m experiments.scripts.03_experiment_rerank_rewrite.main
```

To load a previous run instead of re-running (for viewing results or re-generating plots):

Edit `config.yaml` and set `load_from` to the timestamp folder, e.g. `output/2026-03-08_12-00-00`.

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `questions.source` | Path to `00_questions_generation/output` | `../00_questions_generation/output` |
| `questions.use_latest` | Use most recent timestamp folder | `true` |
| `questions.timestamp` | Specific timestamp (e.g. `2025-03-07_12-30-45`) | — |
| `questions.files` | JSONL files to load | `["validation.jsonl"]` |
| `questions.n_questions` | Limit number of questions (null = all) | `15` |
| `questions.random_seed` | Seed for sampling when n_questions is set | `42` |
| `vector_db.path` | ChromaDB path (relative to this script) | `../../../backend/data/vector_db` |
| `vector_db.collection_name` | ChromaDB collection | `docs` |
| `embedding_model` | Must match 01_llm_chunking_embedding | `text-embedding-3-large` |
| `retrieval.retrieval_k` | Chunks per query | `20` |
| `retrieval.final_k` | Chunks passed to LLM after reranking | `10` |
| `model` | LLM for RAG and judge | `groq/openai/gpt-oss-20b` |
| `output` | Output directory (relative to script) | `output` |
| `load_from` | Load previous run (path to timestamp folder) | `null` |
| `save_plots` | Generate plots after run | `true` |

## Output

Results are written to `output/{timestamp}/`:

- **results.json** — Full per-question results per experiment (question, expected_answer, accuracy, completeness, relevance, feedback, generated_answer)
- **summary.csv** — Mean accuracy, completeness, relevance, overall_score per experiment
- **summary_scores.png** — Bar chart of mean scores by experiment
- **by_question_type.png** — Mean scores by question type for the best experiment

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `evaluation.py` | LLM-as-judge (evaluate_answer) |
| `main.py` | Entry point: load, run experiments, save |
| `plots.py` | Summary visualizations |
| `config.yaml` | All configuration |

RAG logic (retrieval, query rewriting, reranking, answer generation) is reused from `02_advanced_rag_interactive_cli`.
