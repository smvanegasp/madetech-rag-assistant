# 03 Experiment: Advanced RAG Variants

Runs experiments comparing different combinations of query rewriting and re-ranking
to evaluate their impact on RAG quality.

## Experiment Variants

| Name | Query Rewriting | Re-ranking |
|------|-----------------|------------|
| `basic_rag` | No | No |
| `with_reranking` | No | Yes |
| `with_rewriting` | Yes | No |
| `with_rewriting_and_reranking` | Yes | Yes |

## Prerequisites

1. **00_questions_generation** — Run first to generate eval questions. Output goes to
   `00_questions_generation/output/{timestamp}/` with `test.jsonl` and `validation.jsonl`.

2. **01_llm_chunking_embedding** — Run to create the ChromaDB vector store.

## Configuration

Edit `config.yaml` to:

- **questions.source** — Path to `00_questions_generation/output` (relative to this script).
- **questions.use_latest** — If `true`, uses the most recent timestamp folder. If `false`, set `questions.timestamp` to a specific run (e.g. `2025-03-07_12-30-45`).
- **questions.files** — JSONL files to load (`test.jsonl`, `validation.jsonl`, or both).
- **base_config** — RAG parameters (vector_db, retrieval, model). Same structure as `02_advanced_rag/config.yaml`.
- **experiments** — List of experiment variants with `use_query_rewriting` and `use_reranking` flags.

## Running

From the repo root:

```bash
python -m experiments.scripts.03_experiment_advanced_rag.main
```

Results are written to `output/{timestamp}/` as one JSONL file per experiment:
`basic_rag.jsonl`, `with_reranking.jsonl`, `with_rewriting.jsonl`, `with_rewriting_and_reranking.jsonl`.

Each line contains: `question`, `expected_answer`, `question_type`, `model_answer` (and `error` if the run failed).
