# Single-Source Questions Generation

Generates evaluation question pairs from handbook documents for RAG system assessment. Each question is grounded in a single document, scored for relevance, standalone clarity, and groundedness.

## Structure

- **main.py** — Entry point: loads config, sets up logging, runs the pipeline, writes outputs
- **config.yaml** — Parameters (n_documents, seed, questions_per_document, quality cutoff, model, paths)
- **config.py** — Config dataclass and YAML loading
- **generator.py** — QA generation, critique, and eval dataset logic
- **export.py** — JSONL, CSV, and Markdown serialization
- **plots.py** — Bar charts and score distribution plots

## Configuration (config.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_documents` | Number of handbook documents to sample | 10 |
| `seed` | Random seed for document selection | 42 |
| `questions_per_document` | QA pairs per document | 3 |
| `min_filter_score` | Quality cutoff: keep records where every dimension >= this | 4 |
| `model` | LLM for QA generation and critique | groq/openai/gpt-oss-20b |
| `handbook_path` | Path to handbook (relative to project root) | backend/data/handbook |
| `env_path` | Path to .env for API keys | backend/.env |

## How to Run

From this directory:

```bash
cd research/evaluation/01_single_source_questions_generation
python main.py
```

## Logging

The pipeline uses Python's `logging` module combined with `tqdm` progress bars. Two output streams are configured at startup:

| Stream | Level | What you see |
|--------|-------|--------------|
| Console (stdout) | WARNING | Warnings and errors only (e.g. missing `.env`). Silent on a clean run. |
| File (`logs/{timestamp}.log`) | DEBUG | Everything: config values, per-doc generation, individual LLM critique calls, scores, file saves. |

Progress during the two long-running phases is shown via `tqdm` bars — that is the only console output on a clean run:

```
Generating QA pairs: 100%|████████| 3/3 [00:08<00:00,  2.7s/doc]
Critiquing QA pairs: 100%|██████| 9/9 [00:24<00:00,  2.7s/pair]
```

If anything goes wrong (missing API key, LLM error, etc.) a `WARNING` or `ERROR` line will appear above the progress bars. The full debug trail is always available in `logs/{timestamp}.log`.

## Output

Creates `output/{timestamp}/` with:

- `eval_questions.jsonl` — All generated records
- `eval_questions_filtered.jsonl` — Records passing quality cutoff (all dimensions >= min_filter_score)
- `eval_questions.csv` — Tabular format
- `eval_questions.md` — Human-readable Markdown
- `figures/` — `scores_before_after_bar.png`, `scores_by_category_before_after.png`, `scores_distribution.png`

Creates `logs/{timestamp}.log` with the full DEBUG-level log of the entire run.
