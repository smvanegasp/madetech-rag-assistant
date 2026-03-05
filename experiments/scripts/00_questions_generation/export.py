"""Export eval records to JSONL, CSV, and Markdown."""

import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from utils.models import QAPairEvalRecord

logger = logging.getLogger(__name__)

# The three critique dimensions produced by the generator — used as column names
# in the DataFrame and as keys when mapping critiques to scores.
SCORE_DIMS = ["relevance", "standalone", "groundedness"]


def save_eval_jsonl(records: List[QAPairEvalRecord], path: Path) -> None:
    """Serialize all eval records to a JSONL file (one JSON object per line)."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.debug("Saved %d records -> %s", len(records), path)


def save_filtered_eval_jsonl(
    records: List[QAPairEvalRecord],
    path: Path,
    min_score: int = 4,
) -> List[QAPairEvalRecord]:
    """
    Keep only records where every critique dimension scores >= min_score, then save to JSONL.

    A record passes the filter only if ALL three dimensions (relevance,
    standalone, groundedness) meet the threshold — one weak dimension is enough
    to discard the whole QA pair.

    Returns the filtered list for downstream use (plots, stats).
    """
    filtered = [
        r for r in records if all(c.score >= min_score for c in r.critiques)
    ]
    save_eval_jsonl(filtered, path)
    logger.debug(
        "Filtered to %d / %d records (all dimensions >= %d)",
        len(filtered),
        len(records),
        min_score,
    )
    return filtered


def build_eval_dataframe(records: List[QAPairEvalRecord]) -> pd.DataFrame:
    """
    Build a tidy DataFrame from eval records — one row per question.

    Columns: id, question, answer, question_type, doc_title, doc_category,
             relevance, standalone, groundedness.
    """
    rows = []
    for i, record in enumerate(records):
        # Flatten the critiques list into a {dimension: score} mapping so each
        # dimension becomes its own column rather than a nested structure.
        score_map = {c.critique_type: c.score for c in record.critiques}
        doc_titles = " | ".join(m.title for m in record.doc_metadata)
        doc_categories = " | ".join(m.category for m in record.doc_metadata)
        rows.append({
            "id": i,
            "question": record.question,
            "answer": record.answer,
            "question_type": record.question_type,
            "doc_title": doc_titles,
            "doc_category": doc_categories,
            "relevance": score_map.get("relevance"),
            "standalone": score_map.get("standalone"),
            "groundedness": score_map.get("groundedness"),
        })
    df = pd.DataFrame(rows)
    logger.debug("Built DataFrame with shape %s", df.shape)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the row index."""
    df.to_csv(path, index=False)
    logger.debug(
        "Saved DataFrame (%d rows x %d cols) -> %s", df.shape[0], df.shape[1], path
    )


def format_eval_record_md(record: QAPairEvalRecord, id: int) -> str:
    """
    Render a single eval record as a Markdown block.

    Uses native Markdown (headings, tables, bold) so the file renders cleanly
    in any Markdown viewer. The 'ID: <n>' heading is Ctrl+F friendly.
    """
    # Visual star ratings so quality is immediately scannable in the Markdown file.
    SCORE_EMOJI = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}

    lines = []

    lines.append(f"## ID: {id}")
    lines.append("")

    lines.append(f"**Q:** {record.question}")
    lines.append("")
    lines.append(f"**A:** {record.answer}")
    lines.append("")
    doc_sources = ", ".join(
        f"{m.title} [{m.category}]" for m in record.doc_metadata
    )
    lines.append(
        f"**Type:** {record.question_type} &nbsp;|&nbsp; "
        f"**Source:** {doc_sources}"
    )
    lines.append("")

    # Summary table — quick overview of all three dimension scores at a glance.
    lines.append("| Dimension | Score | Rating |")
    lines.append("|-----------|:-----:|--------|")
    for c in record.critiques:
        lines.append(
            f"| {c.critique_type.capitalize()} "
            f"| {c.score} / 5 "
            f"| {SCORE_EMOJI.get(c.score, '')} |"
        )
    lines.append("")

    # Detailed rationale section — one block per dimension.
    for c in record.critiques:
        lines.append(f"**{c.critique_type.capitalize()}** — {c.score}/5")
        lines.append("")
        lines.append(f"> {c.rationale}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _stratified_split(
    records: List[QAPairEvalRecord],
    validation_ratio: float,
    rng: random.Random,
) -> Tuple[List[QAPairEvalRecord], List[QAPairEvalRecord]]:
    """
    Split records into validation and test while preserving the ratio of
    question_type values (single-source vs multi-source).

    Each stratum is shuffled and split independently, then the two validation
    halves are combined and re-shuffled, as are the two test halves.
    """
    strata: dict[str, List[QAPairEvalRecord]] = {}
    for r in records:
        strata.setdefault(r.question_type, []).append(r)

    validation: List[QAPairEvalRecord] = []
    test: List[QAPairEvalRecord] = []

    for group in strata.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        split_idx = round(len(shuffled) * validation_ratio)
        validation.extend(shuffled[:split_idx])
        test.extend(shuffled[split_idx:])

    rng.shuffle(validation)
    rng.shuffle(test)
    return validation, test


def save_dataset_split(
    records: List[QAPairEvalRecord],
    output_dir: Path,
    validation_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    """
    Split quality-filtered records into validation and test sets, preserving the
    proportion of single-source vs multi-source questions in each set.

    Each line in the output files contains only: question, answer, question_type,
    and doc_metadata (critiques and other evaluation fields are excluded).

    Returns (n_validation, n_test) for logging.
    """
    rng = random.Random(seed)
    validation_records, test_records = _stratified_split(records, validation_ratio, rng)

    def _serialise(r: QAPairEvalRecord) -> str:
        return json.dumps({
            "question": r.question,
            "answer": r.answer,
            "question_type": r.question_type,
            "doc_metadata": [m.model_dump() for m in r.doc_metadata],
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / "validation.jsonl"
    validation_path.write_text(
        "\n".join(_serialise(r) for r in validation_records), encoding="utf-8"
    )
    logger.debug("Saved %d validation records -> %s", len(validation_records), validation_path)

    test_path = output_dir / "test.jsonl"
    test_path.write_text(
        "\n".join(_serialise(r) for r in test_records), encoding="utf-8"
    )
    logger.debug("Saved %d test records -> %s", len(test_records), test_path)

    return len(validation_records), len(test_records)


def export_eval_markdown(records: List[QAPairEvalRecord], path: Path) -> None:
    """
    Export eval records to a Markdown file with proper Markdown formatting.

    Each record is a section headed '## ID: <n>' for easy Ctrl+F navigation.
    """
    content = "\n".join(format_eval_record_md(r, i) for i, r in enumerate(records))
    path.write_text(content, encoding="utf-8")
    logger.debug("Exported %d records -> %s", len(records), path)
