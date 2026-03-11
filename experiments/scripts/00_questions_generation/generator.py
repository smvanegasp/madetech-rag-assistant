"""QA generation and critique logic for single-source and multi-source eval questions."""

import logging
import random
from pathlib import Path
from typing import List

import numpy as np
from litellm import completion
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils.models import (
    HandbookDoc,
    HandbookDocMetadata,
    QAPairEvalRecord,
    QAPairList,
    QAPairWithTS,
    QuestionCritique,
    QuestionCritiqueWithType,
)
from utils.prompts import (
    GROUNDEDNESS_CRITIQUE_SYSTEM_PROMPT,
    QA_GENERATION_SYSTEM_PROMPT,
    QA_MULTI_SOURCE_GENERATION_SYSTEM_PROMPT,
    RELEVANCE_CRITIQUE_SYSTEM_PROMPT,
    STANDALONE_CRITIQUE_SYSTEM_PROMPT,
)

from config import MultiSourceConfig, SingleSourceConfig

logger = logging.getLogger(__name__)

# Maps each critique dimension name to its system prompt so critique_question()
# can look up the right prompt without a long if/elif chain.
CRITIQUE_PROMPTS = {
    "groundedness": GROUNDEDNESS_CRITIQUE_SYSTEM_PROMPT,
    "relevance": RELEVANCE_CRITIQUE_SYSTEM_PROMPT,
    "standalone": STANDALONE_CRITIQUE_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Single-source generation
# ---------------------------------------------------------------------------


def generate_qa_pairs_from_single_document(
    document: HandbookDoc,
    num_questions: int = 3,
    model: str = "groq/openai/gpt-oss-20b",
    max_tokens: int = 2048,
) -> List[QAPairWithTS]:
    """
    Generate search-style factoid QA pairs from a single document.

    Args:
        document: The document object.
        num_questions: Number of QA pairs to generate.
        model: Model name to use for LLM.
        max_tokens: Maximum output tokens for the completion call.

    Returns:
        List of QAPairWithTS objects with source document metadata.
    """
    logger.debug(
        "Generating %d QA pairs from doc '%s' (id=%s) using model '%s'",
        num_questions,
        document.title,
        document.id,
        model,
    )

    user_prompt = (
        f"The user has provided the following document:\n\n"
        f"[DOCUMENT BEGINS]\n\n{document.content}\n\n[DOCUMENT ENDS]\n\n\n"
        f"Generate {num_questions} question-answer pairs from the document.\n\n"
        "Reply only with the question-answer pairs, nothing else."
    )
    messages = [
        {"role": "system", "content": QA_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=model,
        messages=messages,
        response_format=QAPairList,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content

    pairs = QAPairList.model_validate_json(reply).pairs
    logger.info("Doc '%s': generated %d QA pairs", document.title, len(pairs))

    return [
        QAPairWithTS(
            **pair.model_dump(),
            question_type="single-source",
            doc_metadata=[
                HandbookDocMetadata(**document.model_dump(exclude={"content"}))
            ],
        )
        for pair in pairs
    ]


def generate_single_source_eval_dataset(
    documents: List[HandbookDoc],
    cfg: SingleSourceConfig,
    seed: int = 42,
    model: str = "groq/openai/gpt-oss-20b",
    max_tokens: int = 2048,
) -> List[QAPairEvalRecord]:
    """
    Sample n_documents documents and run the full single-source QA generation
    and critique pipeline.

    Args:
        documents: Full list of handbook documents to sample from.
        cfg: Single-source pipeline configuration.
        seed: Random seed for reproducible document sampling.
        model: LiteLLM model string for generation and critique.
        max_tokens: Maximum output tokens per QA generation call.

    Returns:
        List of QAPairEvalRecord instances with critiques across all dimensions.
    """
    random.seed(seed)
    sampled_docs = random.sample(documents, min(cfg.n_documents, len(documents)))
    logger.debug(
        "Sampled %d documents (seed=%d) for single-source eval generation",
        len(sampled_docs),
        seed,
    )

    doc_content_lookup = {doc.id: doc.content for doc in sampled_docs}

    # --- Step 1: generate QA pairs for every sampled document ---
    all_qa_pairs: List[QAPairWithTS] = []
    for doc in tqdm(sampled_docs, desc="[single] Generating QA pairs", unit="doc"):
        logger.debug("Generating QA pairs from doc '%s' (id=%s)", doc.title, doc.id)
        all_qa_pairs.extend(
            generate_qa_pairs_from_single_document(
                doc,
                num_questions=cfg.questions_per_document,
                model=model,
                max_tokens=max_tokens,
            )
        )
    logger.debug("Single-source total QA pairs generated: %d", len(all_qa_pairs))

    # --- Step 2: critique every QA pair across all three dimensions ---
    eval_records: List[QAPairEvalRecord] = []
    for qa_pair in tqdm(all_qa_pairs, desc="[single] Critiquing QA pairs", unit="pair"):
        logger.debug("Critiquing pair: '%s'", qa_pair.question[:80])
        doc_ids = [m.id for m in qa_pair.doc_metadata]
        context = "\n\n---\n\n".join(
            doc_content_lookup[doc_id] for doc_id in doc_ids
        )
        eval_records.append(
            QAPairEvalRecord(
                **qa_pair.model_dump(),
                critiques=critique_all_dimensions(context, qa_pair.question, model=model),
            )
        )

    logger.debug(
        "Single-source critique phase complete. %d eval records ready.", len(eval_records)
    )
    return eval_records


# ---------------------------------------------------------------------------
# Multi-source generation
# ---------------------------------------------------------------------------


def compute_or_load_embeddings(
    documents: List[HandbookDoc],
    cache_path: Path,
    embedding_model: str = "text-embedding-3-large",
) -> np.ndarray:
    """
    Return a (n_docs, embedding_dim) float array of document embeddings.

    If cache_path exists the vectors are loaded from disk to avoid re-computing
    them (which would incur an OpenAI API cost on every run). Otherwise the
    embeddings are computed via the OpenAI embeddings API and saved to
    cache_path for future runs.

    Args:
        documents: List of handbook documents whose content will be embedded.
        cache_path: Path to the .npy cache file.
        embedding_model: OpenAI embedding model name.

    Returns:
        NumPy array of shape (len(documents), embedding_dim).
    """
    if cache_path.exists():
        vectors = np.load(cache_path, allow_pickle=True)
        logger.debug(
            "Loaded %d embedding vectors from cache (%s)", len(vectors), cache_path
        )
        return vectors

    logger.debug(
        "Cache not found at %s — computing embeddings with model '%s'",
        cache_path,
        embedding_model,
    )
    client = OpenAI()
    contents = [doc.content for doc in documents]
    response = client.embeddings.create(model=embedding_model, input=contents)
    vectors = np.array([e.embedding for e in response.data])
    np.save(cache_path, vectors)
    logger.debug(
        "Computed and saved %d embedding vectors to %s", len(vectors), cache_path
    )
    return vectors


def build_document_groups(
    documents: List[HandbookDoc],
    vectors: np.ndarray,
    k: int = 7,
    min_sim: float = 0.3,
) -> List[List[HandbookDoc]]:
    """
    Group documents by cosine similarity so each group is suitable for
    generating multi-source questions.

    For each document (the anchor), the top-K most similar neighbours that
    exceed min_sim are collected. The anchor is placed first in its group.
    Only groups with at least 2 documents (anchor + 1 neighbour) are returned.

    Args:
        documents: Full list of handbook documents.
        vectors: Embedding matrix of shape (len(documents), embedding_dim).
        k: Number of top neighbours to consider for each anchor.
        min_sim: Minimum cosine similarity for a neighbour to be included.

    Returns:
        List of document groups; each group is a list of HandbookDoc objects
        with the anchor document at index 0.
    """
    X = np.array(vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norms
    sim_matrix = cosine_similarity(X_norm)

    doc_groups: List[List[HandbookDoc]] = []
    for i, anchor in enumerate(documents):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:k]
        neighbours = [documents[j] for j in top_idx if sims[j] >= min_sim]
        group = [anchor] + neighbours
        if len(group) >= 2:
            doc_groups.append(group)

    logger.debug(
        "Built %d document groups (K=%d, min_sim=%.2f)", len(doc_groups), k, min_sim
    )
    return doc_groups


def generate_qa_pairs_from_multiple_documents(
    documents: List[HandbookDoc],
    num_questions: int = 5,
    model: str = "groq/openai/gpt-oss-20b",
    two_docs_ratio: float = 0.8,
    max_tokens: int = 2048,
) -> List[QAPairWithTS]:
    """
    Generate multi-source question-answer pairs from a list of related documents.

    The first document in the list is the anchor. One additional document is
    selected 80% of the time (two_docs_ratio); otherwise two additional documents
    are selected, so each generated answer draws from 2 or 3 documents at most.

    Args:
        documents: Related documents; the first is the anchor.
        num_questions: Number of QA pairs to generate.
        model: LiteLLM model string.
        two_docs_ratio: Probability of using exactly 2 documents (anchor + 1).
        max_tokens: Maximum output tokens for the completion call.

    Returns:
        List of QAPairWithTS with question_type='multi-source' and a list of
        doc_metadata covering all documents used.
    """
    if len(documents) < 2:
        raise ValueError("Need at least 2 documents (anchor + at least one other).")

    anchor = documents[0]
    others = list(documents[1:])

    use_two_total = random.random() < two_docs_ratio
    n_extra = 1 if (use_two_total or len(others) < 2) else 2
    n_extra = min(n_extra, len(others))
    selected_others = random.sample(others, n_extra)

    selected_docs = [anchor] + selected_others
    doc_titles = [d.title for d in selected_docs]
    logger.debug(
        "Generating %d QA pairs from %d docs: %s | model='%s'",
        num_questions,
        len(selected_docs),
        doc_titles,
        model,
    )

    doc_metadata_list = [
        HandbookDocMetadata(**d.model_dump(exclude={"content"}))
        for d in selected_docs
    ]

    # Build combined context with clear document boundaries so the model can
    # distinguish which facts come from which source.
    doc_blocks = []
    for idx, doc in enumerate(selected_docs, start=1):
        header = f"========== DOCUMENT {idx} =========="
        doc_blocks.append(f"{header}\n\n{doc.content}")
    combined_context = "\n\n".join(doc_blocks)

    user_prompt = (
        f"The user has provided the following {len(selected_docs)} related documents "
        "from Made Tech's handbook:\n\n"
        f"[DOCUMENTS BEGIN]\n\n{combined_context}\n\n[DOCUMENTS END]\n\n\n"
        f"Generate {num_questions} question-answer pairs. Each question MUST require "
        "information from at least two of these documents to answer. "
        "Reply only with the question-answer pairs, nothing else."
    )
    messages = [
        {"role": "system", "content": QA_MULTI_SOURCE_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=model,
        messages=messages,
        response_format=QAPairList,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content

    pairs = QAPairList.model_validate_json(reply).pairs
    logger.info("Docs %s: generated %d QA pairs", doc_titles, len(pairs))

    return [
        QAPairWithTS(
            **pair.model_dump(),
            question_type="multi-source",
            doc_metadata=doc_metadata_list,
        )
        for pair in pairs
    ]


def generate_multi_source_eval_dataset(
    documents: List[HandbookDoc],
    doc_groups: List[List[HandbookDoc]],
    cfg: MultiSourceConfig,
    seed: int = 42,
    model: str = "groq/openai/gpt-oss-20b",
    max_tokens: int = 2048,
) -> List[QAPairEvalRecord]:
    """
    Sample n_groups document groups and run the full multi-source QA generation
    and critique pipeline.

    Args:
        documents: Full list of handbook documents (used for the content lookup).
        doc_groups: All available document groups (each a list of related docs).
        cfg: Multi-source pipeline configuration.
        seed: Random seed for reproducible group sampling.
        model: LiteLLM model string for generation and critique.
        max_tokens: Maximum output tokens per QA generation call.

    Returns:
        List of QAPairEvalRecord instances with critiques across all dimensions.
    """
    random.seed(seed)
    sampled_groups = random.sample(doc_groups, min(cfg.n_groups, len(doc_groups)))
    logger.debug(
        "Sampled %d document groups (seed=%d) for multi-source eval generation",
        len(sampled_groups),
        seed,
    )

    doc_content_lookup = {doc.id: doc.content for doc in documents}

    # --- Step 1: generate QA pairs for every sampled document group ---
    all_qa_pairs: List[QAPairWithTS] = []
    for group in tqdm(sampled_groups, desc="[multi] Generating QA pairs", unit="group"):
        anchor_title = group[0].title
        logger.debug("Generating QA pairs from group anchored at '%s'", anchor_title)
        all_qa_pairs.extend(
            generate_qa_pairs_from_multiple_documents(
                group,
                num_questions=cfg.questions_per_group,
                model=model,
                two_docs_ratio=cfg.two_docs_ratio,
                max_tokens=max_tokens,
            )
        )
    logger.debug("Multi-source total QA pairs generated: %d", len(all_qa_pairs))

    # --- Step 2: critique every QA pair across all three dimensions ---
    eval_records: List[QAPairEvalRecord] = []
    for qa_pair in tqdm(all_qa_pairs, desc="[multi] Critiquing QA pairs", unit="pair"):
        logger.debug("Critiquing pair: '%s'", qa_pair.question[:80])
        # Concatenate all source documents with a separator so the critique LLM
        # sees the full combined context.
        context = "\n\n---\n\n".join(
            doc_content_lookup[m.id] for m in qa_pair.doc_metadata
        )
        eval_records.append(
            QAPairEvalRecord(
                **qa_pair.model_dump(),
                critiques=critique_all_dimensions(context, qa_pair.question, model=model),
            )
        )

    logger.debug(
        "Multi-source critique phase complete. %d eval records ready.", len(eval_records)
    )
    return eval_records


# ---------------------------------------------------------------------------
# Shared critique logic
# ---------------------------------------------------------------------------


def critique_question(
    question: str,
    critique_type: str,
    model: str = "groq/openai/gpt-oss-20b",
    context: str | None = None,
) -> QuestionCritiqueWithType:
    """
    Critique a question using a specified evaluation dimension.

    Args:
        question: The question to evaluate.
        critique_type: One of "groundedness", "relevance", or "standalone".
        model: Model name to use for the LLM call.
        context: The source document text (may be a concatenation of multiple docs).
            Required for "groundedness"; omitted for "relevance" and "standalone"
            to avoid biasing those evaluations.

    Returns:
        A QuestionCritiqueWithType with rationale, score, and critique type.
    """
    logger.debug(
        "Critiquing question on dimension '%s' using model '%s' | Q: %.80s…",
        critique_type,
        model,
        question,
    )

    system_prompt = CRITIQUE_PROMPTS[critique_type]
    user_prompt = (
        f"The user has provided the following question:\n\n"
        f"[QUESTION BEGINS]\n\n{question}\n\n[QUESTION ENDS]\n\n"
    )
    if context is not None:
        user_prompt += (
            f"The user has provided the following context:\n\n"
            f"[CONTEXT BEGINS]\n\n{context}\n\n[CONTEXT ENDS]\n\n\n"
        )
    user_prompt += "Reply only with your rationale for the rating and your score (1-5), nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=model, messages=messages, response_format=QuestionCritique
    )
    reply = response.choices[0].message.content

    result = QuestionCritiqueWithType(
        **QuestionCritique.model_validate_json(reply).model_dump(),
        critique_type=critique_type,
    )
    logger.debug("Dimension '%s' scored %d/5", critique_type, result.score)
    return result


def critique_all_dimensions(
    context: str,
    question: str,
    model: str = "groq/openai/gpt-oss-20b",
) -> List[QuestionCritiqueWithType]:
    """
    Critique a question across all supported evaluation dimensions.

    "relevance" and "standalone" receive only the question to avoid biasing those
    evaluations with the source context. "groundedness" requires the context to
    assess how well the question can be answered from the retrieved documents.

    Args:
        context: The source document text (may be a concatenation of multiple docs).
        question: The question to evaluate.
        model: Model name to use for the LLM call.

    Returns:
        A list of QuestionCritiqueWithType instances, one per critique dimension.
    """
    context_by_dimension: dict[str, str | None] = {
        "relevance": None,
        "standalone": None,
        "groundedness": context,
    }
    critiques = [
        critique_question(question, crit_type, model=model, context=ctx)
        for crit_type, ctx in context_by_dimension.items()
    ]
    scores = {c.critique_type: c.score for c in critiques}
    logger.debug("All dimensions scored: %s", scores)
    return critiques
